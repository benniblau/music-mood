"""SQLite schema, connection, and the identity model.

A track has three identifiers, and knowing which does what is the whole design:

    id           surrogate integer, stable forever. Everything references this.
    (device,inode)  the file's identity on the volume. Survives rename AND move,
                 including a rename Music performs *because* a tag changed.
    content_key  hash of (artist, album, title, genre, year). Survives the file
                 being rewritten or re-added, which changes the inode.

`path` is deliberately NOT an identifier -- it is a mutable attribute. Music
renames files from their tags, so a path is the least stable thing about a track.
An earlier version of this schema used it as the primary key; a tag edit then
changed both the path and the content_key at once, and the track was
indistinguishable from a deletion plus an unrelated new file. It lost its row,
its history, and anything referencing it.

The two identity signals are complementary and cover each other's blind spot:

    renamed, tags unchanged   inode same, content_key same   -> either matches
    renamed BY a tag edit     inode same, content_key NEW    -> inode matches
    rewritten / re-added      inode NEW, content_key same    -> content_key matches
    genuinely gone            neither matches                -> missing
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

from moodlib import config, ontology

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    device        INTEGER,
    inode         INTEGER,
    artist        TEXT NOT NULL DEFAULT '',
    album         TEXT NOT NULL DEFAULT '',
    album_artist  TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    genre_raw     TEXT NOT NULL DEFAULT '',
    year          TEXT NOT NULL DEFAULT '',
    duration      REAL,
    size          INTEGER,
    mtime         REAL,
    camelot_key   TEXT NOT NULL DEFAULT '',
    content_key   TEXT NOT NULL,
    missing_since TEXT
);
CREATE INDEX IF NOT EXISTS tracks_content_key ON tracks(content_key);
CREATE INDEX IF NOT EXISTS tracks_inode       ON tracks(inode);
CREATE INDEX IF NOT EXISTS tracks_missing     ON tracks(missing_since);

CREATE TABLE IF NOT EXISTS moods (
    track_id          INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    energy            REAL, valence          REAL, danceability REAL,
    acousticness      REAL, instrumentalness REAL,
    wonder            REAL, transcendence REAL, nostalgia         REAL,
    tenderness        REAL, peacefulness  REAL, joyful_activation REAL,
    power             REAL, tension       REAL, sadness           REAL,
    adjectives_json   TEXT NOT NULL DEFAULT '[]',
    contexts_json     TEXT NOT NULL DEFAULT '[]',
    discogs_genre     TEXT NOT NULL DEFAULT '',
    discogs_style     TEXT NOT NULL DEFAULT '',
    confidence        INTEGER,
    model             TEXT NOT NULL DEFAULT '',
    ontology_version  INTEGER NOT NULL DEFAULT 0,
    content_key       TEXT NOT NULL DEFAULT '',
    tagged_at         TEXT,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    query_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

#: Columns written by the tagger, in ontology order. One list, so the INSERT and
#: the scoring read can never disagree about what exists.
AXIS_COLUMNS = tuple(ontology.AXES)
GEMS_COLUMNS = tuple(ontology.GEMS)


def connect(path: Path | None = None) -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL keeps a long tagging run from blocking a concurrent `stats` read, and
    # survives the Ctrl-C that a multi-hour job invites.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate_from_path_keyed(conn)
    conn.executescript(SCHEMA)
    _repair_derived_genres(conn)
    return conn


def _migrate_from_path_keyed(conn: sqlite3.Connection) -> None:
    """Upgrade a database whose tracks were keyed by path.

    Preserves existing mood rows by joining on the old path key, so an upgrade
    does not throw away tagging that has already been paid for.
    """
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "tracks" not in tables:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
    if "id" in columns:
        return

    conn.executescript("""
        ALTER TABLE tracks RENAME TO tracks_old;
        ALTER TABLE moods  RENAME TO moods_old;
    """)
    conn.executescript(SCHEMA)
    conn.execute("""
        INSERT INTO tracks (path, artist, album, album_artist, title, genre_raw,
                            year, duration, size, mtime, camelot_key,
                            content_key, missing_since)
        SELECT path, artist, album, album_artist, title, genre_raw, year,
               duration, size, mtime, camelot_key, content_key, missing_since
        FROM tracks_old
    """)
    mood_columns = [row[1] for row in conn.execute("PRAGMA table_info(moods_old)")
                    if row[1] != "path"]
    conn.execute(f"""
        INSERT INTO moods (track_id, {','.join(mood_columns)})
        SELECT t.id, {','.join('m.' + c for c in mood_columns)}
        FROM moods_old m JOIN tracks t ON t.path = m.path
    """)
    conn.executescript("DROP TABLE moods_old; DROP TABLE tracks_old;")
    conn.commit()


def _repair_derived_genres(conn: sqlite3.Connection) -> None:
    """Bring stored genres back in line with the style they are derived from.

    The tagger used to ask the model for genre and style as two independent
    fields, and the model duly produced pairs the Discogs taxonomy does not
    contain -- minimal techno under `Brass & Military`, ~900 drum & bass tracks
    under `Jazz`. On the reference library 8,162 rows (35%) disagreed with their
    own style.

    The style was almost always the reliable half, so this recomputes the genre
    from it rather than re-tagging: seconds instead of hours, and no model call.
    A style the taxonomy no longer contains cannot be repaired that way, so those
    rows are marked stale and picked up by the next `tag` run.

    Idempotent and self-limiting -- after one pass the WHERE clauses match
    nothing, so leaving it on the connect path costs a single scan.
    """
    try:
        rows = list(conn.execute(
            "SELECT DISTINCT discogs_style, discogs_genre FROM moods "
            "WHERE error IS NULL AND discogs_style != ''"))
    except sqlite3.OperationalError:
        return  # pre-migration schema; nothing to repair yet

    corrections: list[tuple[str, str, str]] = []
    orphaned: list[str] = []
    for row in rows:
        derived = ontology.genre_for_style(row["discogs_style"])
        if not derived:
            orphaned.append(row["discogs_style"])
        elif derived != row["discogs_genre"]:
            corrections.append((derived, row["discogs_style"], row["discogs_genre"]))

    if corrections:
        conn.executemany(
            "UPDATE moods SET discogs_genre = ? "
            "WHERE discogs_style = ? AND discogs_genre = ?", corrections)
    if orphaned:
        # Emptying content_key is what makes stale_where() select these: it can
        # no longer equal the track's key, so the next tag run re-does them.
        conn.executemany(
            "UPDATE moods SET content_key = '' WHERE discogs_style = ?",
            [(style,) for style in orphaned])
    if corrections or orphaned:
        conn.commit()


def content_key(artist: str, album: str, title: str, genre: str, year: str) -> str:
    """Stable hash of the tags that define a track's identity for mood purposes.

    Unicode is normalised to NFC first: macOS stores filenames -- and, via
    round-trips through Music, some tags -- in NFD, so the same text can arrive
    with combining marks split out. Without normalising, an unchanged track can
    look changed and get re-tagged for nothing.
    """
    parts = (artist, album, title, genre, year)
    joined = "\x1f".join(unicodedata.normalize("NFC", (p or "").strip()) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def stale_where(alias: str = "t") -> str:
    """SQL predicate selecting tracks that need (re)tagging.

    Stale means never tagged, tagged against different tags, or tagged against an
    older ontology. Errored rows are retried too -- a failure is not a result.
    """
    return f"""(
        m.track_id IS NULL
        OR m.error IS NOT NULL
        OR m.content_key != {alias}.content_key
        OR m.ontology_version < :ontology_version
    )"""


def stale_tracks(conn: sqlite3.Connection, limit: int | None = None,
                 target: str | None = None) -> list[sqlite3.Row]:
    sql = f"""
        SELECT t.* FROM tracks t
        LEFT JOIN moods m ON m.track_id = t.id
        WHERE t.missing_since IS NULL AND {stale_where()}
    """
    params: dict = {"ontology_version": ontology.ONTOLOGY_VERSION}
    if target:
        sql += " AND t.path LIKE :target"
        params["target"] = f"{target}%"
    sql += " ORDER BY t.path"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql, params))


def write_mood(conn: sqlite3.Connection, track_id: int, row: dict) -> None:
    """Upsert one mood row, keyed by the track's stable id.

    Keyed by id rather than path, so a rename never has to rewrite this table --
    which is the point of the surrogate key.
    """
    columns = (
        "track_id", *AXIS_COLUMNS, *GEMS_COLUMNS,
        "adjectives_json", "contexts_json", "discogs_genre", "discogs_style",
        "confidence", "model", "ontology_version", "content_key", "tagged_at",
        "error",
    )
    values = [track_id] + [row.get(c) for c in columns[1:]]
    placeholders = ",".join("?" * len(columns))
    conn.execute(
        f"INSERT OR REPLACE INTO moods ({','.join(columns)}) VALUES ({placeholders})",
        values,
    )


def write_error(conn: sqlite3.Connection, track_id: int, message: str) -> None:
    """Record a tagging failure rather than silently dropping the track."""
    conn.execute(
        "INSERT OR REPLACE INTO moods (track_id, error, tagged_at, ontology_version) "
        "VALUES (?, ?, datetime('now'), ?)",
        (track_id, message[:400], ontology.ONTOLOGY_VERSION),
    )


def record_query(conn: sqlite3.Connection, text: str, query: dict) -> int:
    cur = conn.execute(
        "INSERT INTO queries (text, query_json, created_at) "
        "VALUES (?, ?, datetime('now'))",
        (text, json.dumps(query, ensure_ascii=False)),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Breakdown of what a `tag` run would do, so cost is visible before starting."""
    version = ontology.ONTOLOGY_VERSION
    scalar = lambda sql, **kw: int(conn.execute(sql, kw).fetchone()[0])
    return {
        "tracks": scalar("SELECT COUNT(*) FROM tracks"),
        "missing": scalar("SELECT COUNT(*) FROM tracks WHERE missing_since IS NOT NULL"),
        "tagged": scalar(
            "SELECT COUNT(*) FROM moods m JOIN tracks t ON t.id = m.track_id "
            "WHERE m.error IS NULL AND m.content_key = t.content_key "
            "AND m.ontology_version >= :v", v=version),
        "never_tagged": scalar(
            "SELECT COUNT(*) FROM tracks t LEFT JOIN moods m ON m.track_id = t.id "
            "WHERE t.missing_since IS NULL AND m.track_id IS NULL"),
        "stale_content": scalar(
            "SELECT COUNT(*) FROM tracks t JOIN moods m ON m.track_id = t.id "
            "WHERE t.missing_since IS NULL AND m.error IS NULL "
            "AND m.content_key != t.content_key"),
        "stale_ontology": scalar(
            "SELECT COUNT(*) FROM tracks t JOIN moods m ON m.track_id = t.id "
            "WHERE t.missing_since IS NULL AND m.error IS NULL "
            "AND m.content_key = t.content_key AND m.ontology_version < :v", v=version),
        "errors": scalar("SELECT COUNT(*) FROM moods WHERE error IS NOT NULL"),
    }


def iter_scored(conn: sqlite3.Connection, min_confidence: int = 0,
                genres: Sequence[str] = (), styles: Sequence[str] = (),
                year_from: str | None = None, year_to: str | None = None
                ) -> Iterable[sqlite3.Row]:
    """Every tagged, present track eligible for a playlist, with its scores.

    Genres and styles are lists because a request can name more than one, and
    they are OR-ed within a kind but AND-ed across: asking for the Hip Hop genre
    with the Boom Bap and Gangsta styles means "hip hop, of those two kinds"."""
    sql = """
        SELECT t.path, t.artist, t.title, t.album, t.year, t.duration, m.*
        FROM moods m JOIN tracks t ON t.id = m.track_id
        WHERE t.missing_since IS NULL
          AND m.error IS NULL
          AND m.content_key = t.content_key
          AND m.confidence >= :min_confidence
    """
    params: dict = {"min_confidence": min_confidence}
    # Genre and style are OR-ed, not AND-ed, because a style already implies its
    # genre -- they name regions of one taxonomy, not independent axes. AND-ing
    # them reads "Hip Hop *and* Boom Bap", which for "old-school legendary hip
    # hop" cut a 2,300-track genre down to 19 candidates: every classic tagged
    # Gangsta or Conscious was excluded by a style list meant to describe it.
    clauses, holes = [], {}
    if genres:
        names = ",".join(f":g{n}" for n in range(len(genres)))
        clauses.append(f"m.discogs_genre IN ({names})")
        holes.update({f"g{n}": value for n, value in enumerate(genres)})
    if styles:
        names = ",".join(f":s{n}" for n in range(len(styles)))
        clauses.append(f"m.discogs_style IN ({names})")
        holes.update({f"s{n}": value for n, value in enumerate(styles)})
    if clauses:
        sql += " AND (" + " OR ".join(clauses) + ")"
        params.update(holes)
    if year_from:
        sql += " AND t.year != '' AND t.year >= :year_from"
        params["year_from"] = year_from
    if year_to:
        sql += " AND t.year != '' AND t.year <= :year_to"
        params["year_to"] = year_to
    return conn.execute(sql, params)
