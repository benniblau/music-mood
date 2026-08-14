"""Scoring and selection.

The bug these exist to prevent is not a crash -- it is a plausible-looking
playlist that is quietly wrong. That happened once already: an un-normalised
vocabulary term outweighed the axes and put a romantic ballad at the top of a
request for "dark and tense".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from moodlib import config, db, ontology, query


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "q.sqlite3")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    connection = db.connect(tmp_path / "q.sqlite3")
    yield connection
    connection.close()


def add_track(conn, path, *, artist="A", title="T", adjectives=(), contexts=(),
              confidence=2, **scores):
    key = db.content_key(artist, "Album", title, "", "2020")
    cur = conn.execute(
        "INSERT INTO tracks (path, artist, album, title, genre_raw, year, "
        "duration, content_key) VALUES (?,?,?,?,?,?,?,?)",
        (path, artist, "Album", title, "", "2020", 200.0, key))
    track_id = cur.lastrowid
    row = {name: 0.5 for name in ontology.AXES}
    row.update({name: 50.0 for name in ontology.GEMS})
    row.update(scores)
    row.update({
        "adjectives_json": json.dumps(list(adjectives)),
        "contexts_json": json.dumps(list(contexts)),
        "discogs_genre": "Electronic", "discogs_style": "Techno",
        "confidence": confidence, "model": "test",
        "ontology_version": ontology.ONTOLOGY_VERSION,
        "content_key": key, "tagged_at": "now", "error": None,
    })
    db.write_mood(conn, track_id, row)
    conn.commit()


def axis(target, weight=1.0):
    return {"target": target, "weight": weight}


def test_axis_target_ranks_the_closer_track_first(conn):
    add_track(conn, "loud.m4a", title="Loud", energy=0.95)
    add_track(conn, "quiet.m4a", title="Quiet", energy=0.10)
    scored = query.score_all(conn, {"energy": axis(100)})
    assert [s.title for s in scored] == ["Loud", "Quiet"]


def test_zero_weight_axis_is_ignored(conn):
    # "Weight 0 means ignore" has to be literally true, or every unmentioned
    # dimension silently drags every track toward its target.
    add_track(conn, "a.m4a", title="A", energy=0.0, valence=0.9)
    add_track(conn, "b.m4a", title="B", energy=1.0, valence=0.9)
    scored = query.score_all(conn, {"energy": axis(100, 0.0), "valence": axis(90)})
    assert scored[0].score == pytest.approx(scored[1].score)


def test_gems_factor_targets_its_member_dimensions(conn):
    # A query may set `unease` instead of enumerating tension and sadness.
    add_track(conn, "uneasy.m4a", title="Uneasy", tension=90.0, sadness=90.0)
    add_track(conn, "calm.m4a", title="Calm", tension=5.0, sadness=5.0)
    scored = query.score_all(conn, {"unease": axis(100)})
    assert [s.title for s in scored] == ["Uneasy", "Calm"]


def test_vocabulary_is_a_tiebreaker_not_the_ranking(conn):
    # The live regression: a track matching one favoured adjective must not
    # overtake a track the axes and emotions agree is a much better fit.
    add_track(conn, "dark.m4a", title="Dark", tension=95.0, valence=0.05,
              adjectives=["dark", "tense"])
    add_track(conn, "ballad.m4a", title="Ballad", tension=5.0, valence=0.8,
              adjectives=["melancholic", "romantic"])
    scored = query.score_all(conn, {
        "tension": axis(95), "valence": axis(5),
        "adjectives": ["melancholic", "nostalgic", "cinematic", "wistful", "dreamy"],
    })
    assert scored[0].title == "Dark"


def test_avoided_adjective_penalises(conn):
    add_track(conn, "a.m4a", title="Harsh", adjectives=["aggressive"])
    add_track(conn, "b.m4a", title="Soft", adjectives=["mellow"])
    scored = query.score_all(conn, {"energy": axis(50), "avoid": ["aggressive"]})
    assert scored[0].title == "Soft"


def test_near_miss_scores_through_the_parent_dimension(conn):
    # "yearning" and "wistful" are different words for the same region; layer 3
    # exists so that difference does not cost a good track.
    add_track(conn, "near.m4a", title="Near", nostalgia=100.0, adjectives=["wistful"])
    add_track(conn, "far.m4a", title="Far", nostalgia=0.0, adjectives=["punchy"])
    scored = query.score_all(conn, {"energy": axis(50), "adjectives": ["yearning"]})
    assert scored[0].title == "Near"
    assert scored[0].parts["vocab"] > 0


def test_confidence_breaks_an_otherwise_exact_tie(conn):
    add_track(conn, "known.m4a", title="Known", energy=0.5, confidence=2)
    add_track(conn, "guessed.m4a", title="Guessed", energy=0.5, confidence=0)
    scored = query.score_all(conn, {"energy": axis(50)})
    assert [s.title for s in scored] == ["Known", "Guessed"]


def test_min_confidence_filters(conn):
    add_track(conn, "known.m4a", title="Known", confidence=2)
    add_track(conn, "guessed.m4a", title="Guessed", confidence=0)
    assert len(query.score_all(conn, {"energy": axis(50)}, min_confidence=2)) == 1


def test_selection_caps_tracks_per_artist(conn):
    for n in range(10):
        add_track(conn, f"same{n}.m4a", artist="Prolific", title=f"T{n}")
    for n in range(10):
        add_track(conn, f"other{n}.m4a", artist=f"Other{n}", title=f"O{n}")
    scored = query.score_all(conn, {"energy": axis(50)})

    chosen = query.select(scored, count=8, max_per_artist=2, seed=1)
    counts: dict[str, int] = {}
    for item in chosen:
        counts[item.artist] = counts.get(item.artist, 0) + 1
    assert len(chosen) == 8
    assert max(counts.values()) <= 2


def test_selection_varies_between_runs(conn):
    # Asking for the same mood twice should not hand back an identical playlist.
    for n in range(40):
        add_track(conn, f"t{n}.m4a", artist=f"Artist{n}", title=f"T{n}",
                  energy=0.5 + n * 0.005)
    scored = query.score_all(conn, {"energy": axis(60)})
    first = [s.path for s in query.select(scored, 10, max_per_artist=2, seed=1)]
    second = [s.path for s in query.select(scored, 10, max_per_artist=2, seed=2)]
    assert first != second


def test_selection_is_reproducible_with_a_seed(conn):
    for n in range(40):
        add_track(conn, f"t{n}.m4a", artist=f"Artist{n}", title=f"T{n}")
    scored = query.score_all(conn, {"energy": axis(50)})
    a = [s.path for s in query.select(scored, 10, max_per_artist=2, seed=7)]
    b = [s.path for s in query.select(scored, 10, max_per_artist=2, seed=7)]
    assert a == b


# --- playlist file -----------------------------------------------------------

def test_filename_keeps_the_title_readable(conn):
    """Music.app names the playlist after the file, so the file must read well.

    The regression: slugging this like a URL turned "Sunny Electric Dreams" into
    "Sunny-Electric-Dreams" in the user's sidebar. Spaces are legal in filenames
    everywhere this runs; replacing them corrupts the only thing the generated
    title exists to produce.
    """
    from moodlib import playlist
    assert playlist.filename_for("Sunny Electric Dreams") == "Sunny Electric Dreams"
    assert playlist.filename_for("Rainy Morning Wistfulness") == "Rainy Morning Wistfulness"
    assert playlist.filename_for("Röyksopp Sundays") == "Röyksopp Sundays"
    assert playlist.filename_for("Hands In The Air!! 3am") == "Hands In The Air!! 3am"


def test_filename_strips_only_what_a_filesystem_cannot_carry(conn):
    from moodlib import playlist
    assert "/" not in playlist.filename_for("Dub / Bass")
    assert ":" not in playlist.filename_for("Late Night: Vol 2")
    # Removing the separator must not weld the words together.
    assert playlist.filename_for("Dub / Bass") == "Dub  Bass".replace("  ", " ")
    assert playlist.filename_for(".hidden") == "hidden"
    assert playlist.filename_for("trailing dots...") == "trailing dots"
    assert playlist.filename_for("") == "playlist"
    assert playlist.filename_for("///") == "playlist"


def test_rendered_playlist_carries_the_title_and_absolute_paths(conn):
    from moodlib import playlist
    add_track(conn, "Artist/Album/01 One.m4a", artist="Artist", title="One")
    scored = query.score_all(conn, {"energy": axis(50)})
    text = playlist.render(scored, library=Path("/music"), title="Quiet Hours")

    lines = text.splitlines()
    assert lines[0] == "#EXTM3U"
    assert lines[1] == "#PLAYLIST:Quiet Hours"
    # "Artist - Title" with a plain hyphen is the M3U convention.
    assert lines[2] == "#EXTINF:200,Artist - One"
    assert lines[3] == "/music/Artist/Album/01 One.m4a"


def test_playlist_without_a_title_omits_the_directive(conn):
    from moodlib import playlist
    add_track(conn, "Artist/Album/01 One.m4a")
    scored = query.score_all(conn, {"energy": axis(50)})
    assert "#PLAYLIST" not in playlist.render(scored, library=Path("/music"))


def test_path_prefix_rewrites_the_library_root(conn):
    # For the eventual server deployment, where the same files sit under a
    # different mount point.
    from moodlib import playlist
    add_track(conn, "Artist/Album/01 One.m4a")
    scored = query.score_all(conn, {"energy": axis(50)})
    text = playlist.render(scored, library=Path("/music"), path_prefix="/srv/media")
    assert "/srv/media/Artist/Album/01 One.m4a" in text


def test_playlist_never_lists_the_same_recording_twice(conn):
    """The library holds the same song at several paths; a playlist must not.

    Re-rips, compilation appearances and Music re-adding a track it already had
    all produce identical tags at different paths. They score identically, so
    they land next to each other -- "Modestep — Sunlight" twice in a row.
    """
    for n in range(4):
        add_track(conn, f"dupe{n}.m4a", artist="Modestep", title="Sunlight")
    for n in range(6):
        add_track(conn, f"other{n}.m4a", artist=f"Other{n}", title=f"Song{n}")
    scored = query.score_all(conn, {"energy": axis(50)})

    chosen = query.select(scored, count=6, max_per_artist=2, seed=1)
    titles = [(s.artist, s.title) for s in chosen]
    assert len(titles) == len(set(titles))
    assert titles.count(("Modestep", "Sunlight")) == 1


def test_different_versions_are_not_treated_as_duplicates(conn):
    # "Drive" and "Drive (Acoustic)" are different recordings and may both
    # legitimately appear -- no suffix-stripping cleverness.
    add_track(conn, "a.m4a", artist="Incubus", title="Drive")
    add_track(conn, "b.m4a", artist="Incubus", title="Drive (Acoustic)")
    scored = query.score_all(conn, {"energy": axis(50)})
    chosen = query.select(scored, count=2, max_per_artist=2, seed=1)
    assert len(chosen) == 2
