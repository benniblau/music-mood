"""A small Flask front end: scan, tag, and make playlists.

Deliberately thin. Every route is a call into `moodlib` and a template render --
no logic lives here that the CLI does not already have, so the two cannot drift.

The one idea worth explaining is how "another take" works. A playlist is not
stored; it is *derived* from a saved query plus a seed, and the URL carries both:

    /playlist/42?seed=7

`queries` already records the structured translation of every request, so a
re-roll re-scores from the stored JSON with a different seed. That makes it
instant (no second LLM call), reproducible (the same URL always gives the same
playlist), and shareable -- and it means the server holds no per-user state at
all. Asking for a genuinely fresh reading of the same words is a separate button
that does call the model again.
"""
from __future__ import annotations

import io
import random
import re
import sys
from pathlib import Path

from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_file, url_for)
from mutagen import File as MutagenFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moodlib import config, db, playlist as m3u, query  # noqa: E402
from webapp import jobs  # noqa: E402

app = Flask(__name__)
app.secret_key = config.WEB_SECRET_KEY

PLACEHOLDER = (Path(__file__).resolve().parent / "static" / "img" / "no-cover.svg")


# --------------------------------------------------------------------------
# library status and jobs
# --------------------------------------------------------------------------

@app.route("/")
def index():
    conn = db.connect()
    counts = db.counts(conn)
    # One row per distinct request. The queries table records every run --
    # including CLI and eval runs -- so the raw list is mostly repeats of
    # whatever was last being worked on.
    recent = list(conn.execute("""
        SELECT MAX(id) AS id, text, MAX(created_at) AS created_at
        FROM queries GROUP BY text ORDER BY id DESC LIMIT :limit""",
        {"limit": config.WEB_RECENT_QUERIES}))
    conn.close()
    counts["pending"] = (counts["never_tagged"] + counts["stale_content"]
                         + counts["stale_ontology"] + counts["errors"])
    return render_template("index.html", counts=counts, recent=recent,
                           library=config.LIBRARY_PATH,
                           sizes=config.WEB_PLAYLIST_SIZES,
                           default_size=config.PLAYLIST_SIZE,
                           poll_ms=int(config.WEB_POLL_SECONDS * 1000))


@app.get("/healthz")
def healthz():
    """Liveness, plus the two things that actually break on a server.

    A process that is up but pointed at an unmounted share, or at a database
    nothing has been tagged into, is not healthy in any sense a monitor cares
    about -- and those are the normal failures here, not crashes. Answering 503
    for them is what makes `systemctl status` and an uptime check disagree
    usefully rather than both saying "running".
    """
    conn = db.connect()
    counts = db.counts(conn)
    conn.close()
    mounted = config.LIBRARY_PATH.exists()
    payload = {
        "ok": mounted and counts["tagged"] > 0,
        "library_mounted": mounted,
        "tracks": counts["tracks"],
        "tagged": counts["tagged"],
        "job": (jobs.current().as_dict() if jobs.current() else None),
    }
    return payload, (200 if payload["ok"] else 503)


@app.post("/jobs/<name>")
def start_job(name: str):
    ok, message = jobs.start(name)
    flash(message, "info" if ok else "warning")
    return redirect(url_for("index"))


@app.get("/jobs/status")
def job_status():
    """Polled by the page while a job runs."""
    job = jobs.current()
    if job is None:
        return {"idle": True}
    conn = db.connect()
    try:
        payload = job.as_dict(conn)
        payload["idle"] = False
        payload["counts"] = db.counts(conn)
        return payload
    finally:
        conn.close()


# --------------------------------------------------------------------------
# playlists
# --------------------------------------------------------------------------

@app.post("/playlist")
def create_playlist():
    text = (request.form.get("mood") or "").strip()
    if not text:
        flash("Describe a mood first", "warning")
        return redirect(url_for("index"))

    conn = db.connect()
    if db.counts(conn)["tagged"] == 0:
        conn.close()
        flash("Nothing is tagged yet — run a scan, then tagging", "warning")
        return redirect(url_for("index"))
    try:
        structured = query.translate(text)
    except Exception as exc:                      # noqa: BLE001 - shown to the user
        conn.close()
        flash(f"Could not reach the model: {exc}", "warning")
        return redirect(url_for("index"))
    query_id = db.record_query(conn, text, structured)
    conn.close()
    return redirect(url_for("show_playlist", query_id=query_id,
                            n=request.form.get("count", type=int)))


@app.get("/playlist/<int:query_id>")
def show_playlist(query_id: int):
    import json

    seed = request.args.get("seed", type=int)
    if seed is None:
        seed = random.randint(1, 10_000)
    count = request.args.get("n", type=int) or config.PLAYLIST_SIZE

    conn = db.connect()
    row = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
    if row is None:
        conn.close()
        abort(404)
    structured = json.loads(row["query_json"])

    limits = query.constraints(structured)
    scored = query.score_all(conn, structured, **limits)
    picks = query.select(scored, count, max_per_artist=config.MAX_PER_ARTIST,
                         seed=seed)

    # The track id is what the cover-art route needs, and score_all returns paths.
    ids = {r["path"]: r["id"] for r in conn.execute(
        "SELECT id, path FROM tracks WHERE missing_since IS NULL")}
    conn.close()

    tracks = [{
        "id": ids.get(p.path),
        "artist": p.artist or "Unknown Artist",
        "title": p.title or Path(p.path).stem,
        "album": p.album,
        "year": p.year,
        "duration": p.duration,
        "confidence": int(p.parts.get("confidence", 0)),
    } for p in picks]

    return render_template(
        "playlist.html",
        query_id=query_id, text=row["text"], structured=structured,
        title=structured.get("title") or "Playlist",
        tracks=tracks, seed=seed, count=count,
        candidates=len(scored), limits=limits,
        # Shown on the export menu so the absolute option states the root it
        # will actually write, rather than leaving it to be discovered by a
        # playlist that imports as 40 missing files.
        m3u_base=(config.M3U_PATH_PREFIX or config.LIBRARY_PATH),
        total_seconds=sum(p.duration for p in picks))


@app.post("/playlist/<int:query_id>/rethink")
def rethink(query_id: int):
    """Translate the same words again, from scratch.

    Distinct from a re-roll: this asks the model to read the request afresh, so
    the axis targets and filters can come out differently. A re-roll only
    re-samples the tracks that the existing reading already selected.
    """
    conn = db.connect()
    row = conn.execute("SELECT text FROM queries WHERE id = ?",
                       (query_id,)).fetchone()
    if row is None:
        conn.close()
        abort(404)
    try:
        structured = query.translate(row["text"])
    except Exception as exc:                      # noqa: BLE001
        conn.close()
        flash(f"Could not reach the model: {exc}", "warning")
        return redirect(url_for("show_playlist", query_id=query_id))
    new_id = db.record_query(conn, row["text"], structured)
    conn.close()
    return redirect(url_for("show_playlist", query_id=new_id))


def _clean_prefix(raw: str | None) -> str | None:
    """Sanitise a client-supplied library root.

    Returns None for "not given", which is what `render()` reads as "use the
    configured default" -- an empty string means something else there.

    The value is untrusted text that ends up in a file the user then opens. The
    one hazard worth naming is a newline: it would inject extra lines into the
    playlist, since the format is line-oriented and has no escaping.
    """
    if raw is None:
        return None
    cleaned = re.sub(r"[\r\n\x00-\x1f]", "", raw).strip()[:512]
    return cleaned.rstrip("/") or None


@app.get("/playlist/<int:query_id>/export")
def export_playlist(query_id: int):
    import json

    seed = request.args.get("seed", type=int) or 1
    count = request.args.get("n", type=int) or config.PLAYLIST_SIZE
    conn = db.connect()
    row = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
    if row is None:
        conn.close()
        abort(404)
    structured = json.loads(row["query_json"])
    limits = query.constraints(structured)
    picks = query.select(query.score_all(conn, structured, **limits), count,
                         max_per_artist=config.MAX_PER_ARTIST, seed=seed)
    conn.close()

    title = structured.get("title") or "Playlist"
    # Where the music lives is a fact about the *client*, not the server, so the
    # client says. `?prefix=` carries the importing machine's library root and
    # `?relative=1` asks for no root at all; neither is stored, and the server's
    # own LIBRARY_PATH remains the default. Putting the Mac's SMB path into the
    # server's .env would mean the server holding one client's configuration.
    body = m3u.render(picks, title=title,
                      path_prefix=_clean_prefix(request.args.get("prefix")),
                      relative=request.args.get("relative") == "1")
    return send_file(
        io.BytesIO(body.encode("utf-8")),
        mimetype="audio/x-mpegurl", as_attachment=True,
        download_name=f"{m3u.filename_for(title)}.m3u8")


# --------------------------------------------------------------------------
# machine API — one request in, playable URLs out
# --------------------------------------------------------------------------
#
# Written for Home Assistant, which enqueues the returned URLs onto a Sonos
# speaker. It is the same three calls the browser routes make, so the two cannot
# give different playlists for the same words; only the rendering differs.
#
# The response carries `query_id` and `seed`, which makes /playlist/<id>?seed=N
# show exactly what the speaker is playing. That falls out of a playlist being
# derived rather than stored, and is worth keeping.

# Sonos decides what it can play from the Content-Type, so guessing wrong is
# silent: the track is skipped, not reported. mimetypes does not know .m4a on
# every platform, so the mapping is explicit rather than discovered.
AUDIO_MIMETYPES = {
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".mp3": "audio/mpeg",
    ".flac": "audio/flac", ".aiff": "audio/aiff", ".aif": "audio/aiff",
    ".wav": "audio/wav",
}


def _display_name(artist: str, title: str, suffix: str) -> str:
    """The filename Sonos shows in its queue.

    HA's `media_player.play_media` cannot pass DIDL metadata, so a queued HTTP
    URL is displayed as its last path segment. Ignoring that gives forty rows
    reading `1234`; spending a segment on it gives `Bonobo - Kiara.m4a`.

    Slashes have to go: the route matches with the `path` converter, so a title
    containing one would otherwise split into extra segments and 404.
    """
    label = f"{artist} - {title}".strip(" -") or "track"
    return re.sub(r"[/\\\x00-\x1f]", "_", label)[:120] + suffix


def _api_payload(conn, query_id: int, structured: dict, seed: int, count: int):
    """Derive a playlist and render it as JSON.

    Shared by the POST that creates one and the GET that reads it back, so the
    two cannot describe the same (query_id, seed) differently — which they
    would, eventually, as two copies of this arithmetic.
    """
    limits = query.constraints(structured)
    picks = query.select(query.score_all(conn, structured, **limits), count,
                         max_per_artist=config.MAX_PER_ARTIST, seed=seed)
    ids = {r["path"]: r["id"] for r in conn.execute(
        "SELECT id, path FROM tracks WHERE missing_since IS NULL")}

    tracks = []
    for pick in picks:
        track_id = ids.get(pick.path)
        if track_id is None:                      # went missing between the two reads
            continue
        artist = pick.artist or "Unknown Artist"
        title = pick.title or Path(pick.path).stem
        tracks.append({
            "id": track_id,
            "artist": artist,
            "title": title,
            "album": pick.album,
            "duration": pick.duration,
            "url": url_for("audio", track_id=track_id,
                           name=_display_name(artist, title,
                                              Path(pick.path).suffix.lower()),
                           _external=True),
            # The existing browser route, absolute so a dashboard on another
            # host can render it.
            "cover": url_for("cover", track_id=track_id, _external=True),
        })

    return {
        "query_id": query_id,
        "seed": seed,
        "title": structured.get("title") or "Playlist",
        "count": len(tracks),
        "tracks": tracks,
        "page_url": url_for("show_playlist", query_id=query_id, seed=seed,
                            n=count, _external=True),
    }


@app.post("/api/playlist")
def api_playlist():
    payload = request.get_json(silent=True) or request.form
    text = (payload.get("mood") or "").strip()
    if not text:
        return {"error": "no mood given"}, 400
    count = int(payload.get("count") or config.PLAYLIST_SIZE)
    seed = int(payload.get("seed") or random.randint(1, 10_000))

    conn = db.connect()
    if db.counts(conn)["tagged"] == 0:
        conn.close()
        return {"error": "nothing is tagged yet"}, 503
    try:
        structured = query.translate(text)
    except Exception as exc:                      # noqa: BLE001 - reported as JSON
        conn.close()
        # 502: the request was fine, the model behind us was not. HA shows the
        # status, and a 500 here would send it looking in the wrong place.
        return {"error": f"could not reach the model: {exc}"}, 502

    query_id = db.record_query(conn, text, structured)
    body = _api_payload(conn, query_id, structured, seed, count)
    conn.close()
    return body


@app.get("/api/playlist/<int:query_id>")
def api_playlist_read(query_id: int):
    """Read a playlist back without spending a model call.

    A playlist is (query_id, seed), so a client that kept those two numbers can
    ask for the tracks again — which is how a dashboard shows what it queued
    without needing anywhere to store forty rows.
    """
    import json

    seed = request.args.get("seed", type=int) or 1
    count = request.args.get("n", type=int) or config.PLAYLIST_SIZE
    conn = db.connect()
    row = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
    if row is None:
        conn.close()
        return {"error": "no such query"}, 404
    body = _api_payload(conn, query_id, json.loads(row["query_json"]), seed, count)
    body["text"] = row["text"]
    conn.close()
    return body


@app.get("/api/recent")
def api_recent():
    """Past requests, deduplicated by text — the same list the home page shows.

    Grouped by text because the queries table records every run, including CLI
    and eval ones, so the raw list is mostly repeats of whatever was last being
    worked on.
    """
    limit = request.args.get("limit", type=int) or config.WEB_RECENT_QUERIES
    conn = db.connect()
    rows = list(conn.execute("""
        SELECT MAX(id) AS id, text, MAX(created_at) AS created_at
        FROM queries GROUP BY text ORDER BY id DESC LIMIT :limit""",
        {"limit": limit}))
    conn.close()
    return {"moods": [{"query_id": r["id"], "text": r["text"],
                       "created_at": r["created_at"]} for r in rows]}


@app.get("/audio/<int:track_id>/<path:name>")
def audio(track_id: int, name: str):
    """Stream one track. `name` is display only — the path comes from the id.

    Sonos issues a HEAD and then a ranged GET, and seeking within a track is
    ranged too, so this must answer 206. Flask's send_file does that on its own
    given a real path; building a BytesIO here (as the cover route does) would
    quietly serve the whole file for every seek.
    """
    conn = db.connect()
    row = conn.execute("SELECT path FROM tracks WHERE id = ?",
                       (track_id,)).fetchone()
    conn.close()
    if row is None:
        abort(404)

    path = config.LIBRARY_PATH / row["path"]
    if not path.exists():
        abort(404)
    return send_file(path, conditional=True,
                     mimetype=AUDIO_MIMETYPES.get(path.suffix.lower()))


# --------------------------------------------------------------------------
# cover art
# --------------------------------------------------------------------------

@app.get("/cover/<int:track_id>")
def cover(track_id: int):
    """Serve the artwork embedded in the file.

    98% of this library carries a `covr` atom, median 202 KB. They are served
    raw rather than resized -- adding an imaging dependency to shrink a picture
    that a LAN delivers instantly is not a trade worth making -- but with a long
    cache lifetime, because a playlist page requests twenty of them and a re-roll
    requests many of the same ones again.
    """
    conn = db.connect()
    row = conn.execute("SELECT path FROM tracks WHERE id = ?",
                       (track_id,)).fetchone()
    conn.close()
    if row is None:
        abort(404)

    data, mime = _artwork(config.LIBRARY_PATH / row["path"])
    if data is None:
        return send_file(PLACEHOLDER, mimetype="image/svg+xml")

    response = send_file(io.BytesIO(data), mimetype=mime)
    # Immutable: the artwork for a given track id only changes if the file is
    # re-tagged, and that assigns a new content_key anyway.
    response.headers["Cache-Control"] = (
        f"public, max-age={config.WEB_COVER_CACHE_SECONDS}")
    return response


def _artwork(path: Path) -> tuple[bytes | None, str]:
    try:
        audio = MutagenFile(path)
    except Exception:
        return None, ""
    tags = getattr(audio, "tags", None) or {}

    covers = tags.get("covr") if hasattr(tags, "get") else None   # MP4
    if covers:
        picture = covers[0]
        # MP4Cover.imageformat: 13 = JPEG, 14 = PNG
        mime = "image/png" if getattr(picture, "imageformat", 13) == 14 else "image/jpeg"
        return bytes(picture), mime

    for key in list(getattr(tags, "keys", lambda: [])()):         # ID3
        if str(key).startswith("APIC"):
            frame = tags[key]
            return frame.data, getattr(frame, "mime", "image/jpeg")

    pictures = getattr(audio, "pictures", None)                   # FLAC
    if pictures:
        return pictures[0].data, pictures[0].mime
    return None, ""


@app.template_filter("clock")
def clock(seconds: float | None) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> None:
    import argparse

    # Flags mirror the .env keys and win for a single run, exactly as they do in
    # the CLI. `None` means "not given", so the configured value survives.
    parser = argparse.ArgumentParser(description="music-mood web UI")
    parser.add_argument("--host", help="override WEB_HOST (0.0.0.0 for the LAN)")
    parser.add_argument("--port", type=int, help="override WEB_PORT")
    parser.add_argument("--debug", action="store_true", help="override WEB_DEBUG")
    args = parser.parse_args()

    host = args.host or config.WEB_HOST
    port = args.port or config.WEB_PORT
    debug = args.debug or config.WEB_DEBUG
    print(f"🎧 music-mood → http://{host}:{port}", flush=True)
    if host == "0.0.0.0":
        print("   reachable from the network — this app has no authentication",
              flush=True)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
