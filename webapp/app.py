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
import sys
from pathlib import Path

from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_file, url_for)
from mutagen import File as MutagenFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moodlib import config, db, ontology, playlist as m3u, query  # noqa: E402
from webapp import jobs  # noqa: E402

app = Flask(__name__)
# Only used for flash messages; this app binds to the LAN and has no accounts.
app.secret_key = "music-mood-local"

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
        FROM queries GROUP BY text ORDER BY id DESC LIMIT 8"""))
    conn.close()
    counts["pending"] = (counts["never_tagged"] + counts["stale_content"]
                         + counts["stale_ontology"] + counts["errors"])
    return render_template("index.html", counts=counts, recent=recent,
                           library=config.LIBRARY_PATH)


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
    body = m3u.render(picks, title=title)
    return send_file(
        io.BytesIO(body.encode("utf-8")),
        mimetype="audio/x-mpegurl", as_attachment=True,
        download_name=f"{m3u.filename_for(title)}.m3u8")


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
    response.headers["Cache-Control"] = "public, max-age=604800"
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

    parser = argparse.ArgumentParser(description="music-mood web UI")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from other machines")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"🎧 music-mood → http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
