"""The machine API, as Home Assistant consumes it.

HA enqueues the returned URLs straight onto a Sonos speaker, and every failure
mode here is silent at the speaker: a wrong Content-Type is a skipped track, a
missing 206 is a seek that restarts the song, a slash in a title is a 404. None
of them raise anything on the server, so they are pinned here instead.
"""
from __future__ import annotations

from urllib.parse import unquote, urlsplit

import pytest

from moodlib import config, db, query
from webapp import app as webapp

from tests.test_query import add_track


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.sqlite3")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "LIBRARY_PATH", tmp_path / "Music")
    (tmp_path / "Music").mkdir()
    conn = db.connect(tmp_path / "api.sqlite3")
    conn.close()
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as test_client:
        yield test_client


def library(conn_path, *, artist="Bonobo", title="Kiara", path="Bonobo/Kiara.m4a"):
    conn = db.connect(conn_path)
    add_track(conn, path, artist=artist, title=title)
    conn.close()


def translation(monkeypatch, **extra):
    """Stand in for the one LLM call. Everything after it is local arithmetic."""
    structured = {"title": "Test Playlist", "energy": {"target": 50, "weight": 1.0}}
    structured.update(extra)
    monkeypatch.setattr(query, "translate", lambda text, session=None: structured)


def test_a_mood_comes_back_as_playable_urls(client, tmp_path, monkeypatch):
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)

    response = client.post("/api/playlist", json={"mood": "late night", "count": 5})
    assert response.status_code == 200
    body = response.get_json()

    assert body["title"] == "Test Playlist"
    assert body["count"] == len(body["tracks"]) == 1
    track = body["tracks"][0]
    # Absolute, because Sonos fetches it — a relative URL would resolve against
    # the speaker itself and fail with nothing logged anywhere useful.
    assert track["url"].startswith("http://")
    # Percent-encoded on the wire, and the last segment is the label Sonos
    # shows in its queue — the one piece of metadata that survives play_media.
    assert unquote(urlsplit(track["url"]).path).endswith("/Bonobo - Kiara.m4a")


def test_the_response_points_back_at_a_reproducible_page(client, tmp_path,
                                                         monkeypatch):
    # A playlist is derived from (query_id, seed), so the same pair must render
    # in the browser as whatever the speaker was handed. Losing either turns
    # "what was that track?" into an unanswerable question.
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)

    body = client.post("/api/playlist", json={"mood": "late night"}).get_json()
    assert f"/playlist/{body['query_id']}" in body["page_url"]
    assert f"seed={body['seed']}" in body["page_url"]


def test_a_seed_replays_the_same_selection(client, tmp_path, monkeypatch):
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)

    first = client.post("/api/playlist", json={"mood": "x", "seed": 7}).get_json()
    again = client.post("/api/playlist", json={"mood": "x", "seed": 7}).get_json()
    assert [t["id"] for t in first["tracks"]] == [t["id"] for t in again["tracks"]]


def test_an_empty_mood_is_rejected_rather_than_translated(client, tmp_path,
                                                          monkeypatch):
    library(tmp_path / "api.sqlite3")
    monkeypatch.setattr(query, "translate", lambda *a, **k: pytest.fail(
        "an empty request must not reach the model"))
    assert client.post("/api/playlist", json={"mood": "   "}).status_code == 400


def test_a_model_failure_is_a_bad_gateway(client, tmp_path, monkeypatch):
    # 502 rather than 500: the request was well-formed and the thing that broke
    # is behind us. HA surfaces the status, and 500 sends you reading Flask logs
    # for a vLLM that is simply down.
    library(tmp_path / "api.sqlite3")

    def explode(text, session=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(query, "translate", explode)
    response = client.post("/api/playlist", json={"mood": "late night"})
    assert response.status_code == 502
    assert "connection refused" in response.get_json()["error"]


def test_nothing_tagged_is_reported_before_the_model_is_called(client, tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(query, "translate", lambda *a, **k: pytest.fail(
        "an empty library must not cost a model call"))
    assert client.post("/api/playlist", json={"mood": "x"}).status_code == 503


def test_a_title_with_a_slash_still_resolves(client, tmp_path, monkeypatch):
    # The route matches with the `path` converter, so an unescaped slash would
    # split into extra segments. AC/DC is not a hypothetical library.
    library(tmp_path / "api.sqlite3", artist="AC/DC", title="Back In Black",
            path="ACDC/Back In Black.m4a")
    translation(monkeypatch)
    (tmp_path / "Music" / "ACDC").mkdir()
    (tmp_path / "Music" / "ACDC" / "Back In Black.m4a").write_bytes(b"audio")

    url = client.post("/api/playlist", json={"mood": "x"}).get_json()["tracks"][0]["url"]
    assert client.get(urlsplit(url).path).status_code == 200


def test_audio_is_served_with_a_type_sonos_understands(client, tmp_path):
    library(tmp_path / "api.sqlite3")
    (tmp_path / "Music" / "Bonobo").mkdir()
    (tmp_path / "Music" / "Bonobo" / "Kiara.m4a").write_bytes(b"audio-bytes")

    response = client.get("/audio/1/Bonobo - Kiara.m4a")
    assert response.status_code == 200
    # audio/mp4, not the octet-stream a bare mimetypes lookup gives for .m4a on
    # some platforms — Sonos skips what it cannot identify, without complaining.
    assert response.headers["Content-Type"].startswith("audio/mp4")


def test_audio_answers_a_range_request(client, tmp_path):
    # Sonos seeks with ranges. Serving 200 with the whole body for every seek
    # works, slowly and wrongly, which is why this is asserted rather than
    # assumed from send_file's defaults.
    library(tmp_path / "api.sqlite3")
    (tmp_path / "Music" / "Bonobo").mkdir()
    (tmp_path / "Music" / "Bonobo" / "Kiara.m4a").write_bytes(b"0123456789")

    response = client.get("/audio/1/Bonobo - Kiara.m4a",
                          headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.data == b"2345"


def test_a_missing_file_is_a_404_not_a_traceback(client, tmp_path):
    library(tmp_path / "api.sqlite3")            # row exists, file never written
    assert client.get("/audio/1/Bonobo - Kiara.m4a").status_code == 404
    assert client.get("/audio/999/whatever.m4a").status_code == 404


def test_a_playlist_reads_back_identically_without_a_model_call(client, tmp_path,
                                                                monkeypatch):
    # The POST and the GET derive from the same (query_id, seed). If they ever
    # disagree, the dashboard shows a playlist the speaker is not playing.
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)
    made = client.post("/api/playlist", json={"mood": "x", "seed": 3}).get_json()

    monkeypatch.setattr(query, "translate", lambda *a, **k: pytest.fail(
        "reading a playlist back must not cost a model call"))
    read = client.get(f"/api/playlist/{made['query_id']}?seed=3").get_json()

    assert read["title"] == made["title"]
    assert [t["id"] for t in read["tracks"]] == [t["id"] for t in made["tracks"]]
    assert read["text"] == "x"


def test_every_track_carries_cover_art(client, tmp_path, monkeypatch):
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)
    body = client.post("/api/playlist", json={"mood": "x"}).get_json()
    cover = body["tracks"][0]["cover"]
    # Absolute for the same reason the audio URL is: a dashboard on another host
    # renders it.
    assert cover.startswith("http://") and "/cover/" in cover


def test_an_unknown_playlist_is_a_404(client, tmp_path):
    assert client.get("/api/playlist/9999").status_code == 404


def test_recent_moods_are_deduplicated_newest_first(client, tmp_path, monkeypatch):
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)
    for mood in ["rainy", "workout", "rainy"]:
        client.post("/api/playlist", json={"mood": mood})

    moods = client.get("/api/recent").get_json()["moods"]
    # "rainy" ran twice but is one entry, and its second run moves it to the top.
    assert [m["text"] for m in moods] == ["rainy", "workout"]


def test_recent_moods_honour_a_limit(client, tmp_path, monkeypatch):
    library(tmp_path / "api.sqlite3")
    translation(monkeypatch)
    for mood in ["a", "b", "c"]:
        client.post("/api/playlist", json={"mood": mood})
    assert len(client.get("/api/recent?limit=2").get_json()["moods"]) == 2


def test_the_display_name_is_the_only_thing_the_name_segment_does():
    # The path is looked up from the id, so the segment is untrusted text that
    # never reaches the filesystem. Worth pinning: the day it starts being used
    # to resolve a file is the day it becomes a traversal.
    import inspect
    source = inspect.getsource(webapp.audio)
    assert "row[\"path\"]" in source
    assert "/ name" not in source and "+ name" not in source
