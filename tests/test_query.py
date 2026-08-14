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
              confidence=2, genre="Electronic", style="Techno", year="2020",
              **scores):
    key = db.content_key(artist, "Album", title, "", year)
    cur = conn.execute(
        "INSERT INTO tracks (path, artist, album, title, genre_raw, year, "
        "duration, content_key) VALUES (?,?,?,?,?,?,?,?)",
        (path, artist, "Album", title, "", year, 200.0, key))
    track_id = cur.lastrowid
    row = {name: 0.5 for name in ontology.AXES}
    row.update({name: 50.0 for name in ontology.GEMS})
    row.update(scores)
    row.update({
        "adjectives_json": json.dumps(list(adjectives)),
        "contexts_json": json.dumps(list(contexts)),
        "discogs_genre": genre, "discogs_style": style,
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


# --- non-mood constraints ----------------------------------------------------

def test_named_genre_is_applied_as_a_filter(conn):
    """A request that names a genre must return that genre.

    The regression: "old-school legendary hip hop" returned Pavement, Bloc Party
    and Arctic Monkeys, because the query schema had nowhere to put "hip hop" and
    the leftover mood profile — groovy, nostalgic, mid-energy — describes indie
    rock just as well as it describes Rakim.
    """
    add_track(conn, "rap.m4a", artist="Nas", title="The World Is Yours",
              genre="Hip Hop", style="Boom Bap")
    add_track(conn, "indie.m4a", artist="Pavement", title="Cut Your Hair",
              genre="Rock", style="Indie Rock")

    limits = query.constraints({"genres": ["Hip Hop"], "styles": []})
    scored = query.score_all(conn, {"energy": axis(50)}, **limits)
    assert [s.artist for s in scored] == ["Nas"]


def test_genre_and_style_are_unioned_not_intersected(conn):
    """A style implies its genre, so AND-ing them excludes the genre's own tracks.

    Observed live: the model returned genre Hip Hop with styles Boom Bap and
    G-Funk, and AND-ing cut 2,300 hip hop tracks to 19 — every classic tagged
    Gangsta or Conscious was excluded by a style list meant to describe it.
    """
    for style in ("Boom Bap", "Gangsta", "Conscious"):
        add_track(conn, f"{style}.m4a", artist=style, title="T",
                  genre="Hip Hop", style=style)
    add_track(conn, "rock.m4a", artist="Rock Band", title="T",
              genre="Rock", style="Indie Rock")

    limits = query.constraints({"genres": ["Hip Hop"], "styles": ["Boom Bap"]})
    scored = query.score_all(conn, {"energy": axis(50)}, **limits)
    assert sorted(s.artist for s in scored) == ["Boom Bap", "Conscious", "Gangsta"]


def test_era_words_become_year_bounds(conn):
    limits = query.constraints({"year_from": 1988, "year_to": 1996})
    assert (limits["year_from"], limits["year_to"]) == ("1988", "1996")


def test_zero_is_the_no_bound_sentinel(conn):
    # A nullable integer would be tidier, but guided decoding is fussy about
    # unions and a sentinel cannot 500 the request.
    limits = query.constraints({"year_from": 0, "year_to": 0})
    assert limits["year_from"] is None and limits["year_to"] is None


def test_explicit_flags_override_what_the_model_inferred(conn):
    limits = query.constraints(
        {"genres": ["Hip Hop"], "min_confidence": 2, "year_from": 1990},
        genres=["Rock"], min_confidence=0, year_from="2000")
    assert limits["genres"] == ("Rock",)
    assert limits["min_confidence"] == 0
    assert limits["year_from"] == "2000"


def test_a_pure_mood_request_filters_nothing(conn):
    limits = query.constraints({"genres": [], "styles": [], "min_confidence": 0})
    assert limits["genres"] == () and limits["styles"] == ()
    assert limits["min_confidence"] == 0


def test_the_same_recording_credited_differently_is_one_track(conn):
    """Observed live: one playlist carried Gangsta's Paradise twice.

        Coolio & L.V.  /  Gangsta’s Paradise             (curly apostrophe)
        Coolio         /  Gangsta's Paradise (feat. L.V.)

    Three differences at once — collaborator placement, typographic apostrophe,
    and 4s of duration from a different rip.
    """
    add_track(conn, "a.m4a", artist="Coolio & L.V.", title="Gangsta’s Paradise")
    add_track(conn, "b.m4a", artist="Coolio", title="Gangsta's Paradise (feat. L.V.)")
    scored = query.score_all(conn, {"energy": axis(50)})
    assert len(query.select(scored, count=5, max_per_artist=5, seed=1)) == 1


def test_versions_are_still_distinct(conn):
    # The dedupe must not swallow genuinely different recordings.
    add_track(conn, "a.m4a", artist="Incubus", title="Drive")
    add_track(conn, "b.m4a", artist="Incubus", title="Drive (Acoustic)")
    assert len(query.select(query.score_all(conn, {"energy": axis(50)}),
                            count=5, max_per_artist=5, seed=1)) == 2


def test_different_artists_sharing_a_title_are_distinct(conn):
    add_track(conn, "a.m4a", artist="Incubus", title="Drive")
    add_track(conn, "b.m4a", artist="R.E.M.", title="Drive")
    assert len(query.select(query.score_all(conn, {"energy": axis(50)}),
                            count=5, max_per_artist=5, seed=1)) == 2


def test_a_genre_that_describes_the_style_is_dropped(conn):
    """"jazzy house" is House with a jazzy feel, not House-or-Jazz.

    The model files the modifier as a genre, and OR-ing it in returns jazz-funk
    and acid jazz records ahead of the house the request was about. Jazz does not
    parent House, so the taxonomy itself says which is the subject.
    """
    limits = query.constraints({"genres": ["Jazz"], "styles": ["House"]})
    assert limits["genres"] == ()
    assert limits["styles"] == ("House",)


def test_a_genre_that_contains_the_style_is_kept(conn):
    # The hierarchy case: "hip hop, of the boom bap kind" must still return all
    # hip hop, not just boom bap.
    limits = query.constraints({"genres": ["Hip Hop"], "styles": ["Boom Bap"]})
    assert limits["genres"] == ("Hip Hop",)
    assert limits["styles"] == ("Boom Bap",)


def test_an_explicit_genre_flag_is_never_second_guessed(conn):
    limits = query.constraints({"genres": [], "styles": ["House"]}, genres=["Jazz"])
    assert limits["genres"] == ("Jazz",)


def test_a_genre_alone_is_untouched(conn):
    limits = query.constraints({"genres": ["Hip Hop"], "styles": []})
    assert limits["genres"] == ("Hip Hop",)


def test_a_missing_apostrophe_is_still_the_same_recording(conn):
    """Seen in the web UI at positions 10 and 11 of one playlist.

    Same artist, same 3:46, same artwork — the taggers simply disagreed about
    whether the word is "don't", "don’t" or "dont".
    """
    add_track(conn, "a.m4a", artist="David Guetta Vs The Egg",
              title="Love Dont Let Me Go (Walking Away)")
    add_track(conn, "b.m4a", artist="David Guetta Vs The Egg",
              title="Love Don'T Let Me Go (Walking Away)")
    add_track(conn, "c.m4a", artist="David Guetta Vs The Egg",
              title="Love Don’t Let Me Go (Walking Away)")
    scored = query.score_all(conn, {"energy": axis(50)})
    assert len(query.select(scored, count=5, max_per_artist=5, seed=1)) == 1
