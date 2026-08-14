"""The ontology's internal consistency is load-bearing and easy to break silently.

Every one of these pins an invariant that other modules assume. A broken mapping
here does not raise anywhere -- it just quietly degrades retrieval, which is the
worst kind of bug to have in a system whose output is subjective.
"""
from __future__ import annotations

from moodlib import ontology


def test_every_adjective_maps_to_a_real_gems_dimension():
    # query._vocab_fit awards near-miss credit through this parent. A typo here
    # silently drops that credit instead of failing.
    unknown = {term: parent for term, parent in ontology.ADJECTIVES.items()
               if parent not in ontology.GEMS}
    assert unknown == {}


def test_every_gems_factor_member_is_a_real_dimension():
    unknown = [(factor, member)
               for factor, members in ontology.GEMS_FACTORS.items()
               for member in members if member not in ontology.GEMS]
    assert unknown == []


def test_factors_partition_the_nine_dimensions():
    # GEMS' published structure assigns all nine, each to exactly one factor.
    # A dimension in two factors would be double-counted by a factor-weighted
    # query; one in none would be unreachable through a factor.
    assigned = [m for members in ontology.GEMS_FACTORS.values() for m in members]
    assert sorted(assigned) == sorted(ontology.GEMS)


def test_factor_value_is_the_mean_of_its_members():
    dimensions = {name: 0.0 for name in ontology.GEMS}
    dimensions["tension"] = 80.0
    dimensions["sadness"] = 40.0
    assert ontology.factor_value(dimensions, "unease") == 60.0


def test_genre_hints_reference_the_real_taxonomy():
    bad = [(raw, hint) for raw, hint in ontology.RAW_GENRE_HINTS.items()
           if hint[0] not in ontology.DISCOGS_GENRES
           or (hint[1] and hint[1] not in ontology.DISCOGS_STYLES)]
    assert bad == []


def test_genre_hint_consolidates_the_librarys_fragmentation():
    # The whole justification for layer 4: these raw strings are 4,000+ tracks
    # that would otherwise be four unrelated genres.
    assert ontology.genre_hint("Drum & Bass") == ontology.genre_hint("Drum n Bass")
    assert ontology.genre_hint("Raggae") == ontology.genre_hint("Reggae")
    for raw in ("Hip Hop", "Hip-Hop/Rap", "Rap", "Hip-Hop"):
        assert ontology.genre_hint(raw)[0] == "Hip Hop", raw


def test_genre_hint_declines_uninformative_tags():
    # A wrong hint is worse than none: the model defers to it. Decade tags and
    # junk must return None rather than a guess.
    for raw in ("80s", "127", "PROMO", "", "   ", "Crossover Nonsense"):
        assert ontology.genre_hint(raw) is None, raw


def test_schemas_do_not_use_unique_items():
    # vLLM's grammar backend returns a 500 for `uniqueItems`, so the constraint
    # lives in tag._unique instead. This test exists because the schema looked
    # more correct with it and it took a full failed run to find out.
    import json
    for schema in (ontology.tag_schema(), ontology.query_schema()):
        assert "uniqueItems" not in json.dumps(schema)


def test_tag_schema_requires_every_scored_field():
    item = ontology.tag_schema()["properties"]["r"]["items"]
    for key in (*ontology.AXIS_KEYS, *ontology.GEMS_KEYS, "m", "c", "s", "k"):
        assert key in item["required"], key
    assert item["additionalProperties"] is False


def test_tag_schema_does_not_ask_for_the_genre():
    """The genre is derived from the style, never requested.

    Asking for both produced combinations the taxonomy does not contain --
    `Brass & Military / Minimal`, and ~900 drum & bass tracks under `Jazz`.
    """
    item = ontology.tag_schema()["properties"]["r"]["items"]
    assert "g" not in item["properties"]


def test_every_style_belongs_to_exactly_one_genre():
    seen: dict[str, str] = {}
    for genre, styles in ontology.STYLES_BY_GENRE.items():
        for style in styles:
            assert style not in seen, f"{style} is under {seen.get(style)} and {genre}"
            seen[style] = genre
    assert set(seen) == set(ontology.DISCOGS_STYLES)


def test_every_style_genre_is_a_real_discogs_genre():
    unknown = [g for g in ontology.STYLES_BY_GENRE if g not in ontology.DISCOGS_GENRES]
    assert unknown == []


def test_genre_is_derived_from_style():
    assert ontology.genre_for_style("Minimal") == "Electronic"
    assert ontology.genre_for_style("Drum n Bass") == "Electronic"
    assert ontology.genre_for_style("Jazz-Funk") == "Jazz"
    assert ontology.genre_for_style("nonsense") == ""


def test_query_schema_offers_factors_alongside_dimensions():
    required = set(ontology.query_schema()["required"])
    assert set(ontology.GEMS) <= required
    assert set(ontology.GEMS_FACTORS) <= required
    assert set(ontology.AXES) <= required
