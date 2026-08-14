"""Stage 3: free-text mood -> structured query -> scored, diverse selection.

One LLM call translates the request into the ontology (~8s). Everything after
that is local arithmetic over SQLite and runs in milliseconds, which is the whole
point of paying for tagging up front: the expensive part is already done, so a
playlist comes back instantly and costs nothing to re-roll.
"""
from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass, field
from typing import Sequence

from moodlib import config, db, llm, ontology


@dataclass
class Scored:
    path: str
    artist: str
    title: str
    album: str
    year: str
    duration: float
    score: float
    parts: dict[str, float] = field(default_factory=dict)


def translate(text: str, session=None) -> dict:
    """Turn a listener's phrasing into a structured query over the ontology."""
    return llm.complete_json(
        ontology.query_instructions(text), ontology.query_schema(),
        schema_name="mood_query", max_tokens=1500, temperature=0.2,
        session=session)


def _axis_fit(row, query: dict) -> tuple[float, float]:
    """Weighted closeness on the Spotify axes. Returns (weighted_sum, total_weight)."""
    total = weight_sum = 0.0
    for name in ontology.AXES:
        spec = query.get(name) or {}
        weight = float(spec.get("weight") or 0.0)
        if weight <= 0:
            continue
        value = row[name]
        if value is None:
            continue
        target = float(spec.get("target") or 0) / 100.0
        total += weight * (1.0 - abs(value - target))
        weight_sum += weight
    return total, weight_sum


def _gems_fit(row, query: dict) -> tuple[float, float]:
    """Weighted closeness on GEMS-9 plus its second-order factors.

    Factors are derived here rather than stored, so the hierarchy can never drift
    from the dimensions underneath it. A broad request ("something unsettling")
    sets one factor weight instead of guessing nine dimensions.
    """
    dimensions = {name: (row[name] or 0.0) for name in ontology.GEMS}
    total = weight_sum = 0.0

    for name in ontology.GEMS:
        spec = query.get(name) or {}
        weight = float(spec.get("weight") or 0.0)
        if weight <= 0:
            continue
        target = float(spec.get("target") or 0)
        total += weight * (1.0 - abs(dimensions[name] - target) / 100.0)
        weight_sum += weight

    for factor in ontology.GEMS_FACTORS:
        spec = query.get(factor) or {}
        weight = float(spec.get("weight") or 0.0)
        if weight <= 0:
            continue
        value = ontology.factor_value(dimensions, factor)
        target = float(spec.get("target") or 0)
        total += weight * (1.0 - abs(value - target) / 100.0)
        weight_sum += weight

    return total, weight_sum


def _vocab_fit(row, query: dict) -> float:
    """Adjective and context overlap, normalised to roughly [-1, +1].

    The near-miss rule is what layer 3 buys us: an adjective the query asked for
    that the track does not carry still scores through its parent GEMS
    dimension. "Yearning" therefore finds a track tagged only "wistful", because
    both sit under `nostalgia` -- so a difference in wording costs a good track
    much less than an exact-match-only comparison would.

    Normalising by the number of terms asked for is load-bearing, not tidiness.
    Un-normalised, a query naming five adjectives and five to avoid produces a
    raw range near +/-7, which at any useful weight swamps `fit` -- whose entire
    range is 0-1. Observed live: a request for "dark and tense" ranked a romantic
    ballad first, purely because it matched one favoured adjective while the
    genuinely dark track carried one term the model had listed under `avoid`.
    Dividing by the term count keeps this a tiebreaker between tracks the axes
    and emotions already agree on, which is the job it is meant to do.
    """
    track_adjectives = set(json.loads(row["adjectives_json"] or "[]"))
    track_contexts = set(json.loads(row["contexts_json"] or "[]"))
    dimensions = {name: (row[name] or 0.0) for name in ontology.GEMS}
    wanted = query.get("adjectives") or []
    unwanted = query.get("avoid") or []
    contexts = query.get("contexts") or []
    score = 0.0

    for term in wanted:
        if term in track_adjectives:
            score += 1.0
        else:
            parent = ontology.ADJECTIVES.get(term)
            if parent:
                score += 0.5 * (dimensions.get(parent, 0.0) / 100.0)

    for term in unwanted:
        if term in track_adjectives:
            score -= 1.0
        else:
            parent = ontology.ADJECTIVES.get(term)
            if parent:
                score -= 0.5 * (dimensions.get(parent, 0.0) / 100.0)

    for name in contexts:
        if name in track_contexts:
            score += 0.5

    terms = len(wanted) + len(unwanted) + 0.5 * len(contexts)
    return score / terms if terms else 0.0


def _confidence_factor(confidence: int | None) -> float:
    """Scale a score by how much the model actually knew about the track.

    Confidence 0 is pure genre inference, and a large share of a 23k collection
    is exactly that. Those tracks still compete -- they are just handicapped, so
    a recognised track wins a close call.
    """
    floor = config.SCORE_CONFIDENCE_FLOOR
    level = 0 if confidence is None else max(0, min(2, int(confidence)))
    return floor + (1.0 - floor) * (level / 2.0)


def constraints(query: dict, *, min_confidence: int | None = None,
                genres: Sequence[str] = (), styles: Sequence[str] = (),
                year_from: str | None = None, year_to: str | None = None) -> dict:
    """Merge the filters the model derived with the ones the user typed.

    A request can name things that are not moods -- a genre, an era, a demand for
    canonical tracks -- and those are applied as filters rather than folded into
    the mood axes. Expressing "hip hop" as a mood profile is what produced
    Pavement and Arctic Monkeys for "old-school legendary hip hop": the mood
    residue of that request describes indie rock just as well.

    An explicit CLI flag always wins; the model only fills what the user left
    open, so `--genre Rock` still overrides a request that says "hip hop".
    """
    def clamp_year(value) -> str | None:
        # 0 is the schema's "no bound" sentinel -- a nullable integer would be
        # tidier but guided decoding is fussy about unions.
        try:
            year = int(value or 0)
        except (TypeError, ValueError):
            return None
        return str(year) if 1900 < year < 2100 else None

    chosen_genres = tuple(genres) or tuple(query.get("genres") or ())
    chosen_styles = tuple(styles) or tuple(query.get("styles") or ())
    if not genres:                       # never second-guess an explicit flag
        chosen_genres = _drop_modifier_genres(chosen_genres, chosen_styles)

    return {
        "genres": chosen_genres,
        "styles": chosen_styles,
        "year_from": year_from or clamp_year(query.get("year_from")),
        "year_to": year_to or clamp_year(query.get("year_to")),
        "min_confidence": (min_confidence if min_confidence is not None
                           else int(query.get("min_confidence") or 0)),
    }


def _drop_modifier_genres(genres: Sequence[str], styles: Sequence[str]
                          ) -> tuple[str, ...]:
    """Discard a genre that describes the styles rather than containing them.

    Genre and style are OR-ed, which is right when they are a hierarchy -- "hip
    hop, of the boom bap and g-funk kinds" should return all hip hop. It is
    wrong when the genre is a *modifier*: "jazzy house" comes back as genre Jazz
    plus style House, and OR-ing those returns House **or** Jazz, so jazz-funk
    and acid jazz records outrank the house the request was actually about.

    The taxonomy already knows which case this is. A genre that parents one of
    the listed styles is a container and stays; one that parents none of them is
    describing them, and the styles are the real subject. Prompting alone did not
    fix this reliably -- the model kept filing the modifier as a genre -- but the
    hierarchy makes it decidable rather than a matter of phrasing.
    """
    if not styles or not genres:
        return tuple(genres)
    parents = {ontology.genre_for_style(style) for style in styles}
    containers = tuple(genre for genre in genres if genre in parents)
    return containers


def score_all(conn, query: dict, *, min_confidence: int = 0,
              genres: Sequence[str] = (), styles: Sequence[str] = (),
              year_from: str | None = None, year_to: str | None = None
              ) -> list[Scored]:
    results: list[Scored] = []
    for row in db.iter_scored(conn, min_confidence=min_confidence, genres=genres,
                              styles=styles, year_from=year_from, year_to=year_to):
        axis_total, axis_weight = _axis_fit(row, query)
        gems_total, gems_weight = _gems_fit(row, query)
        axis = axis_total / axis_weight if axis_weight else 0.0
        gems = gems_total / gems_weight if gems_weight else 0.0
        vocab = _vocab_fit(row, query)

        # A query that constrains only axes should not be diluted by an unused
        # GEMS term, so each half contributes only if it was actually weighted.
        raw = (config.SCORE_AXIS_WEIGHT * axis if axis_weight else 0.0) \
            + (config.SCORE_GEMS_WEIGHT * gems if gems_weight else 0.0)
        divisor = (config.SCORE_AXIS_WEIGHT if axis_weight else 0.0) \
            + (config.SCORE_GEMS_WEIGHT if gems_weight else 0.0)
        fit = raw / divisor if divisor else 0.0

        total = (fit + config.SCORE_VOCAB_WEIGHT * vocab) \
            * _confidence_factor(row["confidence"])

        results.append(Scored(
            path=row["path"], artist=row["artist"], title=row["title"],
            album=row["album"], year=row["year"], duration=row["duration"] or 0.0,
            score=total,
            parts={"axis": axis, "gems": gems, "vocab": vocab,
                   "confidence": row["confidence"] or 0},
        ))

    results.sort(key=lambda s: s.score, reverse=True)
    return results


#: Featuring credits, wherever the tagger put them. The same recording is
#: routinely filed as `Coolio & L.V. — Gangsta's Paradise` and
#: `Coolio — Gangsta's Paradise (feat. L.V.)`.
_FEATURING = re.compile(
    r"[\(\[]?\s*\b(?:feat|ft|featuring|with)\b\.?\s+[^\)\]]*[\)\]]?", re.I)
_PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-"})


def _recording(item: Scored) -> tuple[str, str]:
    """Identity of the *recording*, for keeping a playlist free of repeats.

    The library holds the same song at more than one path -- 139 duplicate
    tag-sets in the reference collection, from re-rips, compilation appearances
    and Music re-adding a track it already had. Those score identically, so
    without this a playlist happily lists "Modestep — Sunlight" twice in a row.

    An exact artist+title match is not enough, because the same recording gets
    credited differently. Observed live, one playlist carried both:

        Coolio & L.V.  /  Gangsta’s Paradise            (curly apostrophe)
        Coolio         /  Gangsta's Paradise (feat. L.V.)

    -- three differences at once: where the collaborator is credited, the
    apostrophe, and 4s of duration from a different rip. So the key normalises
    typographic punctuation, drops featuring credits from both fields, and takes
    only the artist's leading token, since that is the part that survives every
    way of writing a collaboration.

    Still deliberately conservative about *versions*: "Drive" and
    "Drive (Acoustic)" keep different keys and may both appear, because they are
    genuinely different recordings.
    """
    def clean(text: str) -> str:
        text = (text or "").translate(_PUNCTUATION).lower()
        return " ".join(_FEATURING.sub(" ", text).split())

    artist = clean(item.artist)
    # Leading token only: "coolio & l.v." and "coolio" must agree. Collapsing a
    # band name like "simon & garfunkel" to "simon" is the cost, and it only
    # bites if another act whose name starts the same way has an identically
    # titled track -- rarer than the credit variation this fixes.
    lead = artist.split(" & ")[0].split(",")[0].strip()
    return lead or artist, clean(item.title)


def select(scored: list[Scored], count: int, max_per_artist: int,
           pool_factor: float | None = None, seed: int | None = None) -> list[Scored]:
    """Pick `count` tracks with artist diversity and run-to-run variety.

    A pure top-N collapses onto whichever handful of artists happens to sit
    closest to the query -- forty tracks, six artists. Two rules fix that: a hard
    cap per artist, and sampling from a scoring band rather than taking a strict
    prefix, so asking for the same mood twice gives a different (equally good)
    playlist instead of the identical one.
    """
    if count <= 0 or not scored:
        return []
    factor = config.PLAYLIST_POOL_FACTOR if pool_factor is None else pool_factor
    pool = scored[:max(count, int(count * max(factor, 1.0)))]

    rng = random.Random(seed)
    # Weight by rank within the pool so better tracks are likelier without the
    # result becoming deterministic. Scores are close together near the top, so
    # ranking is a steadier signal here than the raw score gaps.
    weighted = []
    for rank, item in enumerate(pool):
        weight = (len(pool) - rank) ** 2
        weighted.append((rng.random() ** (1.0 / weight), item))
    weighted.sort(key=lambda pair: pair[0], reverse=True)

    chosen: list[Scored] = []
    per_artist: dict[str, int] = {}
    seen_recordings: set[tuple[str, str]] = set()
    for _, item in weighted:
        if _recording(item) in seen_recordings:
            continue
        key = (item.artist or "").strip().lower()
        if max_per_artist > 0 and per_artist.get(key, 0) >= max_per_artist:
            continue
        per_artist[key] = per_artist.get(key, 0) + 1
        seen_recordings.add(_recording(item))
        chosen.append(item)
        if len(chosen) >= count:
            break

    # If the artist cap starved the pool, widen to the full ranked list rather
    # than returning a short playlist the user did not ask for.
    if len(chosen) < count:
        for item in scored:
            if _recording(item) in seen_recordings:
                continue
            key = (item.artist or "").strip().lower()
            if max_per_artist > 0 and per_artist.get(key, 0) >= max_per_artist:
                continue
            per_artist[key] = per_artist.get(key, 0) + 1
            seen_recordings.add(_recording(item))
            chosen.append(item)
            if len(chosen) >= count:
                break

    chosen.sort(key=lambda s: s.score, reverse=True)
    return chosen
