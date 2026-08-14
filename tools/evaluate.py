#!/usr/bin/env python3
"""Measure playlist quality against stated expectations.

Judging a playlist by reading it is how "Sexy Bitch" ended up in a high-intensity
workout list: most of the twenty picks were right, so nothing looked wrong until
someone actually read line 19. This runs a fixed set of requests and checks each
result against expectations written down in advance, so a change to a prompt or
to the scoring can be shown to help rather than merely felt to.

    python3 tools/evaluate.py                # all cases
    python3 tools/evaluate.py workout sleep  # only matching cases
    python3 tools/evaluate.py --verbose      # list every pick

Every check names the dimension it is about, so a failure says which part of the
ontology was not respected -- not just "this looks wrong".
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moodlib import config, db, progress, query  # noqa: E402


@dataclass
class Case:
    """One request plus what a good answer to it must look like."""
    name: str
    request: str
    #: dimension -> minimum every pick must reach (axes 0-1, GEMS 0-100)
    floor: dict[str, float] = field(default_factory=dict)
    #: dimension -> maximum every pick may reach
    ceiling: dict[str, float] = field(default_factory=dict)
    #: at least this share of picks must carry one of these contexts
    contexts: tuple[tuple[str, ...], float] | None = None
    #: no pick may carry any of these contexts
    forbid_contexts: tuple[str, ...] = ()
    #: every pick's Discogs genre must be one of these
    genres: tuple[str, ...] = ()
    #: shortest acceptable track, seconds -- interludes are not workout music
    min_duration: float = 0.0
    count: int = 15


CASES: list[Case] = [
    Case(
        "workout",
        "High intensity workout, speed, energy",
        # Intensity is power AND edge. Without a tension floor this admits any
        # cheerful club record: "Sexy Bitch" scores power 85 at tension 15.
        floor={"energy": 0.70, "power": 70, "tension": 30, "danceability": 0.55},
        # No sadness ceiling: an earlier version capped it at 40 and flagged
        # Nine Inch Nails and Bullet For My Valentine, which are perfectly good
        # workout music. Aggressive music is often dark; the check was wrong,
        # not the result. What actually disqualified the opera aria in that same
        # list was rhythm, so danceability is the honest discriminator.
        ceiling={"peacefulness": 40},
        forbid_contexts=("sleep", "dinner"),
        min_duration=100,
    ),
    Case(
        "sleep",
        "drifting off to sleep, weightless",
        floor={"peacefulness": 55},
        ceiling={"energy": 0.40, "power": 40, "tension": 30},
        forbid_contexts=("workout", "rage", "party"),
    ),
    Case(
        "melancholy",
        "melancholy but driving, late night",
        floor={"sadness": 40},
        ceiling={"joyful_activation": 60},
    ),
    Case(
        "acoustic",
        "calm acoustic sunday morning, gentle and warm",
        floor={"acousticness": 0.40, "peacefulness": 45},
        ceiling={"energy": 0.55},
    ),
    Case(
        "dnb",
        "ambient drum n bass, feeling blue",
        genres=("Electronic",),
        ceiling={"energy": 0.90},
    ),
    Case(
        "hiphop",
        "old-school legendary hip hop",
        genres=("Hip Hop",),
        count=8,
    ),
    Case(
        "rage",
        "furious angry music, nothing gentle",
        floor={"energy": 0.70, "tension": 45},
        ceiling={"peacefulness": 35},
    ),
]

AXES = ("energy", "valence", "danceability", "acousticness", "instrumentalness")


def _value(row, name: str) -> float | None:
    return row[name] if name in row.keys() else None


def evaluate(case: Case, conn, verbose: bool = False) -> tuple[int, int, list[str]]:
    structured = query.translate(case.request)
    limits = query.constraints(structured)
    scored = query.score_all(conn, structured, **limits)
    picks = query.select(scored, case.count,
                         max_per_artist=config.MAX_PER_ARTIST, seed=11)
    if not picks:
        return 0, 1, ["no tracks selected at all"]

    paths = {p.path for p in picks}
    rows = [r for r in db.iter_scored(conn) if r["path"] in paths]

    failures: list[str] = []
    checks = 0

    def label(row) -> str:
        return f"{row['artist']} — {row['title']}"

    for name, minimum in case.floor.items():
        checks += 1
        bad = [r for r in rows if (_value(r, name) or 0) < minimum]
        if bad:
            failures.append(
                f"{name} < {minimum}: " +
                "; ".join(f"{label(r)} ({_value(r, name):.2f})" for r in bad[:3]) +
                (f" +{len(bad) - 3} more" if len(bad) > 3 else ""))

    for name, maximum in case.ceiling.items():
        checks += 1
        bad = [r for r in rows if (_value(r, name) or 0) > maximum]
        if bad:
            failures.append(
                f"{name} > {maximum}: " +
                "; ".join(f"{label(r)} ({_value(r, name):.2f})" for r in bad[:3]) +
                (f" +{len(bad) - 3} more" if len(bad) > 3 else ""))

    if case.forbid_contexts:
        checks += 1
        bad = [r for r in rows
               if set(json.loads(r["contexts_json"] or "[]")) & set(case.forbid_contexts)]
        if bad:
            failures.append(f"forbidden context {case.forbid_contexts}: " +
                            "; ".join(label(r) for r in bad[:3]))

    if case.genres:
        checks += 1
        bad = [r for r in rows if r["discogs_genre"] not in case.genres]
        if bad:
            failures.append(f"genre not in {case.genres}: " +
                            "; ".join(f"{label(r)} ({r['discogs_genre']})" for r in bad[:3]))

    if case.min_duration:
        checks += 1
        bad = [p for p in picks if p.duration < case.min_duration]
        if bad:
            failures.append(f"shorter than {case.min_duration:.0f}s: " +
                            "; ".join(f"{p.artist} — {p.title} ({p.duration:.0f}s)"
                                      for p in bad[:3]))

    if verbose:
        say(f"    title: “{structured.get('title', '')}”")
        weighted = {k: v for k, v in structured.items()
                    if isinstance(v, dict) and float(v.get("weight") or 0) > 0}
        say("    weighted: " + ", ".join(
            f"{k}={v['target']}@{v['weight']:.1f}" for k, v in sorted(
                weighted.items(), key=lambda kv: -float(kv[1]["weight"]))))
        for r in rows:
            say(f"      {label(r)[:56]:58} e{r['energy']:.2f} "
                  f"pow{r['power']:3.0f} ten{r['tension']:3.0f} "
                  f"joy{r['joyful_activation']:3.0f}")

    return checks - len(failures), checks, failures


def say(text: str = '') -> None:
    sys.stderr.write(text + '\n')
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*", help="run only these case names")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.only or c.name in args.only]
    conn = db.connect()
    passed = total = 0
    failed_cases = 0

    for case in cases:
        progress.note("query", f"{case.name}: “{case.request}”")
        ok, checks, failures = evaluate(case, conn, args.verbose)
        passed += ok
        total += checks
        mark = "✅" if not failures else "❌"
        say(f"  {mark} {ok}/{checks} checks")
        for line in failures:
            say(f"     ⚠️  {line}")
        if failures:
            failed_cases += 1
        say()

    conn.close()
    say(f"{'✅' if not failed_cases else '❌'} {passed}/{total} checks passed "
          f"across {len(cases)} cases ({failed_cases} cases with failures)")
    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
