# Music Mood

Describe a mood in your own words, get back a playlist drawn from your own
library.

```bash
python3 run.py playlist "rainy sunday morning, a bit wistful" -n 30
```

Nothing in a music library records mood — the files carry
`title / artist / album / genre / year` and little else. So this builds a mood
layer first, by asking a local LLM to characterise every track against a fixed
ontology, and then retrieves against it.

The LLM is used exactly twice: once offline over the whole library, and once per
query to translate your phrasing. Everything between is local arithmetic over
SQLite. Tagging is a one-off multi-hour job; a playlist then comes back in about
eight seconds and costs nothing to re-roll.

Standalone and portable — no AppleScript, no Music.app, no macOS dependency. It
writes an `.m3u8` you import yourself.

---

## Install

Requires Python 3.10+, [`ffmpeg`/`ffprobe`](https://ffmpeg.org)
(`brew install ffmpeg`), and an OpenAI-compatible LLM endpoint.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # then set LIBRARY_PATH, LLM_URL, LLM_PORT
```

There is no install step and no packaging. `run.py` is the entry point; it
re-execs itself under `.venv` if launched with an interpreter that lacks the
dependencies, so `./run.py` works too.

## Quick start

```bash
python3 run.py scan                    # index the library (~6 min first time)
python3 run.py tag --limit 200         # validate quality on a slice first
python3 run.py stats                   # check the slice before committing
python3 run.py tag --all               # the full run — resumable, ~6 h for 23k
python3 run.py playlist "melancholy but driving, late night" -n 40 --explain
```

Routine upkeep afterwards is one line — `scan` is incremental and `tag` only
picks up what changed:

```bash
python3 run.py scan && python3 run.py tag --all
```

---

## The ontology

Built on three established standards rather than invented vocabulary. Beyond
correctness, this has a practical payoff: the model has seen these exact terms
and scales throughout its training data, so it is better calibrated on them than
on private coinages.

| Layer | Standard | What it carries |
| --- | --- | --- |
| 1 | **Spotify / Echo Nest** audio features | `energy`, `valence`, `danceability`, `acousticness`, `instrumentalness` (0.0–1.0) |
| 2 | **GEMS-9** (Geneva Emotional Music Scale) | `wonder, transcendence, nostalgia, tenderness, peacefulness, joyful_activation, power, tension, sadness` (0–100) |
| 3 | 48 descriptive adjectives | each mapped **up** to one GEMS dimension |
| 4 | **Discogs** Genre → Style | normalises the library's own messy genre tags |

Layers 3 and 4 are both hierarchies, and in both the parent is **derived, never
asked for**. The model picks a style; its genre follows from
`STYLES_BY_GENRE`. Asking for both independently produced pairs the taxonomy
does not contain — minimal techno filed under `Brass & Military`, ~900 drum &
bass tracks under `Jazz` because liquid D&B has jazz in it. On the reference
library **35% of rows disagreed with their own style**. A hierarchy you compute
cannot contradict itself; one you ask for twice will.

Spotify [retired the `/audio-features` endpoint in Nov 2024](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api),
so those values cannot be fetched — we generate them. The naming is kept because
it is what everyone reads fluently.

GEMS is the domain-specific model of *music-evoked* emotion, validated precisely
because general-purpose emotion scales model music badly. It ships its own
second-order structure, which we get for free as query roll-ups: ask for
**unease** and both `tension` and `sadness` are covered.

| Factor | Dimensions |
| --- | --- |
| sublimity | wonder, transcendence, nostalgia, tenderness, peacefulness |
| vitality | joyful_activation, power |
| unease | tension, sadness |

**Layer 3 is what makes this an ontology rather than a tag list.** "Brooding" is
not a loose string, it is a term under `sadness`. So a query asking for
*yearning* still finds a track tagged only *wistful* — both sit under
`nostalgia` — and a difference in wording costs a good track very little.

**Confidence (`k`, 0–2) is the most important field.** The model recognises
*Smells Like Teen Spirit*; it is guessing from the genre tag on a white-label
drum & bass 12". `2` = recognises the recording, `1` = knows the artist or style,
`0` = pure inference. Without it the ontology looks uniformly confident while a
large share of it is guesswork. It becomes both a filter and a score weight.

Everything is stamped with `ONTOLOGY_VERSION`; bumping it marks every older row
stale so `tag` re-does them.

---

## Track identity — the part worth understanding

A track has three identifiers, and knowing which does what is the design:

| | Survives | Fails on |
| --- | --- | --- |
| `id` | everything — it is a surrogate integer | — |
| `(device, inode)` | rename, directory move, **tag edit** | file rewritten, re-added, restored |
| `content_key` (hash of the tags) | rewrite, re-add, restore | any tag edit |

**`path` is deliberately not an identifier.** Music renames files from their
tags, so a path is the least stable thing about a track. An earlier version of
this schema used it as the primary key, and a tag edit then changed the path
*and* the content hash at once — making the track indistinguishable from a
deletion plus an unrelated new file. It lost its row, its id, and anything
referencing it.

The two real signals cover each other's blind spot exactly:

```
renamed, tags unchanged    inode same, content_key same   either matches
renamed BY a tag edit      inode same, content_key NEW    inode matches
rewritten / re-added       inode NEW,  content_key same   content_key matches
genuinely gone             neither                        soft-marked missing
```

This is not hypothetical. The first real re-scan of the reference library found
456 files renamed underneath it by another process while the scan was running.

Inode matches are corroborated by **duration**, not size — editing a tag changes
a file's length, so a size check would reject the very case the inode branch
exists to catch. Content-key matches must be *unambiguous*: tags are not unique
across 23k tracks (the reference library holds 139 duplicate tag-sets), and a
wrong attribution silently describes the wrong song forever, whereas re-tagging
costs seconds.

### Safety rails

- **Deletions are soft.** A track that vanishes is marked `missing_since`, never
  dropped, so a file that comes back keeps its tagging.
- **A scan that would mark more than `SCAN_MISSING_ABORT_PCT` (default 10%) of
  known tracks missing aborts and changes nothing.** The overwhelmingly likely
  cause is an unmounted or half-mounted share, and a scan that "successfully"
  marks 23,000 tracks gone is indistinguishable from a real library wipe.

---

## How a query works

1. **Translate** — one LLM call turns your phrasing into a structured query,
   constrained to the ontology's enums. Every axis and dimension gets a `target`
   *and* a `weight`, so "loud" can pin `energy` hard while leaving
   `acousticness` free.

   **Not every request is a mood.** When one names a genre, an era, or asks for
   canonical tracks, those become *filters* rather than being folded into the
   mood axes:

   | You say | Becomes |
   | --- | --- |
   | "hip hop", "drum and bass" | `genres` / `styles` filter |
   | "old-school", "90s", "golden age" | `year_from` / `year_to` |
   | "legendary", "classic", "iconic" | `min_confidence: 2` — only recordings the model actually recognises |

   This matters more than it sounds. *"old-school legendary hip hop"* once
   returned Pavement, Bloc Party and Arctic Monkeys: with nowhere to put "hip
   hop", only the mood residue survived, and groovy-nostalgic-mid-energy
   describes indie rock just as well as it describes Rakim. Genre and style are
   **OR-ed**, not AND-ed — a style already implies its genre, so intersecting
   them would exclude the very tracks the genre was named to include. A purely
   mood-based request sets no filters at all.
2. **Score** — locally, over SQLite:
   ```
   axis_fit  = Σ w·(1 − |track − target|)      / Σ w
   gems_fit  = Σ w·(1 − |track − target|/100)  / Σ w     incl. second-order factors
   vocab     = adjective / context overlap, normalised to ±1
   score     = (α·axis_fit + β·gems_fit + γ·vocab) × confidence
   ```
3. **Select** — cap tracks per artist, then sample from the top scoring band so
   the same mood asked twice gives a different, equally good playlist.
4. **Name it** — the same translation call also returns a short title, so it
   costs nothing extra. *"rainy sunday morning, coffee, a bit wistful"* comes
   back as **Rainy Morning Wistfulness**.

The title lands in two places, because importers disagree about where to look:
a `#PLAYLIST` directive inside the file (VLC, foobar2000) **and** the filename
(`Rainy Morning Wistfulness.m3u8`). Music.app names an imported playlist after
the file and ignores the directive, so the filename is what you actually see in
the sidebar — which is why it keeps its spaces and only drops the handful of
characters a filesystem genuinely cannot carry. `--title` overrides it.

`--explain` prints the translated query, its rationale, and every pick's score.

---

## Commands

```
scan      [--library P] [--force] [--workers N] [-v]
tag       [TARGET] [--limit N] [--batch-size N] [--concurrency N] [-v]
stats     [--full]
playlist  MOOD [-n N] [-o FILE] [--title T] [--max-per-artist N]
               [--min-confidence 0|1|2] [--genre G ...] [--style S ...]
               [--year-from Y] [--year-to Y] [--seed N] [--explain]
```

### Watching a run

`scan` and `tag` both show what they are working on right now. Which form you
get depends on where the output is going, because getting that wrong is
invisible until you go looking at the log:

```
🎨 4218/22467  0.96/s  ⏱️ 5.3h left  Aphex Twin — Xtal        ← terminal: one line, rewritten
🎨 4500/22467  0.96/s  ⏱️ 5.2h left  Autechre — Second Bad Vilbel   ← redirected: milestones
```

`-v` forces one line per item in either mode. A terminal gets a live line
because 23,000 scrolling lines are not progress, they are noise; a redirected
log gets periodic milestones because carriage returns turn a six-hour run into
one unreadable smear.

Playlists mark each pick with how well the model knew it — 🎯 recognised the
recording, 👤 knew the artist, ❓ inferred from the genre tag.

`stats` is the honesty check. Confidence must **not** be uniformly `2`, and the
distributions must be spread rather than clustered mid-scale — either would mean
the model is hedging instead of characterising. `--full` also reports GEMS
correlations, so you can see whether two dimensions have collapsed into one.

---

## Configuration

Everything operational lives in `.env` (see `.env.example`); nothing tunable is a
literal anywhere else. `config.py` is the only module that reads the environment,
so there is exactly one place to look — and a test enforces both that and the
absence of documented-but-unread keys.

Every setting also has a matching CLI flag, so `.env` sets the default and a flag
wins for one invocation.

**The ontology is the deliberate exception.** The axes, GEMS dimensions, the
adjective→GEMS mapping and the Discogs taxonomy live in `ontology.py` as code,
because changing one must bump `ONTOLOGY_VERSION` to invalidate stale rows — and
a `.env` edit cannot do that.

---

## Performance, and one counter-intuitive result

Measured against a local vLLM serving Qwen3.6-35B-A3B:

| Stage | Rate | 23,000 tracks |
| --- | --- | --- |
| `scan`, first run | ~60 files/s | ~6 min |
| `scan`, incremental | walk only | ~30 s |
| `tag` | 0.96 tracks/s | **6.5 h** (measured, 22,987 tracks) |
| `playlist` | one 8 s LLM call + local scoring | instant |

Tagging throughput, same hardware:

| Batch | `max_tokens` | Concurrency | tracks/s |
| ---: | ---: | ---: | ---: |
| 20 | 8000 | 4 | 0.19 |
| 20 | 8000 | 1 | 0.25 |
| 10 | 2600 | 8 | 0.77 |
| 10 | 2600 | 16 | 1.08 |
| 10 | 2600 | 28 | 1.17 |

**Bigger batches are worse**, which is the opposite of the intuition. vLLM
schedules against the *reserved* output budget, so a generous `max_tokens` lets
far fewer sequences run concurrently than the server can actually hold, and the
batch decodes almost serially. The two settings are not independent, so the
budget is derived (`TAG_BATCH_SIZE × TAG_TOKENS_PER_TRACK`) rather than
configured separately — changing the batch size stays safe.

Two related traps, both encoded in the code:

- **`uniqueItems` is valid JSON Schema and vLLM's grammar backend returns a 500
  for it.** Deduplication lives in `tag.py` instead.
- **Results are consumed with `as_completed`, not `map`.** `map` yields in
  submission order, so one slow request blocks every finished batch behind it
  from committing — which presents as a dead stall rather than as slowness.
- **Ask for the leaf of a hierarchy, derive the parent.** See layer 4 above.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # 67 passed
```

The scan tests exercise every row of the identity table against real files in a
temp directory, including the tag-edit-plus-rename case. They need `ffmpeg` to
synthesise tagged fixtures and skip themselves without it.

---

## Layout

```
run.py                 entry point — no install needed
moodlib/
├── config.py          the only module that reads the environment
├── ontology.py        THE ONTOLOGY — read this first
├── db.py              schema + the identity model
├── scan.py            files -> tracks, incrementally
├── llm.py             OpenAI-compatible client, schema-constrained
├── tag.py             resumable batch mood-tagging
├── query.py           mood text -> structured query -> scored selection
├── progress.py        live status line / milestone logging
├── playlist.py        selection -> titled .m3u8
└── cli.py             argument parsing + dispatch
data/mood.sqlite3
```
