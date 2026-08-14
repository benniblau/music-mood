# Working on music-mood

Generates playlists from a mood description, over a local music library, using a
self-hosted OpenAI-compatible LLM. Read `README.md` for what it does and
`moodlib/ontology.py` for the model it does it with — that file is the centre of
the project and everything else serves it.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # set LIBRARY_PATH, LLM_URL, LLM_PORT
.venv/bin/python -m pytest tests/ -q
```

Needs `ffmpeg`/`ffprobe` on PATH. `run.py` re-execs under `.venv` if the calling
interpreter lacks the dependencies, so `python3 run.py …` always works.

The scan and identity tests synthesise real tagged `.m4a` fixtures with `ffmpeg`
and skip themselves without it. **Run the suite before trusting a change** — most
of these tests exist because something already went wrong once.

## Six things that will bite you

Each of these cost a real debugging session. They are counter-intuitive, so
inherited assumptions are more dangerous here than ignorance.

**1. `path` is not an identity.** Music renames files from their tags, so a tag
edit changes the path *and* the tag-hash simultaneously. The schema keys on a
surrogate `tracks.id`, resolved via `(device, inode)` first and `content_key`
second — the two cover each other's blind spot. See the table in `db.py`. On one
real re-scan another process renamed 5,782 files; all kept their identity, none
were mistaken for new tracks. Never reintroduce a path-keyed table.

**2. Bigger LLM batches are slower.** vLLM schedules against the *reserved*
`max_tokens`, so a generous budget starves concurrency and the batch decodes
almost serially. Measured: batch 20 / 8000 tokens = 0.19 tracks/s; batch 10 /
2600 = 1.08. The budget is therefore derived (`config.tag_max_tokens()`), not
configured independently. Table of measurements is in `config.py`.

**3. vLLM's grammar backend rejects some valid JSON Schema.** `uniqueItems`
returns a bare HTTP 500 with no message — it cost a full failed run to find.
Deduplication lives in `tag._unique()` instead. `maxLength` *is* supported (it
was checked). **Test any new schema construct against the live endpoint before
relying on it**; the failure mode is a 500, not a validation error.

**4. Consume futures with `as_completed`, never `map`.** `map` yields in
submission order, so one slow request blocks every finished batch behind it from
committing. That presents as a dead stall, not as slowness — the database simply
stops advancing while every worker looks busy.

**5. Ask for the leaf of a hierarchy, derive the parent.** Genre and style were
once two independent enum fields, and the model produced combinations the
taxonomy does not contain (minimal techno under `Brass & Military`). 35% of rows
disagreed with their own style. The model now picks only a style;
`ontology.genre_for_style()` derives the genre. GEMS second-order factors work
the same way. A hierarchy you compute cannot contradict itself.

**6. A request is not always a mood.** The query schema carries `genres`,
`styles`, `year_from/to` and `min_confidence` alongside the mood axes, because a
request that names a genre must *return* that genre. Without somewhere to put
it, "hip hop" is discarded and the leftover mood profile matches indie rock
equally well — that is a real bug this hit. Genre and style are OR-ed in
`db.iter_scored`: a style implies its genre, so AND-ing them cut a 2,300-track
genre to 19 candidates. `0` is the no-bound sentinel for years, because guided
decoding is fussy about nullable unions.

## Where things live

```
run.py                 entry point; re-execs under .venv
moodlib/
├── ontology.py        THE ONTOLOGY — read first, everything is shaped by it
├── config.py          the ONLY module that reads os.environ
├── db.py              schema + the three-part identity model
├── scan.py            files -> tracks, incremental, with the safety rails
├── llm.py             OpenAI-compatible client, schema-constrained
├── tag.py             resumable batch tagging
├── query.py           mood text -> structured query -> scored selection
├── playlist.py        selection -> titled .m3u8
├── progress.py        live status line (TTY) / milestone lines (redirected)
└── cli.py             argument parsing + dispatch
```

## Invariants the tests enforce

Breaking any of these fails the suite. They are asserted rather than documented
because each is invisible when broken:

- `os.environ` appears only in `config.py`; every `.env.example` key is read, and
  every setting is documented. A key nothing reads is worse than no key.
- Every adjective maps to a real GEMS dimension; the three factors partition all
  nine; every style belongs to exactly one genre.
- Genre hints reference styles that exist, and decline to guess for
  uninformative raw tags (`80s`, `127`, `PROMO`) — a wrong hint is worse than
  none, because the model defers to it.
- The scan state machine: rename, tag-edit-plus-rename, touch-without-retag,
  rewrite, soft delete, restore, and the mass-rename-is-not-mass-deletion case.
- Redirected output contains no carriage returns or ANSI escapes.
- A named genre filters; genre and style union rather than intersect; explicit
  CLI flags override anything the model inferred; a pure-mood request filters
  nothing.

## Conventions

**Configuration.** Every operational value goes in `.env` with a default in
`.env.example` and a matching CLI flag. `None` means "not given" in `cli.py`, so
the config value survives — never bake a literal into an argparse default.

**The ontology is the exception**: it lives in `ontology.py` as code because
changing it must bump `ONTOLOGY_VERSION` to invalidate stale rows, which a `.env`
edit cannot do. Bump the version when the vocabularies or the meaning of a score
change; don't bump it for a docstring.

**Repairs beat re-tagging.** A full run is ~6.5 hours. When stored data is wrong
but recoverable from what is already there, write an idempotent migration in
`db.py` (see `_repair_derived_genres`) rather than invalidating rows. Only mark
rows stale when the information genuinely is not there.

**Comments explain why, not what.** Most comments in this codebase record a
measurement or a failure — keep that. Delete a comment that only restates the
line below it.

**Never commit `.env`, `data/`, or real infrastructure detail.** Code defaults
are deliberately generic (`http://localhost`, `~/Music`); the real values live in
the git-ignored `.env`. This is a public repo.

## Testing against the live LLM

Tests are offline and hermetic. Anything touching the real endpoint is manual:

```bash
python3 run.py tag --limit 20 -v      # smallest useful end-to-end check
python3 run.py stats --full           # honesty check on tagged data
```

`stats` is the quality gate, not a vanity report. Confidence must **not** be
uniformly `2`, distributions must be spread rather than clustered mid-scale, and
no GEMS pair should correlate above 0.85. Any of those means the model is hedging
instead of characterising, and the fix is the prompt, not the scoring.
