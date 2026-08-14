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

No external binaries needed to run — tags are read with mutagen. `run.py`
re-execs under `.venv` if the calling interpreter lacks the dependencies, so
`python3 run.py …` always works.

`ffmpeg` is needed only to *synthesise* fixtures for the scan and identity
tests, which skip themselves without it. **Run the suite before trusting a change** — most
of these tests exist because something already went wrong once.

## Eight things that will bite you

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

**6. Tags are read with mutagen, not ffprobe.** Not for the reason the sibling
project warns about — that warning is about *writing* tags with ffmpeg, and this
only reads. The reason is the process spawn per file: measured 200 vs 38 files/s
on this library, with both agreeing on every raw field including the freeform
`INITIALKEY` atoms. Don't reintroduce a subprocess here. Note freeform atoms are
exactly what a generic reader drops, so `_musical_key` searches key names rather
than assuming a spelling.

**7. Genre words can be alternatives, a hierarchy, or a compound.** All three
arrive in the same two fields. "jungle or breakbeat" is alternatives and "hip
hop, boom bap kind" is a hierarchy — both OR. "ambient drum and bass" and "jazzy
house" are compounds, where one word modifies the other, and OR-ing them lets the
modifier win: the first returned Bonobo and Apparat with no d&b at all. The
prompt handles a modifier filed as a *style*; a modifier filed as a *genre* is
caught structurally in `query._drop_modifier_genres`, since the taxonomy knows
`Jazz` does not parent `House`. Prompting alone was not reliable for that.

**8. A request is not always a mood.** The query schema carries `genres`,
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
wsgi.py                WSGI entry point for gunicorn
gunicorn.conf.py       server settings, all of them from config.py
deploy/                systemd unit templates
moodlib/
├── ontology.py        THE ONTOLOGY — read first, everything is shaped by it
├── config.py          the ONLY module that reads os.environ
├── db.py              schema + the three-part identity model
├── scan.py            files -> tracks, incremental, with the safety rails
├── llm.py             OpenAI-compatible client, schema-constrained
├── tag.py             resumable batch tagging
├── query.py           mood text -> structured query -> scored selection
├── playlist.py        selection -> titled .m3u8
├── lock.py            one library writer at a time, across processes
├── progress.py        live status line (TTY) / milestone lines (redirected)
└── cli.py             argument parsing + dispatch
```

## The web UI

`webapp/` is a thin Flask front end. Keep it thin: every route should be a call
into `moodlib` plus a template render. Logic that lands here instead of in
`moodlib` is logic the CLI and the eval harness will not have.

The design decision worth preserving: **a playlist is derived, never stored.**
`/playlist/<query_id>?seed=N` re-scores from the saved translation in the
`queries` table, so a re-roll costs no model call, every URL is reproducible, and
there is no session state. Adding a `playlists` table would break all three.

The UI reads nothing from the environment itself — `config.py` is still the only
module that does, and its settings are the `WEB_*` keys. Two tests police this:
one asserts every `WEB_*` setting exists, another that the key-scanning regex
knows about every reader helper in `config.py` (a missing one makes a real key
look undocumented, which is exactly how `WEB_DEBUG` first failed).

Styling comes from `../ui-template` (Bootstrap 5.3 + the shared `custom.css`);
`webapp/static/css/app.css` holds only what the track grid needs. The track
index and confidence badges carry their own translucent dark background because
they sit on cover art, which is arbitrary user imagery — do not assume a
readable backdrop.

## Deployment

`.venv/bin/gunicorn -c gunicorn.conf.py wsgi:app`, with the unit templates in
`deploy/`. Three things constrain it, and all three are invisible until they
bite:

**One worker, with threads.** `webapp/jobs.py` keeps the current job in module
memory, so a second worker would answer status polls about a run it cannot see,
and `_lock` is a `threading.Lock`, which does not span processes — two workers
could start two scans. This is a genuine ceiling, not a shortcut: lifting it
means moving job state into the database. `workers = 1` in `gunicorn.conf.py` is
therefore not a tuning knob.

**`gunicorn.conf.py` imports config as `settings`.** Gunicorn reads that
module's globals as settings and `config` is one of its own (`-c`), so binding
that name makes it reject the file with `Not a string` — an error that says
nothing about the cause. Every value there still comes from `config.py`; the
invariant holds outside `moodlib/` too.

**Scan and tag take a cross-process lock** (`moodlib/lock.py`), because the
service, the nightly timer and an SSH session can all write the same database.
The lock is skipped when a connection is passed in — that is how the tests drive
a scan against their own database — so it is `build()`/`run()` that take it and
`_build()`/`_run()` that do the work. It is an advisory `flock`, deliberately:
the kernel releases it when the fd closes, so a `kill -9` mid-tag does not wedge
every later run behind a stale lock file.

The worker timeout is derived from `LLM_TIMEOUT`
(`config.web_request_timeout()`), because gunicorn's 30s default is well under
the 240s a mood translation may spend waiting on the model — that is a 502 on a
request that was about to succeed. Access logging defaults off: the page polls
`/jobs/status` every `WEB_POLL_SECONDS`, which is ~43,000 journal lines a day
from one idle tab.

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
- The write lock excludes a second process, frees on release, and survives its
  holder being killed — the last one is why it is a `flock` and not a lock file.
- The home page reloads only for a job transition it watched happen. The runner
  keeps the last job forever, so "a job exists" is not "a job just finished";
  reading it that way reloaded the page on every load.

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
python3 run.py tag --limit 20 -v          # smallest useful end-to-end check
python3 run.py stats --full               # honesty check on tagged data
.venv/bin/python tools/evaluate.py        # playlist quality, measured
.venv/bin/python tools/evaluate.py workout --verbose
```

`tools/evaluate.py` runs a fixed set of requests against expectations written
down in advance. Use it whenever you touch `query_instructions`, the scoring, or
the ontology — a prompt change that "feels better" is worth nothing next to a
check that moves. Judging by reading playlists is how a soft house-pop record sat
at line 19 of a high-intensity workout list unnoticed.

Two rules for the eval itself. **Expect run-to-run variance**: the translation is
an LLM call, so a single failing check may be noise — re-run before chasing it.
And **when a check fails, ask whether the check is wrong first**. One capped
sadness at 40 for workout requests and duly flagged Nine Inch Nails; aggressive
music is often dark, and the honest fix was deleting the check, not tuning the
system to satisfy it.

`stats` is the quality gate, not a vanity report. Confidence must **not** be
uniformly `2`, distributions must be spread rather than clustered mid-scale, and
no GEMS pair should correlate above 0.85. Any of those means the model is hedging
instead of characterising, and the fix is the prompt, not the scoring.
