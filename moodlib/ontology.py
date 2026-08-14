"""THE ONTOLOGY -- read this first.

The library records `title/artist/album/genre/year` and nothing about how a track
feels. This module defines the mood layer we manufacture on top of it, built on
three established standards rather than invented vocabulary. That is not
pedantry: the LLM has seen these exact terms and scales throughout its training
data, so it is better calibrated on them than on private coinages.

    Layer 1  Spotify / Echo Nest audio features  -- the industry lingua franca
    Layer 2  GEMS-9 (Geneva Emotional Music Scale) -- music-evoked emotion
    Layer 3  Descriptive adjectives, each mapped UP to a GEMS dimension
    Layer 4  Discogs Genre -> Style -- the record-metadata standard

Layer 3 is what makes this an ontology rather than a tag list: "brooding" is not
a loose string, it is a term under `sadness`, which is in turn under `unease`. A
query using a word the vocabulary lacks still routes to the right region through
its parent dimension.

ONTOLOGY_VERSION is the contract with the database. Bump it whenever the axes,
dimensions, vocabularies, or the meaning of a score change; every mood row
stamped with an older version is then considered stale and gets re-tagged. Adding
a purely cosmetic docstring does not warrant a bump; adding an adjective does,
because rows tagged before it existed never had the chance to use it.
"""
from __future__ import annotations

ONTOLOGY_VERSION = 1

# ---------------------------------------------------------------------------
# Layer 1 -- audio axes (Spotify / Echo Nest vocabulary)
# ---------------------------------------------------------------------------
# Spotify deprecated the /audio-features endpoint on 2024-11-27, so these cannot
# be fetched -- we generate them. The naming and semantics are kept verbatim
# because they are what everyone (and the model) reads fluently.
#
# Stored as 0.0-1.0 floats to match Spotify. The model emits 0-100 integers
# (cheaper tokens, better behaved under a JSON schema) and we divide on ingest.

AXES: dict[str, str] = {
    "energy":           "perceptual intensity and activity; fast, loud, noisy tracks score high",
    "valence":          "musical positiveness; 0 = sad/angry/depressed, 100 = happy/cheerful/euphoric",
    "danceability":     "how suitable for dancing, from tempo regularity, rhythm stability and beat strength",
    "acousticness":     "confidence the track is acoustic; 0 = pure synthetic/electronic, 100 = unamplified acoustic",
    "instrumentalness": "absence of vocals; 0 = continuous lead vocal, 100 = fully instrumental",
}

#: Short keys used on the wire. Output volume dominates tagging cost (~14
#: numbers per track), so the schema uses these and we expand on ingest.
AXIS_KEYS: dict[str, str] = {
    "e": "energy",
    "v": "valence",
    "d": "danceability",
    "ac": "acousticness",
    "in": "instrumentalness",
}

# ---------------------------------------------------------------------------
# Layer 2 -- GEMS-9, the Geneva Emotional Music Scale
# ---------------------------------------------------------------------------
# Zentner et al.'s domain-specific model of *music-evoked* emotion, validated
# precisely because general-purpose emotion scales model music badly. We use the
# 9-dimension form; each is scored 0-100.
#
# `energy` and `valence` above remain the classic Russell circumplex
# (valence x arousal) that GEMS was built to extend, which is why `tension`
# lives here as an emotion dimension rather than competing as a sixth audio axis.

GEMS: dict[str, str] = {
    "wonder":            "amazed, allured, moved; a sense of beauty or awe",
    "transcendence":     "inspired, spiritual, feeling of transcendence or otherworldliness",
    "nostalgia":         "sentimental, dreamy, longing for something past",
    "tenderness":        "affectionate, in love, mellow, softhearted",
    "peacefulness":      "calm, serene, soothing, relaxed",
    "joyful_activation": "joyful, amused, animated, bouncy, want to move",
    "power":             "strong, triumphant, energetic, fiery, heroic",
    "tension":           "agitated, nervous, tense, uneasy, irritated",
    "sadness":           "sad, sorrowful, mournful",
}

GEMS_KEYS: dict[str, str] = {
    "wo": "wonder",
    "tr": "transcendence",
    "no": "nostalgia",
    "te": "tenderness",
    "pe": "peacefulness",
    "jo": "joyful_activation",
    "po": "power",
    "tn": "tension",
    "sa": "sadness",
}

#: GEMS' own validated second-order structure. These are DERIVED (the mean of
#: their dimensions), never stored -- so the hierarchy cannot drift out of sync
#: with the values underneath it. A query may target a factor instead of
#: enumerating dimensions: "something unsettling" sets one weight, not nine.
GEMS_FACTORS: dict[str, tuple[str, ...]] = {
    "sublimity": ("wonder", "transcendence", "nostalgia", "tenderness", "peacefulness"),
    "vitality":  ("joyful_activation", "power"),
    "unease":    ("tension", "sadness"),
}


def factor_value(dimensions: dict[str, float], factor: str) -> float:
    """Mean of a second-order factor's GEMS dimensions."""
    members = GEMS_FACTORS[factor]
    return sum(dimensions.get(name, 0.0) for name in members) / len(members)


# ---------------------------------------------------------------------------
# Layer 3 -- descriptive adjectives, each mapped up to one GEMS dimension
# ---------------------------------------------------------------------------
# A closed enum, enforced by the JSON schema so the model physically cannot
# invent terms. A free-text mood field was tested first and fragmented instantly
# ("slightly ironic", "jazz-influenced"), which is unqueryable.
#
# The parent dimension is what makes a near-miss in wording cheap: a query asking
# for "yearning" still finds a track tagged only "wistful", because both sit
# under `nostalgia`.

ADJECTIVES: dict[str, str] = {
    # wonder
    "dreamy": "wonder", "luminous": "wonder",
    "psychedelic": "wonder", "quirky": "wonder",
    # transcendence
    "epic": "transcendence", "hypnotic": "transcendence",
    "meditative": "transcendence", "cinematic": "transcendence",
    # nostalgia
    "nostalgic": "nostalgia", "wistful": "nostalgia",
    "bittersweet": "nostalgia", "yearning": "nostalgia",
    # tenderness
    "warm": "tenderness", "tender": "tenderness",
    "sensual": "tenderness", "romantic": "tenderness",
    # peacefulness
    "serene": "peacefulness", "mellow": "peacefulness",
    "soothing": "peacefulness", "pastoral": "peacefulness",
    "cool": "peacefulness",
    # joyful_activation
    "joyful": "joyful_activation", "uplifting": "joyful_activation",
    "playful": "joyful_activation", "bouncy": "joyful_activation",
    "funky": "joyful_activation", "groovy": "joyful_activation",
    "euphoric": "joyful_activation",
    # power
    "triumphant": "power", "anthemic": "power", "driving": "power",
    "punchy": "power", "fierce": "power", "aggressive": "power",
    # tension
    "tense": "tension", "restless": "tension", "menacing": "tension",
    "chaotic": "tension", "gritty": "tension", "dark": "tension",
    "raw": "tension", "sleazy": "tension",
    # sadness
    "melancholic": "sadness", "sombre": "sadness", "lonely": "sadness",
    "brooding": "sadness", "desolate": "sadness", "cathartic": "sadness",
}

ADJECTIVE_TERMS: tuple[str, ...] = tuple(sorted(ADJECTIVES))

# ---------------------------------------------------------------------------
# Contexts -- the one layer with no industry standard
# ---------------------------------------------------------------------------
# Listening context is a product concept, not musicology. There is no Discogs or
# GEMS equivalent to defer to, so this is our own small closed list and is
# flagged as such rather than dressed up as a standard.

CONTEXTS: tuple[str, ...] = (
    "workout", "running", "focus", "study", "party", "club", "drive",
    "commute", "chill", "sleep", "dinner", "morning", "late_night",
    "romance", "rage", "cleaning", "background",
)

# ---------------------------------------------------------------------------
# Layer 4 -- Discogs genre / style taxonomy
# ---------------------------------------------------------------------------
# Discogs' two-level controlled vocabulary is the standard for record metadata
# and unusually strong on electronic music -- 6,495 tracks here. It earns its
# place on this library's data alone: the raw tags are 122 distinct strings with
# obvious fragmentation ("Drum & Bass" 1484 vs "Drum n Bass" 489, "Hip Hop" 1066
# vs "Hip-Hop/Rap" 571 vs "Rap" 216, and a "Raggae" typo at 269 outnumbering the
# correct "Reggae" at 181). Normalising makes genre filtering actually work.

DISCOGS_GENRES: tuple[str, ...] = (
    "Blues", "Brass & Military", "Children's", "Classical", "Electronic",
    "Folk, World, & Country", "Funk / Soul", "Hip Hop", "Jazz", "Latin",
    "Non-Music", "Pop", "Reggae", "Rock", "Stage & Screen",
)

#: Curated subset of Discogs styles -- the ones this library actually contains,
#: not the full ~500 -- grouped under the genre each belongs to. A shorter enum
#: is cheaper per request and keeps the model from reaching for styles no track
#: here would ever use.
#:
#: The nesting is the point. An earlier version asked the model for genre and
#: style as two independent fields, and it duly produced combinations the
#: taxonomy does not contain: Ricardo Villalobos filed as `Brass & Military /
#: Minimal`, and ~900 drum & bass tracks as `Jazz` because liquid D&B has jazz
#: in it. The style was almost always right and the genre almost always the
#: guess. So the model now picks only a style, and the genre is *derived* --
#: the same treatment GEMS' second-order factors get, and for the same reason:
#: a hierarchy you compute cannot contradict itself.
STYLES_BY_GENRE: dict[str, tuple[str, ...]] = {
    "Electronic": (
        "Drum n Bass", "Jungle", "Breakbeat", "Breaks", "Big Beat", "Dubstep",
        "House", "Deep House", "Progressive House", "Tech House", "Acid House",
        "Techno", "Minimal", "Trance", "Psy-Trance", "Hardcore", "Happy Hardcore",
        "Gabber", "Hardstyle", "Electro", "EBM", "Industrial", "IDM", "Ambient",
        "Downtempo", "Trip Hop", "Synth-pop", "New Wave", "Italo-Disco", "Disco",
        "Garage House", "UK Garage", "Grime", "Bassline", "Leftfield",
        "Experimental", "Euro House", "Eurodance", "Hard House", "Acid",
        "Dub Techno", "Future Bass", "Electronic Instrumental",
    ),
    "Hip Hop": (
        "Gangsta", "Boom Bap", "Trap", "Conscious", "Pop Rap",
        "Hardcore Hip-Hop", "G-Funk", "Drill",
    ),
    "Rock": (
        "Alternative Rock", "Indie Rock", "Hard Rock", "Heavy Metal",
        "Death Metal", "Black Metal", "Nu Metal", "Progressive Metal", "Punk",
        "Post-Punk", "Grunge", "Goth Rock", "Psychedelic Rock", "Classic Rock",
        "Pop Rock", "Emo", "Metalcore", "Shoegaze",
    ),
    "Pop": (
        "Ballad", "Europop", "Schlager", "Vocal", "Disco Pop", "Teen Pop",
        "Indie Pop",
    ),
    "Funk / Soul": (
        "Soul", "Funk", "Rhythm & Blues", "Contemporary R&B", "Neo Soul",
    ),
    "Reggae": ("Reggae", "Dancehall", "Dub", "Ragga", "Roots Reggae"),
    "Latin": ("Reggaeton", "Latin Pop", "Salsa"),
    "Folk, World, & Country": (
        "Afrobeat", "Afro House", "Amapiano", "Highlife", "Folk", "Country",
        "Bluegrass", "Singer-Songwriter", "Acoustic",
    ),
    "Jazz": ("Jazz-Funk", "Acid Jazz", "Smooth Jazz", "Bebop", "Swing"),
    "Blues": ("Blues Rock",),
    "Classical": ("Romantic", "Classical", "Opera", "Modern Classical", "Baroque"),
    "Stage & Screen": ("Soundtrack", "Score", "Musical", "Theme"),
    "Non-Music": ("Comedy", "Spoken Word", "Audiobook", "Education"),
    "Children's": ("Children's Music",),
}

#: Flat enum for the schema, and the reverse map that derives the genre.
DISCOGS_STYLES: tuple[str, ...] = tuple(
    style for styles in STYLES_BY_GENRE.values() for style in styles)
STYLE_GENRE: dict[str, str] = {
    style: genre for genre, styles in STYLES_BY_GENRE.items() for style in styles}


def genre_for_style(style: str) -> str:
    """The Discogs genre a style belongs to. Empty string if unknown."""
    return STYLE_GENRE.get(style, "")

#: Deterministic prior mapping raw tag -> (genre, style or None), lowercased key.
#: This is a HINT given to the model, not the answer: "80s" (1,091 tracks) and a
#: bare "Electronic" (6,495) say almost nothing, and the model knows the actual
#: artist. Where the raw tag IS informative, the hint keeps the model consistent.
RAW_GENRE_HINTS: dict[str, tuple[str, str | None]] = {
    # electronic -- including the fragmentation this layer exists to fix
    "electronic": ("Electronic", None),
    "electronica": ("Electronic", None),
    "dance": ("Electronic", None),
    "edm": ("Electronic", None),
    "club": ("Electronic", None),
    "drum & bass": ("Electronic", "Drum n Bass"),
    "drum n bass": ("Electronic", "Drum n Bass"),
    "drum and bass": ("Electronic", "Drum n Bass"),
    "jungle/drum'n'bass": ("Electronic", "Drum n Bass"),
    "jungle": ("Electronic", "Jungle"),
    "house": ("Electronic", "House"),
    "progressive house": ("Electronic", "Progressive House"),
    "afro house": ("Electronic", "Afro House"),
    "acid house": ("Electronic", "Acid House"),
    "euro house": ("Electronic", "Euro House"),
    "hard house": ("Electronic", "Hard House"),
    "techno": ("Electronic", "Techno"),
    "bouncy techno": ("Electronic", "Techno"),
    "trance": ("Electronic", "Trance"),
    "psy-trance": ("Electronic", "Psy-Trance"),
    "trance, melodic house": ("Electronic", "Trance"),
    "dubstep": ("Electronic", "Dubstep"),
    "breakbeat": ("Electronic", "Breakbeat"),
    "breaks": ("Electronic", "Breaks"),
    "big beat": ("Electronic", "Big Beat"),
    "hardcore": ("Electronic", "Hardcore"),
    "happy hardcore": ("Electronic", "Happy Hardcore"),
    "happy rave": ("Electronic", "Happy Hardcore"),
    "hardstyle": ("Electronic", "Hardstyle"),
    "electro": ("Electronic", "Electro"),
    "idm": ("Electronic", "IDM"),
    "ambient": ("Electronic", "Ambient"),
    "downtempo": ("Electronic", "Downtempo"),
    "acid": ("Electronic", "Acid"),
    "uk garage": ("Electronic", "UK Garage"),
    "grime": ("Electronic", "Grime"),
    "industrial": ("Electronic", "Industrial"),
    "abstract": ("Electronic", "Experimental"),
    "bleep": ("Electronic", "Experimental"),
    "freestyle": ("Electronic", "Electro"),
    "new age": ("Electronic", "Ambient"),
    # hip hop
    "hip hop": ("Hip Hop", None),
    "hip-hop": ("Hip Hop", None),
    "hip-hop/rap": ("Hip Hop", None),
    "rap": ("Hip Hop", None),
    "old school rap": ("Hip Hop", "Boom Bap"),
    "gangsta": ("Hip Hop", "Gangsta"),
    "trap": ("Hip Hop", "Trap"),
    "pop rap": ("Hip Hop", "Pop Rap"),
    # reggae -- note the typo outnumbers the correct spelling
    "reggae": ("Reggae", "Reggae"),
    "raggae": ("Reggae", "Reggae"),
    "dancehall": ("Reggae", "Dancehall"),
    "dance hall": ("Reggae", "Dancehall"),
    "modern dancehall": ("Reggae", "Dancehall"),
    "dub": ("Reggae", "Dub"),
    # rock / metal
    "rock": ("Rock", None),
    "alternative": ("Rock", "Alternative Rock"),
    "alternative rock": ("Rock", "Alternative Rock"),
    "indie rock": ("Rock", "Indie Rock"),
    "hard rock": ("Rock", "Hard Rock"),
    "metal": ("Rock", "Heavy Metal"),
    "heavy metal": ("Rock", "Heavy Metal"),
    "death metal": ("Rock", "Death Metal"),
    "death metal/black metal": ("Rock", "Death Metal"),
    "progressive metal": ("Rock", "Progressive Metal"),
    "nu metal": ("Rock", "Nu Metal"),
    "punk": ("Rock", "Punk"),
    "grunge": ("Rock", "Grunge"),
    "goth rock": ("Rock", "Goth Rock"),
    "new wave": ("Rock", "New Wave"),
    "crossover": ("Rock", None),
    # pop
    "pop": ("Pop", None),
    "german pop": ("Pop", None),
    "indie pop": ("Pop", "Indie Pop"),
    "europop": ("Pop", "Europop"),
    "schlager": ("Pop", "Schlager"),
    "afro-pop": ("Pop", None),
    "vocal": ("Pop", "Vocal"),
    "inspirational": ("Pop", None),
    # funk / soul / r&b
    "r&b": ("Funk / Soul", "Contemporary R&B"),
    "r&b/soul": ("Funk / Soul", "Contemporary R&B"),
    "zeitgenössischer r&b": ("Funk / Soul", "Contemporary R&B"),
    # latin
    "latin": ("Latin", None),
    "reggaeton": ("Latin", "Reggaeton"),
    "urbano latino": ("Latin", "Reggaeton"),
    "latin urban": ("Latin", "Reggaeton"),
    "pop auf spanisch": ("Latin", "Latin Pop"),
    "brasilianisch": ("Latin", None),
    # jazz / blues
    "jazz": ("Jazz", None),
    "acid jazz": ("Jazz", "Acid Jazz"),
    "blues": ("Blues", None),
    # classical
    "classical": ("Classical", "Classical"),
    "klassik": ("Classical", "Classical"),
    "opera": ("Classical", "Opera"),
    "opern": ("Classical", "Opera"),
    "zeitgenössische musik": ("Classical", "Modern Classical"),
    # folk / world / country
    "folk": ("Folk, World, & Country", "Folk"),
    "german folk": ("Folk, World, & Country", "Folk"),
    "country": ("Folk, World, & Country", "Country"),
    "bluegrass": ("Folk, World, & Country", "Bluegrass"),
    "singer-songwriter": ("Folk, World, & Country", "Singer-Songwriter"),
    "singer/songwriter": ("Folk, World, & Country", "Singer-Songwriter"),
    "acoustic": ("Folk, World, & Country", "Acoustic"),
    "traditional": ("Folk, World, & Country", None),
    "worldwide": ("Folk, World, & Country", None),
    "weltmusik": ("Folk, World, & Country", None),
    "afrobeats": ("Folk, World, & Country", "Afrobeat"),
    "afro-beat": ("Folk, World, & Country", "Afrobeat"),
    "tribal": ("Folk, World, & Country", None),
    # stage & screen / non-music
    "soundtrack": ("Stage & Screen", "Soundtrack"),
    "soundtracks": ("Stage & Screen", "Soundtrack"),
    "filmmusik": ("Stage & Screen", "Soundtrack"),
    "comedy": ("Non-Music", "Comedy"),
    "books & spoken": ("Non-Music", "Audiobook"),
    "hörspiele": ("Non-Music", "Audiobook"),
    "speech": ("Non-Music", "Spoken Word"),
    "children's": ("Children's", None),
    "instrumental": ("Electronic", "Electronic Instrumental"),
}


def genre_hint(raw: str) -> tuple[str, str | None] | None:
    """Deterministic prior for a raw genre tag, or None when uninformative.

    Decade tags ("80s"), junk ("127", "PROMO") and blanks deliberately return
    None -- there is no honest mapping, and a wrong hint is worse than none
    because the model tends to defer to it.
    """
    key = (raw or "").strip().lower()
    if not key:
        return None
    return RAW_GENRE_HINTS.get(key)


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

def _score() -> dict:
    return {"type": "integer", "minimum": 0, "maximum": 100}


def tag_schema() -> dict:
    """Strict schema for a batch of mood-tagged tracks.

    Short keys throughout: output volume dominates tagging cost, and at ~14
    numbers per track the key names are a measurable share of it.
    """
    track = {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            **{key: _score() for key in AXIS_KEYS},
            **{key: _score() for key in GEMS_KEYS},
            "m": {
                "type": "array", "minItems": 2, "maxItems": 4,
                "items": {"type": "string", "enum": list(ADJECTIVE_TERMS)},
            },
            "c": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {"type": "string", "enum": list(CONTEXTS)},
            },
            "s": {"type": "string", "enum": list(DISCOGS_STYLES)},
            "k": {"type": "integer", "minimum": 0, "maximum": 2},
        },
        "required": ["i", *AXIS_KEYS, *GEMS_KEYS, "m", "c", "s", "k"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"r": {"type": "array", "items": track}},
        "required": ["r"],
        "additionalProperties": False,
    }


def _target() -> dict:
    return {
        "type": "object",
        "properties": {
            "target": {"type": "integer", "minimum": 0, "maximum": 100},
            "weight": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["target", "weight"],
        "additionalProperties": False,
    }


def query_schema() -> dict:
    """Strict schema for a free-text mood translated into the ontology.

    Every axis and dimension carries a target AND a weight, so "loud" can pin
    energy hard while leaving acousticness free. Second-order GEMS factors are
    offered alongside the nine dimensions: a broad request like "something
    unsettling" then sets one weight instead of guessing nine.
    """
    return {
        "type": "object",
        "properties": {
            **{name: _target() for name in AXES},
            **{name: _target() for name in GEMS},
            **{name: _target() for name in GEMS_FACTORS},
            "adjectives": {
                "type": "array", "maxItems": 6,
                "items": {"type": "string", "enum": list(ADJECTIVE_TERMS)},
            },
            "avoid": {
                "type": "array", "maxItems": 6,
                "items": {"type": "string", "enum": list(ADJECTIVE_TERMS)},
            },
            "contexts": {
                "type": "array", "maxItems": 3,
                "items": {"type": "string", "enum": list(CONTEXTS)},
            },
            "title": {"type": "string", "maxLength": 48},
            "rationale": {"type": "string"},
        },
        "required": [
            *AXES, *GEMS, *GEMS_FACTORS,
            "adjectives", "avoid", "contexts", "title", "rationale",
        ],
        "additionalProperties": False,
    }


def tag_instructions() -> str:
    """The prompt preamble for tagging. Kept beside the vocabularies it describes."""
    axes = "\n".join(f"  {k} = {AXES[name]}" for k, name in AXIS_KEYS.items())
    gems = "\n".join(f"  {k} = {GEMS[name]}" for k, name in GEMS_KEYS.items())
    return (
        "Mood-tag each track for playlist curation. Return one object per input "
        "track, in the same order, with `i` echoing the input index.\n\n"
        "Audio axes (0-100), Spotify audio-feature definitions:\n" + axes + "\n\n"
        "Emotion (0-100 each), GEMS-9 music-evoked emotion scale — score how "
        "strongly the track evokes each:\n" + gems + "\n\n"
        "m = 2-4 descriptive adjectives from the vocabulary.\n"
        "c = 1-3 listening contexts.\n"
        "s = the Discogs style that best fits. Its parent genre is derived "
        "from it, so pick the style and do not worry about the genre. The "
        "track's own genre tag is given as `raw` where useful; it is often "
        "wrong, vague or a decade — prefer what you know about the artist "
        "and record.\n"
        "k = confidence: 2 = you recognise this specific recording, "
        "1 = you know the artist or style but not this track, "
        "0 = pure inference from the genre tag.\n\n"
        "Be honest with k — most of a large collection is 1 or 0, and a "
        "uniformly confident library is worse than an honest one. Use the full "
        "0-100 range rather than clustering everything near the middle."
    )


def query_instructions(text: str) -> str:
    """The prompt for translating a listener's free-text mood into the ontology."""
    return (
        "Translate the listener's mood request into a structured query over a "
        "music-mood ontology.\n\n"
        "Every axis and emotion takes a `target` (0-100) and a `weight` (0-1). "
        "Weight 0 means ignore this dimension entirely — use it freely; most "
        "requests genuinely constrain only a few. Weight 1 means critical.\n\n"
        "Audio axes: " + ", ".join(f"{n} ({d})" for n, d in AXES.items()) + "\n\n"
        "GEMS-9 emotions: " + ", ".join(GEMS) + "\n"
        "You may instead set a second-order factor and leave its members at "
        "weight 0: sublimity (wonder/transcendence/nostalgia/tenderness/"
        "peacefulness), vitality (joyful_activation/power), unease "
        "(tension/sadness).\n\n"
        "adjectives = vocabulary terms to favour.\n"
        "avoid = terms to penalise. Use this ONLY for what the request "
        "explicitly rules out (\"not depressing\", \"nothing aggressive\"). Do "
        "not fill it with the opposites of what was asked for — the axis and "
        "emotion targets already handle that, and a term listed here will "
        "actively reject otherwise perfect tracks. Usually empty.\n"
        "Include only terms that genuinely apply — fewer is better. Do not pad "
        "these lists to their maximum length; an empty list is a valid answer.\n\n"
        "title = a short, evocative name for the resulting playlist, 2-5 words, "
        "in the spirit of the request rather than a restatement of it. Title "
        "Case, no quotes, no emoji, no trailing punctuation. It becomes the "
        "playlist's name in the music library, so make it something a person "
        "would be happy to see in a sidebar.\n"
        "rationale = one or two sentences explaining your reading of the request.\n\n"
        f"Request: {text}"
    )
