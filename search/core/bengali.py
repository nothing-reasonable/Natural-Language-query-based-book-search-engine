"""Bengali text handling: normalisation, tokenisation and a light rule-based stemmer.

Deliberately dependency-free and small -- every rule is visible and editable, which
matters more here than squeezing out the last few points of recall.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- normalisation

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))

# Characters that appear in Bengali text but should be folded to a single form.
_CHAR_FOLD = {
    "\u09f0": "\u09b0",  # Assamese RA -> Bengali RA
    "\u09f1": "\u09ac",  # Assamese VA -> BA
    "\u09ce": "\u09a4",  # KHANDA TA -> TA
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
}
_CHAR_FOLD_TABLE = str.maketrans(_CHAR_FOLD)

# ০..৯ -> 0..9 so that "১৯৭১" and "1971" are the same token.
_DIGIT_TABLE = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Sentence/clause punctuation that should simply become whitespace.
_PUNCT = re.compile(r"[।॥,;:!?\"'`()\[\]{}<>/\|@#$%^&*_+=~\u00ab\u00bb\-–—.]+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form for storage *and* matching. Rendering is unchanged."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_ZERO_WIDTH)
    text = text.translate(_CHAR_FOLD_TABLE)
    return _SPACE.sub(" ", text).strip()


def fold_digits(text: str) -> str:
    return text.translate(_DIGIT_TABLE)


# --------------------------------------------------------------------------- tokenisation

# Bengali block, ASCII letters (banglish / English terms) and digits.
_TOKEN = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+|[0-9]+")


# Applied only when analysing (never when storing/displaying): visarga is used both as
# an abbreviation mark ("মোঃ" == "মো.") and inside words ("দুঃখ"/"দুখ"), so folding it
# away makes both spellings collide on purpose.
_ANALYSIS_FOLD = str.maketrans({"ঃ": ""})


def tokenize(text: str) -> list[str]:
    text = fold_digits(normalize(text)).lower().translate(_ANALYSIS_FOLD)
    text = _PUNCT.sub(" ", text)
    return _TOKEN.findall(text)


# --------------------------------------------------------------------------- stemming

# Longest match wins. Bengali has no widely available stemmer, so this is a small
# hand-written list of inflectional endings. It is intentionally a little aggressive
# (e.g. "কিশোর" -> "কিশো"): the same function runs over documents *and* queries, so a
# slightly over-trimmed stem still matches, while under-trimming loses recall
# ("মুক্তিযুদ্ধের" vs "মুক্তিযুদ্ধে" vs "মুক্তিযুদ্ধ").
_SUFFIXES = (
    "গুলোকে", "গুলিকে", "গুলোর", "গুলির", "দেরকে", "খানাতে",
    "গুলো", "গুলি", "খানা", "খানি", "টুকু", "ভাবে", "দের",
    "টাকে", "টিকে", "টিতে", "টাতে", "েরই", "তেই", "রাই", "কেই",
    "টার", "টির", "য়ে", "ের", "রা", "কে", "তে", "টি", "টা",
    "র", "ে",
)
_MIN_STEM = 3
_MAX_STRIP_ROUNDS = 3


def stem(token: str) -> str:
    """Strip inflectional suffixes until none match, never shortening below `_MIN_STEM`.

    Stripping must be repeated, not done once: "একাত্তরের" loses "ের" to give "একাত্তর",
    which still carries a "র" -- if we stopped after one pass the two spellings of the
    same word would land on different stems and never match each other.
    """
    for _ in range(_MAX_STRIP_ROUNDS):
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
                token = token[: -len(suffix)]
                break
        else:
            break
    return token


# --------------------------------------------------------------------------- stopwords

STOPWORDS: frozenset[str] = frozenset(
    """
    এই এবং ও বা কিন্তু তবে তবু যদি যখন তখন যেহেতু কারণ যে যা যার যাদের যাকে
    তার তাদের তাকে তিনি তারা সে এরা ইহা এটা এটি ওটা সেই সেটি সেটা
    আমি আমরা আমার আমাদের তুমি তোমরা তোমার আপনি আপনারা আপনার
    এর ে তে থেকে জন্য সাথে সঙ্গে দিয়ে নিয়ে ছাড়া প্রতি দ্বারা মাধ্যমে
    করে করা করেন করেছেন করছে করবে হয় হয়ে হয়েছে হবে ছিল ছিলেন থাকে আছে নেই
    একটি একটা একজন কিছু কোন কোনো কি কী কেন কিভাবে কীভাবে কোথায় কখন কেমন
    উপর নিচে মধ্যে ভিতরে বাইরে পর আগে পরে সময় মত মতো
    না নয় নাই আর ও আরও অনেক সব সকল প্রতিটি খুব বেশ যেমন তেমন
    বই বইটি বইটির লেখা লেখক সম্পর্কে বিষয়ে নামে চাই খুঁজছি দাও দেখাও
    এমন যেসব যেগুলো যাঁরা যারা যাঁর কারা ধরনের রকম বিভিন্ন সংক্রান্ত জাতীয়
    """.split()
)


def analyze_pairs(text: str) -> list[tuple[str, str]]:
    """(stem, surface form) for every indexable token."""
    pairs = []
    for token in tokenize(text):
        if token in STOPWORDS or len(token) < 2:
            continue
        stemmed = stem(token)
        if stemmed not in STOPWORDS:
            pairs.append((stemmed, token))
    return pairs


def analyze(text: str) -> list[str]:
    """normalize -> tokenize -> drop stopwords -> stem. Used by the lexical index."""
    return [stemmed for stemmed, _ in analyze_pairs(text)]


def surface_forms(terms: list[str]) -> dict[str, str]:
    """stem -> the word a human actually typed, so explanations never show "কিশো"."""
    mapping: dict[str, str] = {}
    for term in terms:
        for stemmed, surface in analyze_pairs(term):
            mapping.setdefault(stemmed, surface)
    return mapping


def key(text: str) -> str:
    """Order-insensitive matching key -- used for dedup and taxonomy alias lookup."""
    return " ".join(sorted(set(analyze(text))))
