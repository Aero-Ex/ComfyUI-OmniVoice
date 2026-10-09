import difflib
import logging
import re

from .lang_map import LANG_IDS, LANG_NAME_TO_ID

# Language/instruct conditioning helpers. Behavior mirrors the reference
# implementation (omnivoice voice-design and generation-config logic).

ZH_RE = re.compile(r"[\u4e00-\u9fff]")

_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {"whisper": "耳语"},
    {
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    },
    {
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    },
]

_INSTRUCT_EN_TO_ZH = {}
_INSTRUCT_ZH_TO_EN = {}
_INSTRUCT_MUTUALLY_EXCLUSIVE = []
for _cat in _INSTRUCT_CATEGORIES:
    if isinstance(_cat, dict):
        _INSTRUCT_EN_TO_ZH.update(_cat)
        _INSTRUCT_ZH_TO_EN.update({v: k for k, v in _cat.items()})
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_cat) | set(_cat.values()))
    else:
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_cat))

_INSTRUCT_ALL_VALID = (
    set(_INSTRUCT_EN_TO_ZH)
    | set(_INSTRUCT_ZH_TO_EN)
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-2]
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-1]
)

_INSTRUCT_VALID_EN = frozenset(i for i in _INSTRUCT_ALL_VALID if not ZH_RE.search(i))
_INSTRUCT_VALID_ZH = frozenset(i for i in _INSTRUCT_ALL_VALID if ZH_RE.search(i))

# Per-category option lists for UI dropdowns (order matches _INSTRUCT_CATEGORIES:
# gender, age, pitch, style, accents, dialects).
INSTRUCT_GENDER = sorted(_INSTRUCT_CATEGORIES[0])
INSTRUCT_AGE = sorted(_INSTRUCT_CATEGORIES[1])
INSTRUCT_PITCH = sorted(_INSTRUCT_CATEGORIES[2])
INSTRUCT_STYLE = sorted(_INSTRUCT_CATEGORIES[3])
INSTRUCT_ACCENTS = sorted(_INSTRUCT_CATEGORIES[4])
INSTRUCT_DIALECTS = sorted(_INSTRUCT_CATEGORIES[5])

END_PUNCTUATION = {
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "…",
    ")",
    "]",
    "}",
    '"',
    "'",
    "“",
    "”",
    "‘",
    "’",
    "；",
    "：",
    "，",
    "。",
    "！",
    "？",
    "、",
    "……",
    "）",
    "】",
}


def resolve_language(language):
    if language is None or language.lower() == "none":
        return None
    if language in LANG_IDS:
        return language
    key = language.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    logging.warning(
        "Language '%s' is not recognized. "
        "Please use a valid language ID (e.g., 'en', 'zh', 'ja', 'de') "
        "or a full language name (e.g., 'English', 'Chinese', 'Japanese'). "
        "Falling back to None (language-agnostic mode).",
        language,
    )
    return None


def resolve_instruct(instruct, use_zh=False):
    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [x for x in raw_items if x]

    unknown = []
    normalised = []
    for raw in raw_items:
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalised.append(n)
        else:
            sug = difflib.get_close_matches(n, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, sug[0] if sug else None))

    if unknown:
        lines = []
        for raw, n, sug in unknown:
            if sug:
                lines.append("  '{}' -> '{}' (unsupported; did you mean '{}'?)".format(raw, n, sug))
            else:
                lines.append("  '{}' -> '{}' (unsupported)".format(raw, n))
        raise ValueError(
            "Unsupported instruct items found in {}:\n".format(instruct_str)
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. "
            "'male, indian accent'),\nChinese instructs should use full-width "
            "comma (e.g. '男，河南话')."
        )

    has_dialect = any(n.endswith("话") for n in normalised)
    has_accent = any(" accent" in n for n in normalised)

    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech."
        )

    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    if use_zh:
        normalised = [_INSTRUCT_EN_TO_ZH.get(n, n) for n in normalised]
    else:
        normalised = [_INSTRUCT_ZH_TO_EN.get(n, n) for n in normalised]

    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalised if n in cat]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = []
        for group in conflicts:
            parts.append(" vs ".join("'{}'".format(x) for x in group))
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) "
            "allows at most one item."
        )

    has_zh = any(any("\u4e00" <= c <= "\u9fff" for c in n) for n in normalised)
    separator = "，" if has_zh else ", "

    return separator.join(normalised)


def add_punctuation(text):
    text = text.strip()

    if not text:
        return text

    if text[-1] not in END_PUNCTUATION:
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)
        text += "。" if is_chinese else "."

    return text


def combine_text(text, ref_text=None):
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()
    full_text = re.sub(r"[\r\n]+", "", full_text)
    return full_text.replace("\uff08", "(").replace("\uff09", ")")
