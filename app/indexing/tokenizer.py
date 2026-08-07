"""Tokenization and normalization for sparse term-based retrieval.

BM25 compares terms, so indexing and querying must use exactly the same
preprocessing. This module owns that contract instead of allowing the index
and the API layer to normalize text differently.
"""

from dataclasses import dataclass, field
import re
import unicodedata


# ``\w`` is Unicode-aware in Python. The explicit pattern keeps words and
# numbers while discarding punctuation and Markdown formatting characters.
_TOKEN_PATTERN = re.compile(r"\b[\w]+\b", flags=re.UNICODE)


# These lists are deliberately local and dependency-free. They cover common
# function words without requiring a large NLP model or downloading language
# data at runtime. Users can add domain-specific words through the config.
ENGLISH_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has
have having he her here hers herself him himself his how i if in into is it its itself
me more most my myself no nor not of on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
after again against all am any because before below between both down during each few
further get got just least less might must need off rather shall since still therefore
though throughout toward towards upon
""".split())

ITALIAN_STOPWORDS = frozenset("""
a ad al allo ai agli all' alla alle anche ancora avere aveva avevano
c che chi ci cioe come con contro cui da dal dallo dai dagli dall' dalla dalle
dei degli dell' della delle di dopo dove e ed era erano essere esserci fa fai fanno
fino fra gia gli ha hai hanno ho i il in io la le lei li lo loro lui ma mi mia mie
miei mio molto ne nei nel nell' nella nelle no non nulla o ogni per piu quale quali
quando quanto quella quelle quelli quello questa queste questi questo qui quindi
sarebbe se sei sempre senza si sia siano sono sopra su sua sue sui sul sull' sulla
sulle tra tu tutte tutti tutto un una uno vi voi vostra vostre vostri vostro
perche pero pur pure quale quali
""".split())

STOPWORDS_BY_LANGUAGE = {
    "en": ENGLISH_STOPWORDS,
    "it": ITALIAN_STOPWORDS,
}


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for the shared document/query tokenizer.

    English and Italian stopword removal is enabled by default. It can be
    disabled for a domain where function words carry meaning, and additional
    languages or custom terms can be supplied without changing code.
    """

    lowercase: bool = True
    strip_accents: bool = True
    min_length: int = 1
    remove_stopwords: bool = True
    languages: tuple[str, ...] = ("en", "it")
    # Custom stopwords are added to the built-in language lists when removal
    # is enabled. They are also honored when remove_stopwords is False.
    stopwords: frozenset[str] = field(default_factory=frozenset)


def _strip_accents(value: str) -> str:
    """Convert accented characters to their base characters when possible."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def normalize_text(text: str, config: TokenizerConfig | None = None) -> str:
    """Apply Unicode normalization and case/diacritic normalization."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    settings = config or TokenizerConfig()
    normalized = unicodedata.normalize("NFKC", text)
    if settings.strip_accents:
        normalized = _strip_accents(normalized)
    if settings.lowercase:
        normalized = normalized.casefold()
    return normalized


def tokenize(text: str, config: TokenizerConfig | None = None) -> list[str]:
    """Return normalized terms for either a document or a query.

    The same function is intentionally used in both places. A token is a
    contiguous Unicode word/number sequence; punctuation is a separator. We
    do not stem by default because stemming can merge unrelated terms and
    makes citations and debugging less transparent.
    """
    settings = config or TokenizerConfig()
    normalized = normalize_text(text, settings)

    configured_stopwords: set[str] = set()
    if settings.remove_stopwords:
        for language in settings.languages:
            try:
                configured_stopwords.update(STOPWORDS_BY_LANGUAGE[language])
            except KeyError as exc:
                supported = ", ".join(sorted(STOPWORDS_BY_LANGUAGE))
                raise ValueError(f"Unsupported stopword language {language!r}; use: {supported}") from exc

    # Normalize stopwords with the same rules as document terms. Running them
    # through the token pattern also handles Italian elisions such as
    # ``dall'`` -> ``dall`` and makes ``Café`` match an accent-folded token.
    raw_stopwords = set(configured_stopwords) | set(settings.stopwords)
    configured_stopwords = {
        token
        for word in raw_stopwords
        for token in _TOKEN_PATTERN.findall(normalize_text(word, settings))
    }

    return [
        token
        for token in _TOKEN_PATTERN.findall(normalized)
        if len(token) >= settings.min_length and token not in configured_stopwords
    ]
