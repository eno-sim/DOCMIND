from app.indexing.tokenizer import TokenizerConfig, normalize_text, tokenize


def test_tokenizer_is_shared_friendly_for_document_and_query():
    text = "Café, DOCMIND!\nretrieval"
    assert normalize_text(text) == "cafe, docmind!\nretrieval"
    assert tokenize(text) == ["cafe", "docmind", "retrieval"]


def test_english_and_italian_stopwords_are_removed_by_default():
    assert tokenize("The document and la riunione") == ["document", "riunione"]


def test_stopword_removal_can_be_disabled_or_extended():
    without_builtin = TokenizerConfig(remove_stopwords=False)
    assert tokenize("The document", without_builtin) == ["the", "document"]

    with_custom = TokenizerConfig(stopwords=frozenset({"Café"}))
    assert tokenize("Café document", with_custom) == ["document"]
