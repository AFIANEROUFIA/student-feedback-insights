from src.preprocessing import clean_text, normalize_label, normalize_text, tokenize


def test_keeps_negation_tokens():
    tokens = tokenize("The teacher is not helpful.")
    assert "not" in tokens and "helpful" in tokens


def test_contraction_is_expanded_to_negation():
    # Bug corrigé : « isn't » n'était pas reconnu comme négation.
    assert "not" in tokenize("The course isn't clear at all.")
    assert "not" in tokenize("The teacher doesn't help.")


def test_curly_and_escaped_apostrophes():
    assert clean_text("Ce cours n’est pas clair.") == "ce cours n'est pas clair."
    assert normalize_text("He doesn\\'t explain") == "He doesn't explain"


def test_normalize_label_eduRABSA_values():
    assert normalize_label("Pos") == "Positive"
    assert normalize_label("Neg") == "Negative"
    assert normalize_label("Neu") == "Neutral"
    # Bugs corrigés : « Conf/Neu » devenait un label inconnu, « [NA] » devenait « [Na] ».
    assert normalize_label("Conf/Neu") == "Neutral"
    assert normalize_label("[NA]") is None
    assert normalize_label(float("nan")) is None


def test_clean_spaces():
    assert clean_text("  Hello   world  ") == "hello world"
