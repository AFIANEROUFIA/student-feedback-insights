from src.anonymize import anonymize_text


def test_email_phone_handle_removed():
    out, n = anonymize_text("Write john.doe@uni.ca or call 519-888-4567, @jdoe42")
    assert "[EMAIL]" in out and "[TEL]" in out and "[PSEUDO]" in out
    assert "john.doe" not in out and "888-4567" not in out
    assert n >= 3


def test_title_keeps_title_drops_name():
    out, _ = anonymize_text("Dr. Zhong is a really good chemistry teacher.")
    assert out.startswith("Dr. [NOM]")


def test_mid_sentence_name_removed_but_tools_kept():
    out, _ = anonymize_text("Great course, thanks to Brenda Lee. The Moodle slides are good.")
    assert "Brenda" not in out and "Lee" not in out
    assert "Moodle" in out


def test_year_range_is_not_a_phone():
    out, _ = anonymize_text("The semester 2023-2024 was long.")
    assert "2023-2024" in out


def test_title_case_review_is_not_over_redacted():
    text = "Great Teacher! The Course Was Very Good And Fun"
    out, n = anonymize_text(text)
    assert out == text and n == 0
