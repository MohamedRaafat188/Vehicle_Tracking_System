import pytest

from src.inference.plate_postprocess import PlatePostProcessor


@pytest.fixture
def processor():
    return PlatePostProcessor()


def test_already_valid_pattern1_passes_through(processor):
    result = processor.process("5545GZN")
    assert result.valid is True
    assert result.text == "5545GZN"
    assert result.matched_format == "pattern1"


def test_already_valid_pattern2_passes_through(processor):
    result = processor.process("MA4844CC")
    assert result.valid is True
    assert result.matched_format == "pattern2"


def test_corrects_letter_o_in_digit_position(processor):
    # "O545GZN": leading digit misread as letter O -> should become "0545GZN"
    result = processor.process("O545GZN")
    assert result.valid is True
    assert result.text == "0545GZN"


def test_corrects_digit_in_letter_position(processor):
    # "5545G2N": last letter-position character misread as digit 2 -> letter Z
    result = processor.process("5545G2N")
    assert result.valid is True
    assert result.text == "5545GZN"


def test_corrects_within_pattern2_digit_block(processor):
    # "MA4B44CC": one digit misread as letter B -> should become 8
    result = processor.process("MA4B44CC")
    assert result.valid is True
    assert result.text == "MA4844CC"


def test_uncorrectable_text_is_invalid(processor):
    result = processor.process("12345")
    assert result.valid is False


def test_lowercase_input_is_normalized(processor):
    result = processor.process("5545gzn")
    assert result.valid is True
    assert result.text == "5545GZN"
