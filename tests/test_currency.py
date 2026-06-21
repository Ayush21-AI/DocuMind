from ocr.src.main.utils.currency import detect_currency, normalize_amount


def test_basic_gbp():
    assert normalize_amount("£0.50") == ("0.50 GBP", "GBP", True)


def test_thousands_separator_stripped():
    assert normalize_amount("1,234") == ("1234.00 GBP", "GBP", True)
    assert normalize_amount("£1,234.5") == ("1234.50 GBP", "GBP", True)


def test_foreign_currency_detected():
    assert normalize_amount("$28,040 USD") == ("28040.00 USD", "USD", True)
    assert normalize_amount("€100") == ("100.00 EUR", "EUR", True)


def test_currency_word():
    formatted, ccy, ok = normalize_amount("100 pounds")
    assert (formatted, ccy, ok) == ("100.00 GBP", "GBP", True)


def test_negative_parentheses_and_sign():
    assert normalize_amount("(1,234.50)")[0] == "-1234.50 GBP"
    assert normalize_amount("-50")[0] == "-50.00 GBP"


def test_two_decimals_enforced():
    assert normalize_amount("12.5")[0] == "12.50 GBP"


def test_no_number():
    assert normalize_amount("N/A") == ("", "GBP", False)
    assert normalize_amount("") == ("", "GBP", False)


def test_detect_currency_default():
    assert detect_currency("nothing here") == "GBP"
    assert detect_currency("¥500") == "JPY"
