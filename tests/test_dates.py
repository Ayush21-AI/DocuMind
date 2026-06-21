from ocr.src.main.utils.dates import is_canonical, normalize_date


def test_already_canonical():
    assert normalize_date("31/12/2024") == "31/12/2024"
    assert normalize_date("03/04/2024") == "03/04/2024"  # UK: 3 April, untouched


def test_spelled_out_months():
    assert normalize_date("December 31, 2024") == "31/12/2024"
    assert normalize_date("31 December 2024") == "31/12/2024"


def test_iso_format():
    assert normalize_date("2024-12-31") == "31/12/2024"


def test_us_ordering_corrected():
    # 17 cannot be a month, so this US date is recovered to UK order
    # (fixes the old swap logic that dropped it).
    assert normalize_date("10/17/2020") == "17/10/2020"


def test_two_digit_year():
    assert normalize_date("1/2/24") == "01/02/2024"


def test_unparseable_returns_empty():
    assert normalize_date("") == ""
    assert normalize_date("N/A") == ""
    assert normalize_date("13/13/2024") == ""


def test_is_canonical():
    assert is_canonical("31/12/2024")
    assert not is_canonical("2024-12-31")
    assert not is_canonical("32/12/2024")
