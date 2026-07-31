import pytest

from agent.validation.response_validator import ResponseValidator, collect_numbers, extract_numbers

pytestmark = pytest.mark.validator


@pytest.fixture
def validator():
    return ResponseValidator()


ANALYSIS = {
    "row_count": 34567,
    "results": [
        {"_period": "2024-03", "total_sales": 6916634.21},
        {"_period": "2024-04", "total_sales": 42.37},
    ],
    "trend": {"total_sales": {"pct_change_per_period": -2.35}},
}


def test_grounded_summary_passes(validator):
    summary = ("Marca 2024 je prodaja znašala približno 6,9 milijona €, "
               "aprila pa le 42 €. Trend pada za 2,35 % na obdobje.")
    result = validator.validate(summary, ANALYSIS)
    assert result["valid"] is True
    assert result["numbers_unverified"] == 0


def test_hallucinated_number_is_flagged(validator):
    summary = "Prodaja je zrasla za 23,5 %."
    result = validator.validate(summary, ANALYSIS)
    assert result["valid"] is False
    assert "23,5" in result["unverified"]


def test_slovenian_thousands_format_is_understood(validator):
    result = validator.validate("Analiziranih je bilo 34.567 transakcij.", ANALYSIS)
    assert result["valid"] is True


def test_small_prose_integers_are_ignored(validator):
    result = validator.validate("Tukaj so 3 ključne ugotovitve v 2 delih.", ANALYSIS)
    assert result["numbers_checked"] == 0


def test_dates_in_results_ground_dates_in_summary(validator):
    analysis = {"anomalies": [{"period": "2024-03-15", "value": 500.0}]}
    result = validator.validate("Dne 15. marca 2024 je prišlo do skoka na 500 €.", analysis)
    assert result["valid"] is True


def test_collect_numbers_reads_numbers_inside_strings():
    numbers = collect_numbers({"period": "2024-03-15"})
    assert {2024.0, 3.0, 15.0} <= numbers


def test_extract_numbers_handles_both_locale_formats():
    values = dict(extract_numbers("1.234.567,89 in 1,234,567.89 in 6,9"))
    assert 1234567.89 in values.values()
    assert 6.9 in values.values()