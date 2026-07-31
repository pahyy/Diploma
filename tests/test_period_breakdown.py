import pytest

from tests.helpers import import_query_parser_module

pytestmark = pytest.mark.period_breakdown

apply_period_breakdown = import_query_parser_module().apply_period_breakdown


def make_parsed(intent="summarize", granularity="month"):
    """Minimal finalize_query-shaped dict — only the fields this function touches."""
    return {"intent": intent, "time": {"start": None, "end": None, "granularity": granularity}}


@pytest.mark.parametrize("prompt, expected_granularity", [
    ("Zanima me prodaja igrac v ljubljani po mesecih", "month"),
    ("Prikaži prodajo po mesecih", "month"),
    ("Kakšna je mesečna prodaja pijač?", "month"),
    ("Prodaja po posameznih mesecih", "month"),
    ("Prodaja po tednih v letu 2024", "week"),
    ("Tedenska prodaja elektronike", "week"),
    ("Prodaja po dnevih", "day"),
    ("Po dneh v januarju 2024", "day"),
    ("Dnevna prodaja v Kopru", "day"),
    ("Prodaja po letih", "year"),
    ("Letna prodaja igrač", "year"),
])
def test_period_phrase_forces_trend_intent_and_granularity(prompt, expected_granularity):
    result = apply_period_breakdown(make_parsed(), prompt)

    assert result["intent"] == "trend"
    assert result["time"]["granularity"] == expected_granularity


def test_unitless_time_phrase_forces_trend_but_keeps_granularity():
    result = apply_period_breakdown(make_parsed(granularity="week"), "Kako se je prodaja gibala skozi čas?")

    assert result["intent"] == "trend"
    assert result["time"]["granularity"] == "week"


@pytest.mark.parametrize("prompt", [
    "Kateri izdelek je imel najboljšo prodajo?",
    "Celotna prodaja v vseh trgovinah v Ljubljani",
    "Kateri mesec v letu 2024 je prinesel največ prihodkov?",
    "Poletni izdelki v Kopru",
    "Primerjaj prodajo elektronike v Ljubljani in Mariboru",
])
def test_queries_without_a_period_breakdown_are_left_alone(prompt):
    result = apply_period_breakdown(make_parsed(intent="summarize"), prompt)

    assert result["intent"] == "summarize"


def test_existing_trend_intent_still_gets_the_named_granularity():
    result = apply_period_breakdown(make_parsed(intent="trend", granularity="day"), "Prodaja po mesecih")

    assert result["intent"] == "trend"
    assert result["time"]["granularity"] == "month"


def test_non_string_input_is_a_safe_no_op():
    parsed = make_parsed()

    assert apply_period_breakdown(parsed, None) is parsed
    assert parsed["intent"] == "summarize"
