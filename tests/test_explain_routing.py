"""
Unit tests for query_parser.apply_explain_routing().

RULE 16 says a why/interpret question is query_type "explain", but the LLM keeps
routing "Razloži gibanje prodaje pijač skozi leto 2024" to sales_trend and
"Pojasnite, zakaj stranke ... kupujejo pogosteje" to customer_analysis. That
costs the user the anomaly context, since ResultsAggregator only pairs the sales
numbers with the anomaly detector for query_type "explain".

promotion_analysis and anomaly_detection are RULE 16's documented exceptions and
must survive untouched. No LLM call is involved.
"""
import pytest

from tests.helpers import import_query_parser_module

pytestmark = pytest.mark.explain_routing

apply_explain_routing = import_query_parser_module().apply_explain_routing


@pytest.mark.parametrize("prompt", [
    # the four that still failed batch 11 after the rule-22 narrowing
    "Razloži mi, kako se je prodaja elektronike razvijala skozi leto 2023",
    "Pojasnite, zakaj stranke s kreditnimi karticami kupujejo pogosteje",
    "Pojasni, kako so se obnesle trgovine v Ljubljani",
    "Razloži gibanje prodaje pijač skozi leto 2024",
    # the rest of batch 11's phrasings
    "Zakaj je prodaja v Mariboru nižja kot v Ljubljani?",
    "Vzrok za visoko prodajo — pojasni",
    "Daj mi razlago, zakaj je Q4 vedno najboljši kvartal",
    "Interpretiraj prodajne trende za zadnje leto",
    "Razlozi mi prodajne rezultate za leto 2025",  # no diacritics
])
@pytest.mark.parametrize("llm_query_type", ["sales_trend", "comparison", "customer_analysis", "product_analysis"])
def test_why_questions_are_routed_to_explain(prompt, llm_query_type):
    result = apply_explain_routing({"query_type": llm_query_type}, prompt)

    assert result["query_type"] == "explain"


@pytest.mark.parametrize("llm_query_type", ["promotion_analysis", "anomaly_detection"])
def test_rule_16_exceptions_are_never_overridden(llm_query_type):
    # "Zakaj je bila Flash Sale manj uspešna v Kopru?" (promotion_analysis) and
    # "Zakaj je prodaja nenadoma padla?" (anomaly_detection) keep their routing.
    result = apply_explain_routing({"query_type": llm_query_type}, "Zakaj je prodaja nenadoma padla?")

    assert result["query_type"] == llm_query_type


@pytest.mark.parametrize("prompt", [
    "Kateri izdelek je imel najboljšo prodajo?",
    "Celotna prodaja v vseh trgovinah v Ljubljani",
    "Primerjaj prodajo elektronike v Ljubljani in Mariboru",
    "Zanima me prodaja igrač v Ljubljani po mesecih",
])
def test_questions_without_an_explain_marker_keep_their_query_type(prompt):
    result = apply_explain_routing({"query_type": "sales_trend"}, prompt)

    assert result["query_type"] == "sales_trend"


def test_non_string_input_is_a_safe_no_op():
    parsed = {"query_type": "sales_trend"}

    assert apply_explain_routing(parsed, None) is parsed
    assert parsed["query_type"] == "sales_trend"
