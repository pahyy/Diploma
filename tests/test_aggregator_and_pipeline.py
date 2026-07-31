import pytest

from agent.aggregation.results_aggregator import ResultsAggregator
from agent.response.response_formatter import ResponseFormatter
from agent.summary.summary_generator import shrink_for_prompt

pytestmark = pytest.mark.aggregator


class StubAnalyzer:

    def __init__(self, name):
        self.name = name
        self.called_with = None

    def analyze(self, params):
        self.called_with = params
        return {"from": self.name}


def make_params(query_type, intent="summarize"):
    return {
        "metrics": ["total_sales"],
        "date_filter": {"column": "transaction_date", "start": None, "end": None, "granularity": "month"},
        "filters": {}, "entity_filters": {}, "group_by": [],
        "sort_by": None, "sort_order": "desc", "limit": None,
        "comparison": {"enabled": False, "dimension": None, "groups": []},
        "metadata": {"query_type": query_type, "intent": intent},
    }


@pytest.fixture
def aggregator():
    return ResultsAggregator(
        sales_analyzer=StubAnalyzer("sales"),
        promotion_analyzer=StubAnalyzer("promotion"),
        anomaly_detector=StubAnalyzer("anomaly"),
    )


@pytest.mark.parametrize("query_type", ["sales_trend", "comparison", "product_analysis", "customer_analysis"])
def test_sales_query_types_route_to_sales_analyzer(aggregator, query_type):
    result = aggregator.run(make_params(query_type))
    assert result["analyzers_used"] == ["sales"]
    assert result["analysis"] == {"from": "sales"}


def test_promotion_analysis_routes_to_promotion_analyzer(aggregator):
    result = aggregator.run(make_params("promotion_analysis"))
    assert result["analyzers_used"] == ["promotion"]
    assert result["analysis"] == {"from": "promotion"}


def test_anomaly_detection_routes_to_anomaly_detector(aggregator):
    result = aggregator.run(make_params("anomaly_detection"))
    assert result["analyzers_used"] == ["anomaly"]
    assert result["analysis"] == {"from": "anomaly"}


def test_explain_runs_sales_and_anomaly_together(aggregator):
    result = aggregator.run(make_params("explain", intent="explain"))

    assert result["analyzers_used"] == ["sales", "anomaly"]
    assert result["analysis"] == {"from": "sales"}
    assert result["supporting"]["anomalies"] == {"from": "anomaly"}
    assert aggregator.sales_analyzer.called_with["metadata"]["intent"] == "trend"
    assert aggregator.anomaly_detector.called_with["metadata"]["intent"] == "explain"


def test_unknown_query_type_falls_back_to_sales(aggregator):
    result = aggregator.run(make_params("something_new"))
    assert result["analyzers_used"] == ["sales"]


# ---------------------------------------------------------------------------
# ResponseFormatter
# ---------------------------------------------------------------------------

def test_formatter_attaches_warning_when_validation_fails():
    formatter = ResponseFormatter()
    aggregated = {"analyzers_used": ["sales"], "analysis": {"x": 1}, "supporting": {}}
    validation = {"valid": False, "numbers_checked": 3, "numbers_unverified": 1, "unverified": ["99,9"]}

    response = formatter.format("q?", make_params("sales_trend"), aggregated,
                                "summary text", None, validation)

    assert "99,9" in response["warning"]
    assert "Opozorilo" in formatter.to_text(response)


def test_formatter_no_warning_when_valid():
    formatter = ResponseFormatter()
    aggregated = {"analyzers_used": ["sales"], "analysis": {}, "supporting": {}}
    validation = {"valid": True, "numbers_checked": 2, "numbers_unverified": 0, "unverified": []}

    response = formatter.format("q?", make_params("sales_trend"), aggregated,
                                "summary", "output/charts/x.png", validation)

    assert "warning" not in response
    assert "output/charts/x.png" in formatter.to_text(response)


# ---------------------------------------------------------------------------
# shrink_for_prompt
# ---------------------------------------------------------------------------

def test_shrink_caps_long_lists_but_keeps_head_and_tail():
    records = [{"period": i, "value": i * 10} for i in range(100)]
    shrunk = shrink_for_prompt({"results": records}, max_items=12)

    assert len(shrunk["results"]) == 13  # 6 head + marker + 6 tail
    assert shrunk["results"][0]["period"] == 0
    assert shrunk["results"][-1]["period"] == 99
    assert "izpuščenih" in shrunk["results"][6]


def test_shrink_leaves_short_structures_alone():
    data = {"results": [1, 2, 3], "total": 6}
    assert shrink_for_prompt(data) == data