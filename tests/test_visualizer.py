import os

import pytest

from agent.visualization.visualizer import Visualizer

pytestmark = pytest.mark.visualizer


def make_params(metrics=None, query_type="sales_trend"):
    return {
        "metrics": metrics or ["total_sales"],
        "date_filter": {"column": "transaction_date", "start": None, "end": None, "granularity": "month"},
        "filters": {}, "entity_filters": {}, "group_by": [],
        "sort_by": None, "sort_order": "desc", "limit": None,
        "comparison": {"enabled": False, "dimension": None, "groups": []},
        "metadata": {"query_type": query_type, "intent": "summarize"},
    }


def wrap(analysis, supporting=None):
    return {"analyzers_used": ["x"], "analysis": analysis, "supporting": supporting or {}}


@pytest.fixture
def viz(tmp_path):
    return Visualizer(output_dir=str(tmp_path))


def assert_is_png(path):
    assert path is not None and os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_trend_records_make_a_line_chart(viz):
    analysis = {"query_type": "sales_trend", "row_count": 3, "results": [
        {"_period": "2024-01", "total_sales": 100.0},
        {"_period": "2024-02", "total_sales": 120.0},
        {"_period": "2024-03", "total_sales": 90.0},
    ]}
    assert_is_png(viz.create_chart(wrap(analysis), make_params()))


def test_grouped_records_make_a_bar_chart(viz):
    analysis = {"query_type": "sales_trend", "row_count": 2, "results": [
        {"product_category": "Electronics", "total_sales": 500.0},
        {"product_category": "Food", "total_sales": 300.0},
    ]}
    assert_is_png(viz.create_chart(wrap(analysis), make_params()))


def test_comparison_groups_make_a_bar_chart(viz):
    analysis = {"query_type": "comparison", "comparison_dimension": "store_city", "row_count": 10,
                "results": [
                    {"label": "Ljubljana", "row_count": 6, "total_sales": 760.0},
                    {"label": "Maribor", "row_count": 4, "total_sales": 670.0},
                ]}
    assert_is_png(viz.create_chart(wrap(analysis), make_params(query_type="comparison")))


def test_promotion_summary_makes_an_uplift_chart(viz):
    analysis = {"query_type": "promotion_analysis", "by_promotion_type": [
        {"promotion_type": "20% Off", "avg_uplift_pct": 12.5},
        {"promotion_type": "Flash Sale", "avg_uplift_pct": -3.1},
    ]}
    assert_is_png(viz.create_chart(wrap(analysis), make_params(query_type="promotion_analysis")))


def test_anomaly_series_makes_a_marked_line_chart(viz):
    analysis = {"query_type": "anomaly_detection",
                "series": [{"period": f"2024-01-{d:02d}", "value": 100.0} for d in range(1, 20)],
                "anomalies": [{"period": "2024-01-10", "value": 100.0, "z_score": 4.0, "type": "spike"}]}
    assert_is_png(viz.create_chart(wrap(analysis), make_params(query_type="anomaly_detection")))


def test_single_total_needs_no_chart(viz):
    analysis = {"query_type": "sales_trend", "row_count": 100, "results": [{"total_sales": 12345.0}]}
    assert viz.create_chart(wrap(analysis), make_params()) is None


def test_same_result_reuses_same_file(viz):
    analysis = {"query_type": "sales_trend", "row_count": 3, "results": [
        {"_period": "2024-01", "total_sales": 100.0},
        {"_period": "2024-02", "total_sales": 120.0},
    ]}
    path_1 = viz.create_chart(wrap(analysis), make_params())
    path_2 = viz.create_chart(wrap(analysis), make_params())
    assert path_1 == path_2