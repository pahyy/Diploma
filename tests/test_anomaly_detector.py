import pandas as pd
import pytest

from agent.analyzers.anomaly_detector import AnomalyDetector

pytestmark = pytest.mark.anomaly_detector

SPIKE_DAY = "2023-03-01"
DROP_DAY = "2023-04-15"


def make_params(**overrides):
    params = {
        "metrics": ["total_sales"],
        "date_filter": {"column": "transaction_date", "start": None, "end": None, "granularity": "month"},
        "filters": {},
        "entity_filters": {},
        "group_by": [],
        "sort_by": None,
        "sort_order": "desc",
        "limit": None,
        "comparison": {"enabled": False, "dimension": None, "groups": []},
        "metadata": {"query_type": "anomaly_detection", "intent": "detect"},
    }
    params.update(overrides)
    return params


@pytest.fixture
def anomaly_df():
    rows = []
    days = pd.date_range("2023-01-01", "2023-04-30", freq="D")

    for i, day in enumerate(days):
        day_str = day.strftime("%Y-%m-%d")
        base = 100.0 + (i % 3)

        lj_sales = base
        mb_sales = base
        if day_str == SPIKE_DAY:
            lj_sales = 1000.0   # spike comes from Ljubljana only
        if day_str == DROP_DAY:
            lj_sales = 10.0     # drop hits both stores
            mb_sales = 10.0

        rows.append({"transaction_date": day, "total_sales": lj_sales, "quantity": 1,
                     "store_city": "Ljubljana", "product_category": "Food",
                     "holiday_season": "No", "weekend": "No", "promotion_type": "20% Off"})
        rows.append({"transaction_date": day, "total_sales": mb_sales, "quantity": 1,
                     "store_city": "Maribor", "product_category": "Electronics",
                     "holiday_season": "No", "weekend": "No", "promotion_type": "Flash Sale"})

    return pd.DataFrame(rows)


def anomalies_by_period(result):
    return {a["period"]: a for a in result["anomalies"]}


def test_spike_and_drop_are_both_detected(anomaly_df):
    result = AnomalyDetector(anomaly_df).analyze(make_params())

    found = anomalies_by_period(result)
    assert SPIKE_DAY in found
    assert found[SPIKE_DAY]["type"] == "spike"
    assert DROP_DAY in found
    assert found[DROP_DAY]["type"] == "drop"


def test_normal_days_are_not_flagged(anomaly_df):
    result = AnomalyDetector(anomaly_df).analyze(make_params())

    normal_days = set(anomalies_by_period(result)) - {SPIKE_DAY, DROP_DAY}

    for period in normal_days:
        anomaly = anomalies_by_period(result)[period]
        assert anomaly["value"] not in (200.0, 201.0, 202.0, 203.0, 204.0)


def test_spike_explanation_names_the_guilty_store(anomaly_df):
    result = AnomalyDetector(anomaly_df).analyze(make_params())

    explanation = anomalies_by_period(result)[SPIKE_DAY]["explanation"]
    top = explanation["top_contributors"][0]

    assert top["segment"] in ("Ljubljana", "Food")
    assert top["deviation"] > 0
    assert explanation["holiday_season"] == "No"


def test_unusual_segments_include_store_with_own_anomaly(anomaly_df):
    result = AnomalyDetector(anomaly_df).analyze(make_params())

    stores = {s["segment"] for s in result["unusual_segments"]["store_city"]}

    assert "Ljubljana" in stores
    assert "Maribor" in stores


def test_monthly_default_granularity_falls_back_to_daily(anomaly_df):

    result = AnomalyDetector(anomaly_df).analyze(make_params())
    assert result["granularity"] == "day"
    assert result["periods_analyzed"] == 120  # Jan 1 - Apr 30, 2023 is not a leap year


def test_series_included_for_charting(anomaly_df):
    result = AnomalyDetector(anomaly_df).analyze(make_params())
    assert len(result["series"]) == result["periods_analyzed"]
    assert set(result["series"][0]) == {"period", "value"}


def test_empty_filter_result_returns_note(anomaly_df):
    result = AnomalyDetector(anomaly_df).analyze(make_params(filters={"store_city": "Koper"}))
    assert result["anomaly_count"] == 0
    assert "note" in result


def test_result_is_json_serializable(anomaly_df):
    import json
    result = AnomalyDetector(anomaly_df).analyze(make_params())
    json.dumps(result)