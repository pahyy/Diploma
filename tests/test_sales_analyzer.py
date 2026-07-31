import json

import pandas as pd
import pytest

from agent.analyzers.sales_analyzer import SalesAnalyzer

pytestmark = pytest.mark.sales_analyzer


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
        "metadata": {"query_type": "sales_trend", "intent": "summarize"},
    }
    params.update(overrides)
    return params


@pytest.fixture
def sample_df():
    rows = [
        # tx, cust,    date,      category,   brand,        city,        location,  discount,    freq, rating, age, sales, tot_tx, qty, price, promo_eff, promo_type
        (1, 1, "2023-01-05", "Electronics", "Apple", "Ljubljana", "Ljubljana BTC",      0.10, "Daily", 4.5,    30,  100.0,     50,   1, 100.0,    "High", "20% Off"),
        (2, 1, "2023-01-20", "Electronics", "Apple", "Ljubljana", "Ljubljana BTC", 0.20, "Daily", 4.0, 30, 200.0, 50, 2, 100.0, "High", "Flash Sale"),
        (3, 2, "2023-02-10", "Electronics", "Samsung", "Maribor", "Maribor Center", 0.00, "Weekly", 4.2, 45, 300.0, 10, 1, 300.0, "Medium", "20% Off"),
        (4, 3, "2024-01-15", "Food", "Milka", "Ljubljana", "Ljubljana Center", 0.05, "Monthly", 3.8, 25, 50.0, 5, 5, 10.0, "Low", "Buy One Get One Free"),
        (5, 4, "2024-01-20", "Food", "Barilla", "Maribor", "Maribor Center", 0.15, "Weekly", 4.9, 60, 75.0, 20, 3, 25.0, "High", "Flash Sale"),
        (6, 5, "2024-06-01", "Electronics", "Samsung", "Maribor", "Maribor Center", 0.30, "Daily", 4.6, 22, 150.0, 8, 1, 150.0, "High", "20% Off"),
        (7, 6, "2023-01-25", "Electronics", "Apple", "Maribor", "Maribor Center", 0.00, "Weekly", 4.1, 51, 120.0, 12, 1, 120.0, "Medium", "Flash Sale"),
        (8, 7, "2024-02-05", "Food", "Milka", "Ljubljana", "Ljubljana BTC", 0.10, "Monthly", 3.5, 19, 60.0, 6, 4, 15.0, "Low", "20% Off"),
        (9, 8, "2023-02-14", "Electronics", "Samsung", "Ljubljana", "Ljubljana Center", 0.25, "Daily", 4.8, 33, 220.0, 15, 2, 110.0, "High", "Flash Sale"),
        (10, 9, "2024-01-10", "Food", "Barilla", "Maribor", "Maribor Center", 0.05, "Weekly", 4.0, 40, 90.0, 9, 3, 30.0, "Medium", "Buy One Get One Free"),
        (11, 10, "2023-03-01", "Electronics", "Apple", "Ljubljana", "Ljubljana BTC", 0.00, "Monthly", 4.3, 28, 130.0, 7, 1, 130.0, "High", "20% Off"),
        (12, 11, "2024-07-15", "Food", "Milka", "Maribor", "Maribor Center", 0.20, "Daily", 3.9, 35, 45.0, 4, 3, 15.0, "Low", "Flash Sale"),
    ]
    columns = [
        "transaction_id", "customer_id", "transaction_date", "product_category", "product_brand",
        "store_city", "store_location", "discount_applied", "purchase_frequency", "product_rating",
        "age", "total_sales", "total_transactions", "quantity", "unit_price",
        "promotion_effectiveness", "promotion_type",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def test_store_location_within_city_filter_maps_to_real_column(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(filters={"store_location_within_city": "Ljubljana BTC"})

    result = analyzer.analyze(params)

    assert result["row_count"] == 4
    assert result["results"][0]["total_sales"] == 490.0


def test_trend_bucketing_by_month_is_chronological(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(
        date_filter={"column": "transaction_date", "start": "2023-01-01", "end": "2023-03-31", "granularity": "month"},
        metadata={"query_type": "sales_trend", "intent": "trend"},
    )

    result = analyzer.analyze(params)

    periods = [row["_period"] for row in result["results"]]
    assert periods == ["2023-01", "2023-02", "2023-03"]
    sales_by_period = {row["_period"]: row["total_sales"] for row in result["results"]}
    assert sales_by_period == {"2023-01": 420.0, "2023-02": 520.0, "2023-03": 130.0}


def test_trend_is_fitted_on_chronological_order_not_on_sorted_results(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(
        date_filter={"column": "transaction_date", "start": "2023-01-01", "end": "2023-03-31", "granularity": "month"},
        metadata={"query_type": "sales_trend", "intent": "trend"},
        sort_by="total_sales",
        sort_order="desc",
    )

    result = analyzer.analyze(params)

    assert [row["total_sales"] for row in result["results"]] == [520.0, 420.0, 130.0]
    assert result["trend"]["total_sales"]["slope_per_period"] == -145.0


def test_trend_is_omitted_when_a_second_dimension_is_grouped_with_the_period(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(
        metadata={"query_type": "sales_trend", "intent": "trend"},
        group_by=["product_category"],
    )

    result = analyzer.analyze(params)

    assert all("_period" in row and "product_category" in row for row in result["results"])
    assert "trend" not in result


def test_group_by_category_matches_manual_sums(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(metrics=["total_sales", "quantity"], group_by=["product_category"])

    result = analyzer.analyze(params)

    by_category = {row["product_category"]: row for row in result["results"]}
    assert by_category["Electronics"]["total_sales"] == 1220.0
    assert by_category["Electronics"]["quantity"] == 9
    assert by_category["Food"]["total_sales"] == 320.0
    assert by_category["Food"]["quantity"] == 18


def test_comparison_mode_two_groups_including_renamed_brand_entity(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(
        comparison={
            "enabled": True,
            "dimension": "custom",
            "groups": [
                {"label": "Ljubljana", "filters": {"store_city": "Ljubljana"}, "entities": {}},
                {"label": "Samsung brand", "filters": {}, "entities": {"brands": ["Samsung"]}},
            ],
        },
    )

    result = analyzer.analyze(params)

    by_label = {group["label"]: group for group in result["results"]}
    assert by_label["Ljubljana"]["row_count"] == 6
    assert by_label["Ljubljana"]["total_sales"] == 760.0
    assert by_label["Samsung brand"]["row_count"] == 3
    assert by_label["Samsung brand"]["total_sales"] == 670.0


def test_comparison_mode_with_trend_intent_buckets_each_group_by_period(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(
        date_filter={"column": "transaction_date", "start": "2023-01-01", "end": "2023-03-31", "granularity": "month"},
        metadata={"query_type": "comparison", "intent": "trend"},
        comparison={
            "enabled": True,
            "dimension": "store_city",
            "groups": [
                {"label": "Ljubljana", "filters": {"store_city": "Ljubljana"}, "entities": {}},
                {"label": "Maribor", "filters": {"store_city": "Maribor"}, "entities": {}},
            ],
        },
    )

    result = analyzer.analyze(params)

    by_label = {group["label"]: group for group in result["results"]}
    assert by_label["Ljubljana"]["row_count"] == 4
    ljubljana_periods = {p["_period"]: p["total_sales"] for p in by_label["Ljubljana"]["periods"]}
    assert ljubljana_periods == {"2023-01": 300.0, "2023-02": 220.0, "2023-03": 130.0}

    assert by_label["Maribor"]["row_count"] == 2
    maribor_periods = {p["_period"]: p["total_sales"] for p in by_label["Maribor"]["periods"]}
    assert maribor_periods == {"2023-01": 120.0, "2023-02": 300.0}


def test_sort_and_limit(sample_df):
    analyzer = SalesAnalyzer(sample_df)

    desc_top_1 = analyzer.analyze(make_params(
        group_by=["product_category"], sort_by="total_sales", sort_order="desc", limit=1,
    ))
    assert len(desc_top_1["results"]) == 1
    assert desc_top_1["results"][0]["product_category"] == "Electronics"

    asc = analyzer.analyze(make_params(
        group_by=["product_category"], sort_by="total_sales", sort_order="asc",
    ))
    assert [row["product_category"] for row in asc["results"]] == ["Food", "Electronics"]

    unaffected = analyzer.analyze(make_params(
        group_by=["product_category"], sort_by="not_a_real_field", sort_order="desc",
    ))
    assert len(unaffected["results"]) == 2


def test_total_transactions_metric_is_row_count_not_sum(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(metrics=["total_transactions"], entity_filters={"customer_id": [1]})

    result = analyzer.analyze(params)

    assert result["row_count"] == 2
    assert result["results"][0]["total_transactions"] == 2


def test_discount_applied_filter_thresholds_the_rate_column(sample_df):

    analyzer = SalesAnalyzer(sample_df)

    yes_result = analyzer.analyze(make_params(filters={"discount_applied": "Yes"}))
    assert yes_result["row_count"] == 9

    no_result = analyzer.analyze(make_params(filters={"discount_applied": "No"}))
    assert no_result["row_count"] == 3


def test_purchase_frequency_min_filter_is_safe_no_op(sample_df):
    analyzer = SalesAnalyzer(sample_df)

    frequency_result = analyzer.analyze(make_params(filters={"purchase_frequency_min": 5}))
    assert frequency_result["row_count"] == 12


def test_analyze_result_is_json_serializable(sample_df):
    analyzer = SalesAnalyzer(sample_df)
    params = make_params(
        metrics=["total_sales", "quantity", "product_rating", "promotion_effectiveness", "total_transactions"],
        group_by=["product_category"],
    )

    result = analyzer.analyze(params)

    json.dumps(result)
