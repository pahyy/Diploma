"""
Synthetic DataFrame za testiranje

  Product A (promo 1, "20% Off", Feb 2023):
      prodaja 100 na dan v januaru, 200 na dan v februaru, 100 na dan od marca naprej

  Product B (promo 2, "Flash Sale", March 2023):
      prodaja 100 na dan vedno razen v marcu, 80 na dan
"""

import pandas as pd
import pytest

from agent.analyzers.promotion_analyzer import PromotionAnalyzer

pytestmark = pytest.mark.promotion_analyzer


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
        "metadata": {"query_type": "promotion_analysis", "intent": "summarize"},
    }
    params.update(overrides)
    return params


@pytest.fixture
def promo_df():
    rows = []
    days = pd.date_range("2023-01-01", "2023-04-30", freq="D")

    for day in days:
        # Product A
        a_sales = 200.0 if day.month == 2 else 100.0
        rows.append({
            "transaction_date": day, "product_name": "A", "total_sales": a_sales,
            "quantity": 1, "promotion_id": 1, "promotion_type": "20% Off",
            "promotion_start_date": "2023-02-01", "promotion_end_date": "2023-02-28",
            "promotion_effectiveness": "High", "store_city": "Ljubljana",
        })
        # Product B
        b_sales = 80.0 if day.month == 3 else 100.0
        rows.append({
            "transaction_date": day, "product_name": "B", "total_sales": b_sales,
            "quantity": 1, "promotion_id": 2, "promotion_type": "Flash Sale",
            "promotion_start_date": "2023-03-01", "promotion_end_date": "2023-03-31",
            "promotion_effectiveness": "Low", "store_city": "Maribor",
        })

    return pd.DataFrame(rows)


def scores_by_id(result):
    all_scored = result["worst_promotions"] + result["best_promotions"]
    return {s["promotion_id"]: s for s in all_scored}


def test_uplift_computed_from_before_and_during_daily_averages(promo_df):
    result = PromotionAnalyzer(promo_df).analyze(make_params())

    assert result["promotions_analyzed"] == 2
    promo_1 = scores_by_id(result)[1]
    assert promo_1["avg_daily_before"] == 100.0
    assert promo_1["avg_daily_during"] == 200.0
    assert promo_1["uplift_pct"] == 100.0
    assert promo_1["effect"] == "positive"


def test_negative_promotion_is_classified_negative(promo_df):
    result = PromotionAnalyzer(promo_df).analyze(make_params())

    promo_2 = scores_by_id(result)[2]
    assert promo_2["uplift_pct"] == -20.0
    assert promo_2["effect"] == "negative"
    assert result["worst_promotions"][0]["promotion_id"] == 2


def test_long_term_effect_measures_after_vs_before(promo_df):
    result = PromotionAnalyzer(promo_df).analyze(make_params())

    promo_1 = scores_by_id(result)[1]

    assert promo_1["avg_daily_after"] == 100.0
    assert promo_1["long_term_pct"] == 0.0


def test_per_type_summary_sorted_by_uplift(promo_df):
    result = PromotionAnalyzer(promo_df).analyze(make_params())

    types = [t["promotion_type"] for t in result["by_promotion_type"]]
    assert types == ["20% Off", "Flash Sale"]
    best = result["by_promotion_type"][0]
    assert best["avg_uplift_pct"] == 100.0
    assert best["positive_count"] == 1


def test_entity_filter_restricts_to_one_promotion_type(promo_df):
    params = make_params(entity_filters={"promotion_type": ["Flash Sale"]})
    result = PromotionAnalyzer(promo_df).analyze(params)

    assert result["promotions_analyzed"] == 1
    assert result["worst_promotions"][0]["promotion_type"] == "Flash Sale"


def test_type_filter_selects_promotions_but_keeps_full_baseline(promo_df):
    rows = []
    for day in pd.date_range("2023-01-01", "2023-04-30", freq="D"):
        in_promo = day.month == 2
        rows.append({
            "transaction_date": day, "product_name": "C",
            "total_sales": 150.0 if in_promo else 100.0, "quantity": 1,
            "promotion_id": 3 if in_promo else 4,
            "promotion_type": "20% Off" if in_promo else "Flash Sale",
            "promotion_start_date": "2023-02-01", "promotion_end_date": "2023-02-28",
            "promotion_effectiveness": "Medium", "store_city": "Ljubljana",
        })
    df = pd.DataFrame(rows)

    params = make_params(entity_filters={"promotion_type": ["20% Off"]})
    result = PromotionAnalyzer(df).analyze(params)

    assert result["promotions_analyzed"] == 1
    promo_3 = scores_by_id(result)[3]
    assert promo_3["avg_daily_before"] == 100.0
    assert promo_3["avg_daily_during"] == 150.0
    assert promo_3["uplift_pct"] == 50.0


def test_no_matching_rows_returns_empty_result_with_note(promo_df):
    params = make_params(filters={"store_city": "Koper"})
    result = PromotionAnalyzer(promo_df).analyze(params)

    assert result["promotions_analyzed"] == 0
    assert result["row_count"] == 0
    assert "note" in result


def test_window_outside_dataset_is_skipped(promo_df):

    extra = promo_df.iloc[:1].copy()
    extra["promotion_id"] = 99
    extra["promotion_start_date"] = "2024-01-01"
    extra["promotion_end_date"] = "2024-06-30"
    df = pd.concat([promo_df, extra], ignore_index=True)

    result = PromotionAnalyzer(df).analyze(make_params())

    assert result["promotions_analyzed"] == 2
    assert result["promotions_skipped"] == 1


def test_result_is_json_serializable(promo_df):
    import json
    result = PromotionAnalyzer(promo_df).analyze(make_params())
    json.dumps(result)