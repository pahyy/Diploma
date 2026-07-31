import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

FILTER_KEY_RENAME = {
    "store_location_within_city": "store_location",
}

ENTITY_KEY_RENAME = {
    "products": "product_name",
    "categories": "product_category",
    "brands": "product_brand",
    "promotions": "promotion_type",
    "customers": "customer_id",
    "stores": "store_city",
}

# Granularity name -> pandas period code
PERIOD_CODES = {"day": "D", "week": "W", "month": "M", "year": "Y"}


def apply_date_filter(df: pd.DataFrame, date_filter: dict) -> pd.DataFrame:
    # Keep only rows whose transaction_date falls inside [start, end].
    start = date_filter.get("start")
    end = date_filter.get("end")
    if start:
        df = df[df["transaction_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["transaction_date"] <= pd.to_datetime(end)]
    return df


def bucket_by_period(df: pd.DataFrame, granularity: str):
    # Add a "_period" column (npr. "2021-03" za mesecno granulacijo).

    code = PERIOD_CODES.get(granularity, "M")
    df = df.copy()
    df["_period"] = df["transaction_date"].dt.to_period(code).astype(str)
    return df, "_period"


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    # Apply attribute filters ({column: value or [values]}) one by one.
    for key, value in filters.items():
        if key == "age_min":
            df = df[df["age"] >= value]
        elif key == "age_max":
            df = df[df["age"] <= value]
        elif key == "product_rating_min":
            try:
                df = df[df["product_rating"] >= value]
            except TypeError:
                logger.warning("product_rating_min filter value %r is not usable, skipping", value)
        elif key == "discount_applied":
            if value == "Yes":
                df = df[df["discount_applied"] > 0]
            elif value == "No":
                df = df[df["discount_applied"] == 0]
            else:
                logger.warning("Unexpected discount_applied filter value %r, skipping", value)
        elif key == "purchase_frequency_min":

            logger.warning("purchase_frequency_min has no numeric column to compare against, skipping")
            continue
        else:
            column = FILTER_KEY_RENAME.get(key, key)
            if column not in df.columns:
                logger.warning("Skipping unknown filter column '%s'", column)
                continue
            if isinstance(value, list):
                df = df[df[column].isin(value)]
            else:
                df = df[df[column] == value]
    return df


def apply_entity_filters(df: pd.DataFrame, entities: dict) -> pd.DataFrame:
    # Like apply_filters, but renames entity keys to CSV columns.
    renamed = {ENTITY_KEY_RENAME.get(k, k): v for k, v in entities.items()}
    return apply_filters(df, renamed)


def apply_all_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = apply_date_filter(df, params["date_filter"])
    df = apply_filters(df, params["filters"])
    df = apply_filters(df, params["entity_filters"])
    return df


def compute_metric(df: pd.DataFrame, metric: str):
    # Compute a single named metric over the given (already filtered) rows.
    if len(df) == 0:
        return None
    if metric == "total_sales":
        return round(float(df["total_sales"].sum()), 2)
    if metric == "quantity":
        return int(df["quantity"].sum())
    if metric == "unit_price":
        return round(float(df["unit_price"].mean()), 2)
    if metric == "discount_applied":
        return round(float(df["discount_applied"].mean()), 2)
    if metric == "product_rating":
        return round(float(df["product_rating"].mean()), 2)
    if metric == "total_transactions":
        return len(df)
    if metric == "promotion_effectiveness":
        return str(df["promotion_effectiveness"].mode()[0])
    logger.warning("Unknown metric '%s', skipping", metric)
    return None


def aggregate_metrics(df: pd.DataFrame, metrics: list) -> dict:
    return {metric: compute_metric(df, metric) for metric in metrics}


def fit_linear_trend(values: list) -> dict:
    # x is just the period index (0, 1, 2, ...), y is the metric value.
    
    # Returns None if there are fewer than 3 usable points (a line through
    # 2 points is always perfect, so a "trend" there is meaningless).
    pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pairs) < 3:
        return None

    x = np.array([[p[0]] for p in pairs], dtype=float)  # 2D, as sklearn expects
    y = np.array([p[1] for p in pairs], dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    slope = float(model.coef_[0])
    r_squared = float(model.score(x, y))
    mean_value = float(y.mean())

    # Slope expressed as % of the average value per period
    pct_change_per_period = (slope / mean_value * 100) if mean_value != 0 else 0.0

    if pct_change_per_period > 0.5:
        direction = "increasing"
    elif pct_change_per_period < -0.5:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "slope_per_period": round(slope, 2),
        "pct_change_per_period": round(pct_change_per_period, 2),
        "r_squared": round(r_squared, 3),
        "periods_used": len(pairs),
    }