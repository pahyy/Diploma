import logging

import pandas as pd

from agent.analyzers import common

logger = logging.getLogger(__name__)


class PromotionAnalyzer:

    # A period (before/during) must have at least this many days of data,
    # otherwise the comparison would be too noisy to trust.
    MIN_COMPARISON_DAYS = 7

    # Uplift thresholds for classifying a promotion's effect (in %).
    POSITIVE_THRESHOLD = 5.0
    NEGATIVE_THRESHOLD = -5.0

    # How many best/worst promotions to list by default.
    DEFAULT_TOP_N = 5

    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info("PromotionAnalyzer initialized with %d rows", len(self.df))

    def analyze(self, params: dict) -> dict:
        entity_filters = dict(params["entity_filters"])
        promo_type_filter = entity_filters.pop("promotion_type", None)

        base_df = common.apply_filters(self.df, params["filters"])
        base_df = common.apply_filters(base_df, entity_filters)
        dated_df = common.apply_date_filter(base_df, params["date_filter"])

        if len(dated_df) == 0:
            return self._empty_result(params, reason="No transactions match the given filters.")

        metric = self._pick_metric(params["metrics"])

        # a small "daily sales per product" table 
        daily = self._daily_sales_by_product(base_df, metric)
        data_start, data_end = daily.index.min(), daily.index.max()

        promotions = self._list_promotions(dated_df)
        if promo_type_filter:
            wanted = promo_type_filter if isinstance(promo_type_filter, list) else [promo_type_filter]
            promotions = promotions[promotions["promotion_type"].isin(wanted)]
            if len(promotions) == 0:
                return self._empty_result(params, reason="No promotions of the requested type found.")

        products_per_promo = base_df.groupby("promotion_id")["product_name"].unique()

        scored = []
        skipped = 0
        for promo in promotions.itertuples(index=False):
            products = list(products_per_promo.get(promo.promotion_id, []))
            score = self._score_promotion(promo, products, daily, data_start, data_end)
            if score is None:
                skipped += 1
            else:
                scored.append(score)

        if not scored:
            return self._empty_result(params, reason="No promotion had enough data for a before/during comparison.")

        top_n = params["limit"] or self.DEFAULT_TOP_N
        by_uplift = sorted(scored, key=lambda s: s["uplift_pct"])

        return {
            "query_type": params["metadata"]["query_type"],
            "metric": metric,
            "row_count": len(dated_df),
            "promotions_analyzed": len(scored),
            "promotions_skipped": skipped,
            "overall": self._summarize(scored),
            "by_promotion_type": self._summarize_per_type(scored),
            "worst_promotions": by_uplift[:top_n],
            "best_promotions": list(reversed(by_uplift[-top_n:])),
        }

    def _pick_metric(self, metrics: list) -> str:
        if "quantity" in metrics:
            return "quantity"
        return "total_sales"

    def _daily_sales_by_product(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        day = df["transaction_date"].dt.normalize()
        daily = df.groupby([day, "product_name"])[metric].sum().unstack(fill_value=0)
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        return daily.reindex(full_range, fill_value=0)

    def _list_promotions(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["promotion_id", "promotion_type", "promotion_start_date",
                "promotion_end_date", "promotion_effectiveness"]
        promos = df[cols].drop_duplicates("promotion_id").copy()
        promos["promotion_start_date"] = pd.to_datetime(promos["promotion_start_date"]).dt.normalize()
        promos["promotion_end_date"] = pd.to_datetime(promos["promotion_end_date"]).dt.normalize()
        return promos

    def _mean_daily(self, daily: pd.DataFrame, products: list, start, end):
        days = (end - start).days + 1
        if days < self.MIN_COMPARISON_DAYS:
            return None
        window = daily.loc[start:end, daily.columns.intersection(products)]
        return float(window.sum(axis=1).mean())

    def _score_promotion(self, promo, products, daily, data_start, data_end):

        if not products:
            return None

        start = max(promo.promotion_start_date, data_start)
        end = min(promo.promotion_end_date, data_end)
        if start > end:
            return None  # window entirely outside the dataset

        during = self._mean_daily(daily, products, start, end)
        if during is None:
            return None

        # "before" is the same number of days immediately before the window
        window_days = (end - start).days + 1
        before_end = start - pd.Timedelta(days=1)
        before_start = max(before_end - pd.Timedelta(days=window_days - 1), data_start)
        if before_end < data_start:
            return None
        before = self._mean_daily(daily, products, before_start, before_end)
        if before is None or before == 0:
            return None

        
        # "after" is the same number of days immediately before if it exists
        after_start = end + pd.Timedelta(days=1)
        if after_start <= data_end:
            after_end = min(after_start + pd.Timedelta(days=window_days - 1), data_end)
            after = self._mean_daily(daily, products, after_start, after_end)

        uplift_pct = (during - before) / before * 100
        long_term_pct = (after - before) / before * 100 if after is not None else None

        return {
            "promotion_id": int(promo.promotion_id),
            "promotion_type": promo.promotion_type,
            "window_start": str(start.date()),
            "window_end": str(end.date()),
            "avg_daily_before": round(before, 2),
            "avg_daily_during": round(during, 2),
            "avg_daily_after": round(after, 2) if after is not None else None,
            "uplift_pct": round(uplift_pct, 2),
            "long_term_pct": round(long_term_pct, 2) if long_term_pct is not None else None,
            "effect": self._classify(uplift_pct),
            "dataset_effectiveness_label": promo.promotion_effectiveness,
        }

    def _classify(self, uplift_pct: float) -> str:
        if uplift_pct >= self.POSITIVE_THRESHOLD:
            return "positive"
        if uplift_pct <= self.NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    def _summarize(self, scored: list) -> dict:

        uplifts = [s["uplift_pct"] for s in scored]
        long_terms = [s["long_term_pct"] for s in scored if s["long_term_pct"] is not None]
        return {
            "avg_uplift_pct": round(sum(uplifts) / len(uplifts), 2),
            "positive_count": sum(1 for s in scored if s["effect"] == "positive"),
            "neutral_count": sum(1 for s in scored if s["effect"] == "neutral"),
            "negative_count": sum(1 for s in scored if s["effect"] == "negative"),
            "avg_long_term_pct": round(sum(long_terms) / len(long_terms), 2) if long_terms else None,
            "long_term_measurable_count": len(long_terms),
        }

    def _summarize_per_type(self, scored: list) -> list:
        by_type = {}
        for s in scored:
            by_type.setdefault(s["promotion_type"], []).append(s)

        summaries = []
        for promo_type, group in sorted(by_type.items()):
            summary = self._summarize(group)
            summary["promotion_type"] = promo_type
            summary["promotions_analyzed"] = len(group)
            summaries.append(summary)

        summaries.sort(key=lambda s: s["avg_uplift_pct"], reverse=True)
        return summaries

    def _empty_result(self, params: dict, reason: str) -> dict:
        return {
            "query_type": params["metadata"]["query_type"],
            "row_count": 0,
            "promotions_analyzed": 0,
            "promotions_skipped": 0,
            "note": reason,
            "overall": None,
            "by_promotion_type": [],
            "worst_promotions": [],
            "best_promotions": [],
        }