import logging

import pandas as pd

from agent.analyzers import common

logger = logging.getLogger(__name__)


class AnomalyDetector:

    Z_THRESHOLD = 3.0     # how many std-devs away counts as an anomaly
    ROLLING_WINDOW = 30   # how many previous periods form the baseline
    MIN_PERIODS = 14      # need at least this much history before judging
    MAX_REPORTED = 20     # cap the anomaly list so the result stays readable
    TOP_CONTRIBUTORS = 3  # how many segments to name in an explanation

    def __init__(self, df: pd.DataFrame, z_threshold: float = None):
        self.df = df
        self.z_threshold = z_threshold if z_threshold is not None else self.Z_THRESHOLD
        logger.info("AnomalyDetector initialized with %d rows", len(self.df))

    def analyze(self, params: dict) -> dict:
        df = common.apply_all_filters(self.df, params)
        metric = "quantity" if "quantity" in params["metrics"] else "total_sales"
        granularity = self._pick_granularity(params["date_filter"]["granularity"])

        if len(df) == 0:
            return {
                "query_type": params["metadata"]["query_type"],
                "row_count": 0,
                "note": "No transactions match the given filters.",
                "anomalies": [],
                "anomaly_count": 0,
                "unusual_segments": {},
            }

        series = self._build_series(df, metric, granularity)
        anomalies = self._detect(series)

        # every anomaly also gets an explanation
        # (which store/category drove it, plus holiday/weekend context).
        for anomaly in anomalies:
            anomaly["explanation"] = self._explain(df, metric, granularity, anomaly["period"])

        # Keep only the strongest anomalies in the report
        anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)
        reported = anomalies[: self.MAX_REPORTED]
        reported.sort(key=lambda a: a["period"])

        return {
            "query_type": params["metadata"]["query_type"],
            "row_count": len(df),
            "metric": metric,
            "granularity": granularity,
            "z_threshold": self.z_threshold,
            "periods_analyzed": len(series),
            "anomaly_count": len(anomalies),
            "anomalies": reported,
            "series": [
                {"period": str(p), "value": round(float(v), 2)} for p, v in series.items()
            ],
            "unusual_segments": {
                "store_city": self._unusual_segments(df, metric, granularity, "store_city"),
                "product_category": self._unusual_segments(df, metric, granularity, "product_category"),
            },
        }

    def _pick_granularity(self, requested: str) -> str:
        # month nimamo ker imamo samo 24m podatkov
        if requested in ("day", "week"):
            return requested
        return "day"

    def _build_series(self, df: pd.DataFrame, metric: str, granularity: str) -> pd.Series:
        # sum the metric per period, filling periods with no sales with 0.
        code = common.PERIOD_CODES.get(granularity, "D")
        periods = df["transaction_date"].dt.to_period(code)
        series = df.groupby(periods)[metric].sum()
        full_range = pd.period_range(series.index.min(), series.index.max(), freq=code)
        return series.reindex(full_range, fill_value=0)

    def _detect(self, series: pd.Series) -> list:
        # shift(1) -> trolling window pokrije samo prejsnje
        baseline_mean = series.shift(1).rolling(self.ROLLING_WINDOW, min_periods=self.MIN_PERIODS).mean()
        baseline_std = series.shift(1).rolling(self.ROLLING_WINDOW, min_periods=self.MIN_PERIODS).std()

        anomalies = []
        for period, value in series.items():
            mean = baseline_mean[period]
            std = baseline_std[period]
            if pd.isna(mean) or pd.isna(std) or std == 0:
                continue

            z = (value - mean) / std
            if abs(z) >= self.z_threshold:
                anomalies.append({
                    "period": str(period),
                    "value": round(float(value), 2),
                    "expected": round(float(mean), 2),
                    "z_score": round(float(z), 2),
                    "deviation_pct": round((value - mean) / mean * 100, 2) if mean != 0 else None,
                    "type": "spike" if z > 0 else "drop",
                })
        return anomalies

    def _explain(self, df: pd.DataFrame, metric: str, granularity: str, period_str: str) -> dict:
        code = common.PERIOD_CODES.get(granularity, "D")
        period_col = df["transaction_date"].dt.to_period(code).astype(str)
        in_period = df[period_col == period_str]

        explanation = {"top_contributors": []}

        for segment_column in ("store_city", "product_category"):
            segment_totals = df.groupby([period_col, segment_column])[metric].sum()
            period_count = period_col.nunique()

            for segment, value in segment_totals.loc[period_str].items():
                usual = segment_totals.xs(segment, level=1).sum() / period_count
                explanation["top_contributors"].append({
                    "segment_type": segment_column,
                    "segment": str(segment),
                    "value": round(float(value), 2),
                    "usual": round(float(usual), 2),
                    "deviation": round(float(value - usual), 2),
                })

        explanation["top_contributors"].sort(key=lambda c: abs(c["deviation"]), reverse=True)
        explanation["top_contributors"] = explanation["top_contributors"][: self.TOP_CONTRIBUTORS]

        # context flags, was this a holiday period/weekend
        if len(in_period) > 0:
            explanation["holiday_season"] = str(in_period["holiday_season"].mode()[0])
            explanation["weekend"] = str(in_period["weekend"].mode()[0])
            explanation["dominant_promotion"] = str(in_period["promotion_type"].mode()[0])

        return explanation

    def _unusual_segments(self, df: pd.DataFrame, metric: str, granularity: str, segment_column: str) -> list:
        if segment_column not in df.columns:
            return []

        results = []
        for segment, segment_df in df.groupby(segment_column):
            series = self._build_series(segment_df, metric, granularity)
            anomalies = self._detect(series)
            if anomalies:
                results.append({
                    "segment": str(segment),
                    "anomaly_count": len(anomalies),
                    "max_abs_z": round(max(abs(a["z_score"]) for a in anomalies), 2),
                    "worst_period": max(anomalies, key=lambda a: abs(a["z_score"]))["period"],
                })

        results.sort(key=lambda r: r["anomaly_count"], reverse=True)
        return results