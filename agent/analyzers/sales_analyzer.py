import logging

import pandas as pd

from agent.analyzers import common

logger = logging.getLogger(__name__)


class SalesAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info("SalesAnalyzer initialized with %d rows", len(self.df))

    def analyze(self, params: dict) -> dict:
        # parametri iz parameter extractora
        df = common.apply_all_filters(self.df, params)

        if params["comparison"]["enabled"]:
            return self._run_comparison(df, params)

        group_columns = list(params["group_by"])
        is_trend = params["metadata"]["intent"] == "trend"
        if is_trend:
            df, period_column = common.bucket_by_period(df, params["date_filter"]["granularity"])
            group_columns = [period_column] + group_columns

        if group_columns:
            records = self._aggregate_grouped(df, group_columns, params["metrics"])
        else:
            records = [common.aggregate_metrics(df, params["metrics"])]

        # Za trende fituje lin regresiju za period, da mozemo dobit growing/falling/stable

        trend = None
        if is_trend and not params["group_by"]:
            trend = self._fit_trends(records, params["metrics"])

        records = self._sort_and_limit(records, params["sort_by"], params["sort_order"], params["limit"])

        result = {
            "query_type": params["metadata"]["query_type"],
            "row_count": len(df),
            "results": records,
        }

        if trend:
            result["trend"] = trend

        return result

    def _fit_trends(self, period_records: list, metrics: list) -> dict:
        trends = {}
        for metric in metrics:
            values = [r.get(metric) for r in period_records]

            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(numeric) != len([v for v in values if v is not None]):
                continue
            fitted = common.fit_linear_trend(values)
            if fitted:
                trends[metric] = fitted
        return trends

    def _aggregate_grouped(self, df: pd.DataFrame, group_columns: list, metrics: list) -> list:
        group_columns = [c for c in group_columns if c in df.columns]
        if not group_columns:
            return [common.aggregate_metrics(df, metrics)]

        records = []
        for group_keys, group_df in df.groupby(group_columns, dropna=False):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            row = dict(zip(group_columns, group_keys))
            row.update(common.aggregate_metrics(group_df, metrics))
            records.append(row)

        if "_period" in group_columns:
            records.sort(key=lambda r: r["_period"])

        return records

    def _run_comparison(self, df: pd.DataFrame, params: dict) -> dict:
        is_trend = params["metadata"]["intent"] == "trend"

        results = []
        for group in params["comparison"]["groups"]:
            group_df = common.apply_filters(df, group["filters"])
            group_df = common.apply_entity_filters(group_df, group["entities"])

            if is_trend:
                bucketed_df, period_column = common.bucket_by_period(group_df, params["date_filter"]["granularity"])
                periods = self._aggregate_grouped(bucketed_df, [period_column], params["metrics"])
                group_result = {"label": group["label"], "row_count": len(group_df), "periods": periods}
                trend = self._fit_trends(periods, params["metrics"])
                if trend:
                    group_result["trend"] = trend
                results.append(group_result)
            else:
                metrics = common.aggregate_metrics(group_df, params["metrics"])
                results.append({"label": group["label"], "row_count": len(group_df), **metrics})

        return {
            "query_type": params["metadata"]["query_type"],
            "comparison_dimension": params["comparison"]["dimension"],
            "row_count": len(df),
            "results": results,
        }

    def _sort_and_limit(self, records: list, sort_by, sort_order: str, limit) -> list:
        if sort_by and records and sort_by in records[0]:
            reverse = sort_order == "desc"
            # (value is None, value) pushes missing values to the end either way
            records = sorted(records, key=lambda r: (r[sort_by] is None, r[sort_by]), reverse=reverse)
        if limit:
            records = records[:limit]
        return records