"""

    trend records ("_period" rows)     -> line chart
    comparison groups (non-trend)      -> bar chart
    comparison groups (trend)          -> multi-line chart
    grouped records (top-N etc.)       -> horizontal bar chart
    promotion type summaries           -> uplift bar chart (+/- around zero)
    anomaly series                     -> line chart with marked anomalies

"""

import hashlib
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _readable_number(value, _pos=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f} mio".replace(".", ",")
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f} tis."
    return f"{value:g}"

logger = logging.getLogger(__name__)

SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
                 "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

POSITIVE_COLOR = "#2a78d6"
NEGATIVE_COLOR = "#e34948"
GRID_COLOR = "#dddddd"
TEXT_COLOR = "#333333"

METRIC_LABELS = {
    "total_sales": "Prodaja (€)",
    "quantity": "Prodana količina",
    "unit_price": "Povprečna cena (€)",
    "product_rating": "Povprečna ocena",
    "total_transactions": "Število transakcij",
    "discount_applied": "Povprečen popust",
}


class Visualizer:

    MAX_BARS = 8
    MAX_LINES = 4

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_chart(self, aggregated: dict, params: dict):

        analysis = aggregated["analysis"]
        metric = self._main_metric(params)

        try:
            if analysis.get("query_type") == "anomaly_detection" or "anomalies" in analysis:
                return self._anomaly_chart(analysis, metric)
            if analysis.get("query_type") == "promotion_analysis" or "by_promotion_type" in analysis:
                return self._promotion_chart(analysis)
            if "comparison_dimension" in analysis:
                return self._comparison_chart(analysis, metric, params)
            return self._sales_chart(analysis, metric, aggregated)
        except Exception as e:
            logger.error("Chart creation failed: %s", e)
            return None


    def _sales_chart(self, analysis: dict, metric: str, aggregated: dict):
        records = analysis.get("results", [])
        if len(records) < 2:
            return None

        if "_period" in records[0]:
            periods = [r["_period"] for r in records]
            values = [r.get(metric) for r in records]
            fig, ax = self._new_figure()
            ax.plot(periods, values, color=SERIES_COLORS[0], linewidth=2)

            # mark anomalies
            anomalies = aggregated.get("supporting", {}).get("anomalies", {}).get("anomalies", [])
            marked = [(a["period"], a["value"]) for a in anomalies if a["period"] in periods]
            if marked:
                ax.scatter([m[0] for m in marked], [m[1] for m in marked],
                           color=NEGATIVE_COLOR, zorder=3, s=45, label="Anomalija")
                ax.legend()

            self._style_axes(ax, x_label="Obdobje", y_label=METRIC_LABELS.get(metric, metric),
                             title="Gibanje prodaje po obdobjih")
            self._thin_x_ticks(ax, periods)
            return self._save(fig, {"kind": "trend", "records": records})

        label_key = next((k for k in records[0] if k != metric and not k.startswith("_")), None)
        if label_key is None:
            return None
        records = [r for r in records if r.get(metric) is not None][: self.MAX_BARS]
        labels = [str(r[label_key]) for r in records]
        values = [r[metric] for r in records]

        fig, ax = self._new_figure()
        bars = ax.barh(labels[::-1], values[::-1], color=SERIES_COLORS[0], height=0.6)
        ax.bar_label(bars, fmt="%.0f", padding=4, color=TEXT_COLOR, fontsize=9)

        self._style_axes(ax, x_label=METRIC_LABELS.get(metric, metric), y_label="",
                         title=f"Primerjava po: {label_key}", value_axis="x")
        return self._save(fig, {"kind": "grouped", "records": records})

    def _comparison_chart(self, analysis: dict, metric: str, params: dict):
        groups = analysis.get("results", [])
        if not groups:
            return None

        if groups and "periods" in groups[0]:
            fig, ax = self._new_figure()
            for i, group in enumerate(groups[: self.MAX_LINES]):
                periods = [p["_period"] for p in group["periods"]]
                values = [p.get(metric) for p in group["periods"]]
                ax.plot(periods, values, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                        linewidth=2, label=group["label"])
            ax.legend()
            self._style_axes(ax, x_label="Obdobje", y_label=METRIC_LABELS.get(metric, metric),
                             title="Primerjava gibanja prodaje")
            all_periods = [p["_period"] for p in groups[0]["periods"]]
            self._thin_x_ticks(ax, all_periods)
            return self._save(fig, {"kind": "comparison_trend", "groups": groups})

        labels = [str(g["label"]) for g in groups]
        values = [g.get(metric) or 0 for g in groups]
        fig, ax = self._new_figure()
        bars = ax.bar(labels, values, color=SERIES_COLORS[0], width=0.5)
        ax.bar_label(bars, fmt="%.0f", padding=4, color=TEXT_COLOR, fontsize=9)
        self._style_axes(ax, x_label="", y_label=METRIC_LABELS.get(metric, metric),
                         title="Primerjava skupin")
        return self._save(fig, {"kind": "comparison", "groups": groups})

    def _promotion_chart(self, analysis: dict):
        summaries = analysis.get("by_promotion_type", [])
        if not summaries:
            return None

        labels = [s["promotion_type"] for s in summaries]
        values = [s["avg_uplift_pct"] for s in summaries]
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

        fig, ax = self._new_figure()
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.bar_label(bars, fmt="%+.1f %%", padding=4, color=TEXT_COLOR, fontsize=9)
        ax.axhline(0, color=TEXT_COLOR, linewidth=1)
        self._style_axes(ax, x_label="", y_label="Povprečen dvig prodaje (%)",
                         title="Učinkovitost promocij po tipu")
        return self._save(fig, {"kind": "promotion", "summaries": summaries})

    def _anomaly_chart(self, analysis: dict, metric: str):
        series = analysis.get("series", [])
        if len(series) < 2:
            return None

        periods = [p["period"] for p in series]
        values = [p["value"] for p in series]

        fig, ax = self._new_figure()
        ax.plot(periods, values, color=SERIES_COLORS[0], linewidth=1.5, label="Prodaja")

        anomalies = analysis.get("anomalies", [])
        if anomalies:
            ax.scatter([a["period"] for a in anomalies], [a["value"] for a in anomalies],
                       color=NEGATIVE_COLOR, zorder=3, s=45, label="Anomalija")
        ax.legend()
        self._style_axes(ax, x_label="Obdobje", y_label=METRIC_LABELS.get(metric, metric),
                         title="Zaznane anomalije v prodaji")
        self._thin_x_ticks(ax, periods)
        return self._save(fig, {"kind": "anomaly", "series_len": len(series),
                                "anomalies": anomalies})

    def _main_metric(self, params: dict) -> str:
        for metric in params["metrics"]:
            if metric in METRIC_LABELS:
                return metric
        return "total_sales"

    def _new_figure(self):
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
        fig.patch.set_facecolor("white")
        return fig, ax

    def _style_axes(self, ax, x_label: str, y_label: str, title: str, value_axis: str = "y"):
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=12)
        ax.set_xlabel(x_label, color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel(y_label, color=TEXT_COLOR, fontsize=10)
        ax.grid(True, color=GRID_COLOR, linewidth=0.7)
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        if value_axis == "x":
            ax.xaxis.set_major_formatter(FuncFormatter(_readable_number))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(_readable_number))

    def _thin_x_ticks(self, ax, periods: list, max_ticks: int = 12):
        if len(periods) > max_ticks:
            step = len(periods) // max_ticks + 1
            ax.set_xticks(range(0, len(periods), step))
            ax.tick_params(axis="x", rotation=45)
        elif len(periods) > 6:
            ax.tick_params(axis="x", rotation=45)

    def _save(self, fig, payload: dict) -> str:
        digest = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
        path = os.path.join(self.output_dir, f"chart_{digest}.png")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Chart saved to %s", path)
        return path