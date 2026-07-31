"""
Results aggregator.

Iz parametara odlucuje koji analizator se pokrece, spoji rezultate u jedan dict za ostale dijelove

Routing table:

    query_type            -> analyzer
    ------------------------------------------
    sales_trend           -> SalesAnalyzer
    comparison            -> SalesAnalyzer
    product_analysis      -> SalesAnalyzer
    customer_analysis     -> SalesAnalyzer
    promotion_analysis    -> PromotionAnalyzer
    anomaly_detection     -> AnomalyDetector
    explain               -> SalesAnalyzer + AnomalyDetector
"""

import logging

logger = logging.getLogger(__name__)

# query types answered by the plain sales analyzer
SALES_QUERY_TYPES = ("sales_trend", "comparison", "product_analysis", "customer_analysis")


class ResultsAggregator:

    def __init__(self, sales_analyzer, promotion_analyzer, anomaly_detector):
        self.sales_analyzer = sales_analyzer
        self.promotion_analyzer = promotion_analyzer
        self.anomaly_detector = anomaly_detector

    def run(self, params: dict) -> dict:
        
        #    "analyzers_used": list of analyzer names that ran
        #    "analysis":       the main analyzer's result
        #    "supporting":     optional extra results (anomalies for explain)
        
        query_type = params["metadata"]["query_type"]
        logger.info("Routing query_type=%s", query_type)

        if query_type == "promotion_analysis":
            return {
                "analyzers_used": ["promotion"],
                "analysis": self.promotion_analyzer.analyze(params),
                "supporting": {},
            }

        if query_type == "anomaly_detection":
            return {
                "analyzers_used": ["anomaly"],
                "analysis": self.anomaly_detector.analyze(params),
                "supporting": {},
            }

        if query_type == "explain":
            # "why did X happen?" give the summary LLM both the sales
            # numbers (trend view) and any anomalies in the same period.
            explain_params = self._with_trend_intent(params)
            return {
                "analyzers_used": ["sales", "anomaly"],
                "analysis": self.sales_analyzer.analyze(explain_params),
                "supporting": {"anomalies": self.anomaly_detector.analyze(params)},
            }

        if query_type not in SALES_QUERY_TYPES:
            logger.warning("Unknown query_type '%s' - falling back to SalesAnalyzer", query_type)

        return {
            "analyzers_used": ["sales"],
            "analysis": self.sales_analyzer.analyze(params),
            "supporting": {},
        }

    def _with_trend_intent(self, params: dict) -> dict:
        new_params = dict(params)
        new_params["metadata"] = dict(params["metadata"], intent="trend")
        return new_params