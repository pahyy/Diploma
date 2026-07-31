from datetime import datetime
import json
import os
import pandas as pd
from agent.query_processing.query_parser import RetailQueryParser
from agent.query_processing.parameter_extractor import ParameterExtractor, ValidationError, ParameterExtractionError
from agent.analyzers.sales_analyzer import SalesAnalyzer
from agent.analyzers.promotion_analyzer import PromotionAnalyzer
from agent.analyzers.anomaly_detector import AnomalyDetector
from agent.aggregation.results_aggregator import ResultsAggregator
from agent.summary.summary_generator import SummaryGenerator
from agent.validation.response_validator import ResponseValidator
from agent.visualization.visualizer import Visualizer
from agent.response.response_formatter import ResponseFormatter


class RetailAnalyticsHub:
    """
    question -> RetailQueryParser (LLM) -> ParameterExtractor
    -> ResultsAggregator (SalesAnalyzer / PromotionAnalyzer / AnomalyDetector)
    -> Visualizer (chart) + SummaryGenerator (LLM) -> ResponseValidator
    -> ResponseFormatter -> final response dict
    """

    def __init__(self):
        print("Inicijalizacija..")
        self.extractor = ParameterExtractor()

        print("Loading dataset...")
        self.retail_df = pd.read_csv("data/retail_data_v2.csv", low_memory=False)
        self.retail_df["transaction_date"] = pd.to_datetime(self.retail_df["transaction_date"])
        print("Dataset loaded.")

        # Parser dobi prvi in zadnji datum v podatkih, da kasneje lahko relativne
        # case omeji na samo to obdobje
        self.parser = RetailQueryParser(
            data_start=str(self.retail_df["transaction_date"].min().date()),
            data_end=str(self.retail_df["transaction_date"].max().date()),
        )

        self.aggregator = ResultsAggregator(
            sales_analyzer=SalesAnalyzer(self.retail_df),
            promotion_analyzer=PromotionAnalyzer(self.retail_df),
            anomaly_detector=AnomalyDetector(self.retail_df),
        )
        self.summary_generator = SummaryGenerator()
        self.validator = ResponseValidator()
        self.visualizer = Visualizer()
        self.formatter = ResponseFormatter()

    # logiranje za debugging
    # dodaje u jedan veliki file i jedan file gdje ima samo zadnji run
    def log_to_file(self, query, parser_out, extractor_out, error=None):
        latest_file = "./log.txt"
        history_file = "./all_logs.txt"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = (
            f"\n{'='*70}\n"
            f"Run time: {timestamp}\n"
            f"{'='*70}\n"
            f'Query: "{query}"\n\n'
            f"Query parser output:\n"
            f"{json.dumps(parser_out, indent=4, ensure_ascii=False)}\n;\n\n"
            f"Parameter extractor output:\n"
            f"{json.dumps(extractor_out, indent=4, ensure_ascii=False)}\n;\n"
            f"\n{'-'*70}\n"
        )

        if error is not None:
            log_entry += (
                f"ERROR:\n"
                f"{error}\n\n"
            )

        log_entry += f"\n{'-'*70}\n"

        # Overwrite latest run
        with open(latest_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # Append to history
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def process_query(self, user_input: str):
        """
        Celi pipeline za vprasanje.

        Vrne formatiran response dict ali None ce kaj faila.
        """
        print(f"\n--- Processing: {user_input} ---")

        # 1. Natural language v json (LLM call #1)
        parsed_json_dict = self.parser.parse(user_input)
        if "error" in parsed_json_dict:
            print(f"Parsing Error: {parsed_json_dict['error']}")
            return None

        # 2. Extrakcija i validacija parametrov
        try:
            final_params = self.extractor.extract(parsed_json_dict)
        except (ValidationError, ParameterExtractionError) as e:
            print(f"Validation/Extraction Error: {e}")
            self.log_to_file(user_input, parsed_json_dict, None, error=str(e))
            return None

        self.log_to_file(user_input, parsed_json_dict, final_params)

        # 3. Parametri grejo naprej v analizo
        try:
            aggregated = self.aggregator.run(final_params)
        except Exception as e:
            print(f"Analysis Error: {e}")
            self.parser.last_result_entities = None  # nothing valid to remember
            self.log_to_file(user_input, parsed_json_dict, final_params, error=str(e))
            return None

        # Zapomni si entitete prejsnjega rezultata v primeru vprasanj tipa
        # "Kateri so najbolje prodani produkti" -> "Za vsaki produkt od teh napisi katera trgovina je najvec prodala tega produkta"
        self.parser.remember_result_entities(final_params.get("group_by"), aggregated["analysis"])

        # 4. Chart, ce je smiselno da ima
        chart_path = self.visualizer.create_chart(aggregated, final_params)

        # 5. Sumarizacija podatkov (LLM call #2) in validacija podatkov
        summary = self.summary_generator.generate(user_input, final_params, aggregated)
        validation = self.validator.validate(summary, aggregated["analysis"])

        # 6. Pack everything into the final response
        # 6. Formatiranje vsega v odgovor
        return self.formatter.format(user_input, final_params, aggregated,
                                     summary, chart_path, validation)

    # testni user mode
    def run_user_mode(self):
        print("\n[USER INPUT MODE] Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your query: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            response = self.process_query(user_input)
            if response:
                print(self.formatter.to_text(response))

    # test mode sa predetermined pitanjima
    def run_test_mode(self):
        print("\n[TEST MODE] Running predefined test scenarios...")

        test_queries = [
            "Celotna prodaja v vseh trgovinah v Ljubljani",
            "Kaj pa v Kopru?",
            "Primerjaj prodajo elektronike v Ljubljani in v Celju",
            "Ali se Apple izdelki boljše prodajajo v Kopru in Celju skupaj ali v Ljubljani?",
            "Kako učinkovite so bile promocije tipa Flash Sale?",
            "Ali so v prodaji v letu 2021 kakšne anomalije?",
            "Zakaj se Milka čokolada bolje prodaja v Celju kot v Kopru?",
        ]

        for query in test_queries:
            response = self.process_query(query)
            if response:
                print(self.formatter.to_text(response))
            else:
                print(f"Failed to process: {query}")

        print("\nTest mode completed.")


if __name__ == "__main__":

    latest_file = "./log.txt"
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write("")

    hub = RetailAnalyticsHub()

    print("\nSelect Mode:")
    print("1. User Input Mode")
    print("2. Test Mode")

    choice = input("Choice (1/2): ")

    if choice == '1':
        hub.run_user_mode()
    elif choice == '2':
        hub.run_test_mode()
    else:
        print("Invalid choice.")