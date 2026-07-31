import json
import openai
import os
import re
from datetime import datetime
from dotenv import load_dotenv

from agent.query_processing.parameter_extractor import ParameterExtractor

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY").strip()
LLM_MODEL = os.getenv("LLM_MODEL").strip()
SAFETY_IDENTIFIER = os.getenv("SAFETY_IDENTIFIER").strip()

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not LLM_MODEL:
    raise ValueError("LLM_MODEL not found in .env file")
if not SAFETY_IDENTIFIER:
    raise ValueError("SAFETY_IDENTIFIER not found in .env file")

DIACRITIC_MAP = str.maketrans({"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d"})

PERIOD_BREAKDOWN_PATTERNS = [
    (re.compile(r"\bpo\s+(?:\w+\s+){0,2}dne(?:h|vih)\b|\bdnevn[aeio]\w*\b"), "day"),
    (re.compile(r"\bpo\s+(?:\w+\s+){0,2}tednih\b|\btedensk[aeio]\w*\b"), "week"),
    (re.compile(r"\bpo\s+(?:\w+\s+){0,2}mesecih\b|\bmesecn[aeio]\w*\b"), "month"),
    (re.compile(r"\bpo\s+(?:\w+\s+){0,2}letih\b|\bletn[aeio]\w*\b"), "year"),
]

PERIOD_TREND_PATTERN = re.compile(r"\bskozi\s+cas\b|\bpo\s+obdobjih\b")

EXPLAIN_PATTERN = re.compile(
    r"\bzakaj\b|\bpojasni\w*\b|\brazlozi\w*\b|\brazlag\w*\b|\bvzrok\w*\b|\binterpretiraj\w*\b"
)

EXPLAINABLE_QUERY_TYPES = {"sales_trend", "comparison", "customer_analysis", "product_analysis"}

RESULT_ENTITY_SCHEMA_FIELD = {
    "product_name": "entities.products",
    "product_category": "entities.categories",
    "product_brand": "entities.brands",
    "promotion_type": "entities.promotions",
    "customer_id": "entities.customers",
    "store_city": "filters.store_city",
    "store_location": "filters.store_location_within_city",
    "gender": "filters.gender",
    "payment_method": "filters.payment_method",
    "season": "filters.season",
    "day_of_week": "filters.day_of_week",
}


class RetailQueryParser:
    def __init__(self, api_key=None, llm_model=None, data_start=None, data_end=None):
        api_key = api_key or API_KEY
        llm_model = llm_model or LLM_MODEL

        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model = llm_model
        self.history = []

        self.data_start = data_start
        self.data_end = data_end


        # Which items the LAST answers rows actually covered
        # Entitete v proslem vprasanju, za followup uprasanja tipa
        # Kateri so najbolje prodani produkti v Ljubljani -> Za vsak od teh..
        self.last_result_entities = None

        self.schema = {
            "query_type": "sales_trend | comparison | promotion_analysis | anomaly_detection | customer_analysis | product_analysis | explain",
            "intent": "trend | compare | explain | detect | summarize",

            # metrics je predetermined, da LLM ne izmislja neke stvari koje ne postoje 
            # / drugacija imena za stvari koje postoje
            "metrics": ParameterExtractor.VALID_METRICS,

            "time": {
                "type": "absolute | relative | range | single",
                "start": "YYYY-MM-DD or None",
                "end": "YYYY-MM-DD or None",
                "granularity": "day | week | month | year"
            },

            "entities": {
                "products": [],
                "categories": ParameterExtractor.VALID_VALUES["product_category"],
                "brands": ParameterExtractor.VALID_BRANDS,
                "customers": [],
                "promotions": ParameterExtractor.VALID_VALUES["promotion_type"]
            },

            "filters": {
                "gender": ParameterExtractor.VALID_VALUES["gender"],
                "age_range": [None, None],
                "income_bracket": ParameterExtractor.VALID_VALUES["income_bracket"],
                "marital_status": ParameterExtractor.VALID_VALUES["marital_status"],
                "education_level": ParameterExtractor.VALID_VALUES["education_level"],

                "loyalty_program": ParameterExtractor.VALID_VALUES["loyalty_program"],
                "churned": ParameterExtractor.VALID_VALUES["churned"],
                "purchase_frequency_min": [],

                "payment_method": ParameterExtractor.VALID_VALUES["payment_method"],
                "discount_applied": ParameterExtractor.VALID_VALUES["discount_applied"],
                "weekend": ParameterExtractor.VALID_VALUES["weekend"],

                "season": ParameterExtractor.VALID_VALUES["season"],
                "holiday_season": ParameterExtractor.VALID_VALUES["holiday_season"],
                "day_of_week": [],

                "store_city": ParameterExtractor.VALID_VALUES["store_city"],
                "store_state": ParameterExtractor.VALID_VALUES["store_state"],
                "store_location_within_city": ParameterExtractor.VALID_VALUES["store_location_within_city"],
                "customer_city": ParameterExtractor.VALID_VALUES["customer_city"],
                "customer_state": ParameterExtractor.VALID_VALUES["customer_state"],

                "product_rating_min": [],
                "product_color": []
            },

            "comparison": {
                "enabled": False,
                "dimension": None,

                "groups": [
                    {
                        "label": None,
                        "filters": {},
                        "entities": {}
                    }
                ]
            },

            "context_reference": {
                "use_previous_query": False,
                "modification_type": "refine | override | extend"
            },

            "aggregation": {
                "group_by": ["product_name", "product_category", "product_brand",
                             "store_city", "store_location", "promotion_type", "customer_id"],
                "sort_by": None,
                "sort_order": "desc | asc",
                "limit": None
            }
        }

    def remember_result_entities(self, group_by_columns, analysis_result):
        
        # Zapamti entitete koje smo dobili sad, da ako sledece pitanje bude pitalo nesto
        # kao "Koji od tih produkta...", mozemo znat sta su "ti produkti".
        if not group_by_columns:
            self.last_result_entities = None
            return

        column = group_by_columns[0]
        schema_field = RESULT_ENTITY_SCHEMA_FIELD.get(column)
        if schema_field is None:
            self.last_result_entities = None
            return

        results = (analysis_result or {}).get("results", [])
        values = [r[column] for r in results if isinstance(r, dict) and column in r]
        values = list(dict.fromkeys(values))

        self.last_result_entities = (
            {"schema_field": schema_field, "values": values} if values else None
        )

    def get_system_prompt(self):
        dataset_line = ""
        if self.data_start and self.data_end:
            dataset_line = (
                f"\n        Dataset range: transactions span {self.data_start} to {self.data_end}. "
                f"Resolve relative time phrases ('lani', 'zadnje leto', 'letos') against the END of this range, "
                f"so they land on data that exists."
            )
        return f"""
        You are a Retail Data Expert. Convert user queries into the EXACT JSON schema provided.
        Current Date: {datetime.now().strftime('%Y-%m-%d')}{dataset_line}

        SCHEMA:
        {json.dumps(self.schema, indent=2)}
        
        RULES:
        1. Only use values specified in the schema strings (e.g., query_type must be one of the 6 provided).
        2. If a value is unknown, use an empty list or None.
        3. For relative time (e.g., 'last month'), calculate the specific YYYY-MM-DD.
        4. If the user is following up on a previous topic (an elliptical message like "Kaj pa v Kopru?", "Samo ženske stranke", "Ne, mislil sem 2023"), set use_previous_query to true AND copy EVERY field (entities, filters, time, metrics, query_type...) from the latest PREVIOUS_JSON_CONTEXT entry, changing ONLY what the new message changes - context must never be silently dropped. If the new message is a complete, self-contained question that stands on its own, set use_previous_query to false and ignore the previous context entirely.
        5. If user refers to a company/brand (e.g. Apple, Barilla, Milka, LG), put it in entities.brands.
        6. Only use entities.products for explicit individual products or generic product keywords (e.g. čokolada, iPhone 15, AirPods Pro).
        7. store_city is for specific cities like Ljubljana, Maribor, Celje, Koper.
        8. store_location is only for specific locations in the given cities, like "Ljubljana BTC City" or "Maribor Europark".
        9. If the user's query does not mention any time period (no dates, no relative phrase like 'last month', no year), leave time.type, time.start, and time.end as None. Do NOT default to the current date - "Current Date" above is only for resolving relative phrases (e.g. 'last month', 'this year'), not for filling in a time range that wasn't asked for.
        10. store_state is for Slovenian regions (e.g. Osrednjeslovenska, Podravska, Štajerska, Primorska) describing where the STORE is located - use it for queries about a region/regija. Do not confuse it with store_city (a specific city) or customer_state (the CUSTOMER's home region, which may differ from where they shopped).
        11. Any filter field in the schema whose example value is a list of specific strings (e.g. gender, education_level, payment_method, season, store_city, store_state, customer_city, customer_state, store_location_within_city, loyalty_program, churned, weekend, holiday_season, discount_applied) may ONLY be filled with one or more of those exact strings - they are the complete set of allowed values, not illustrative examples. This includes the Yes/No fields: even for something phrased like "kartico zvestobe" (has a loyalty card) or "brez naročnine" (no subscription), output the literal string "Yes" or "No" for that field - never a descriptive phrase. If the user's phrasing doesn't map cleanly to one of them, leave that field empty rather than inventing a new string.
        12. When the query asks to COMPARE two or more specific things (cities, brands, categories, demographics, etc.), set comparison.enabled=true and create ONE separate entry in comparison.groups PER compared item, each with its own label and its own specific filters/entities - e.g. for "Ljubljana vs Maribor": group 1 = {{"label": "Ljubljana", "filters": {{"store_city": "Ljubljana"}}}}, group 2 = {{"label": "Maribor", "filters": {{"store_city": "Maribor"}}}}. Do NOT merge the compared items into a single group, and do NOT put them together as a multi-value list in the top-level filters/entities (e.g. store_city: ["Ljubljana", "Maribor"]) instead of separate groups - that produces one combined result instead of one result per item being compared. Multi-value lists in filters/entities are only for the non-comparison case, where the user wants several values included together in ONE result (e.g. "sales in Koper and Celje combined"). Filters/entities SHARED by all compared groups (e.g. the city in "primerjaj prodajo hrane in elektronike v Mariboru") stay at the TOP level - each group carries only what makes it different.
        13. entities.categories is a CLOSED SET - only the five listed English category names are valid. When the user names a product CATEGORY in Slovenian, translate it: elektronika -> Electronics, hrana -> Food, igrače -> Toys, oblačila -> Clothing, pijače -> Drinks (any declined form: "pijač", "igrač", "oblačil", "elektronike"...). Category words go ONLY into entities.categories, NEVER into entities.products.
        14. entities.brands is a CLOSED SET - the listed brand names are the complete catalogue. Slovenian declensions of a brand still mean that brand: "Radenske" -> Radenska, "Milki"/"Milkinih" -> Milka, "Barille" -> Barilla, "Samsunga" -> Samsung, "Fructala" -> Fructal, "Hasbra" -> Hasbro. A brand name alone (even declined) goes into entities.brands; only put something in entities.products when a concrete product (or generic product keyword like "čokolada") is named. A brand combined with a category word ("LEGO igrače", "Radenske pijače") fills BOTH entities.brands and entities.categories - never dump the whole phrase into products.
        15. Ranking questions: "top/največ/najboljših N ..." -> aggregation.limit = N, sort_order = "desc" and sort_by = the metric; "najslabše/najmanj prodajani" -> sort_order = "asc". sort_order is "desc" whenever the user wants the biggest/best first - only use "asc" when they explicitly ask for the smallest/worst first. Superlatives about quantity sold ("največkrat prodan", "v največji količini") also add "quantity" to metrics. A superlative ranking question with NO explicit count ("kateri so najbolje prodani izdelki", "which products sell best") still means a manageable leaderboard, not the entire catalog - default aggregation.limit to 10 in that case rather than leaving it unset/unbounded.
        16. A query asking WHY something happened or asking to explain/interpret results ("zakaj", "pojasni", "razloži") has query_type "explain" - BUT two subjects take precedence over "explain": (a) if the question is about promotions or their effect/effectiveness (even a why-question like "zakaj je bila akcija neuspešna" or "dolgoročni učinek promocij"), use "promotion_analysis"; (b) if it is about a SUDDEN or UNUSUAL change ("nenadoma", "nenaden skok", "nenavaden porast", "anomalija", "outlier"), use "anomaly_detection". A gradual development the user wants explained ("razloži, kako se je prodaja razvijala") is "explain".
        17. entities.promotions is a CLOSED SET - the three listed promotion types are the only ones that exist. Any mention of a promotion maps to exactly one of them: "20% popust"/"20% Off" -> "20% Off", "Flash Sale"/"bliskovita akcija" -> "Flash Sale", "Buy One Get One Free"/"BOGO"/"1+1 gratis" -> "Buy One Get One Free". A generic mention of promotions ("promocije", "akcije") leaves the list empty. Comparing sales BEFORE and AFTER a promotion ("pred in po akciji") is query_type "promotion_analysis" (that before/during/after comparison is exactly what promotion analysis does), NOT "comparison".
        18. metrics is a CLOSED SET. Questions about spending/revenue ("koliko so porabili", "poraba", "prihodki", "revenue") use "total_sales"; questions about units sold ("koliko kosov", "količina", "največkrat prodan") use "quantity". Never invent metric names.
        19. Holiday mentions ("med prazniki", "v prazničnem času") map to filters.holiday_season = "Yes" - they are NOT a date range, so do not turn them into December dates. An explicitly named year/month alongside them still goes into time as usual.
        20. A ranking question NEEDS aggregation.group_by, otherwise there is nothing to rank. "Kateri izdelek/katera znamka ... naj..." and "top N X" MUST set group_by to the dimension being ranked, using the CLOSED SET of column names in the schema: izdelek -> "product_name", kategorija -> "product_category", znamka -> "product_brand", trgovina/prodajalna/mesto -> "store_city", lokacija -> "store_location", promocija -> "promotion_type", stranka/kupec -> "customer_id". Ranking questions use intent "summarize" (rank whole-period totals), NEVER intent "trend" - trend would rank each (time period, item) combination separately. They are also NOT comparisons: comparison.enabled stays false unless the user NAMES the specific things to compare. Only time rankings ("kateri mesec/teden/dan je bil najboljši") use intent "trend" with that granularity plus sort_by/sort_order/limit, with group_by left empty. Ranking questions follow RULES 9 like every other query: "kateri izdelek je imel najboljšo prodajo" mentions NO time period, so time.start and time.end stay None - never invent a date range for them.
        21. If a PREVIOUS_RESULT_ENTITIES message is present and this question refers back to those SAME items without naming different ones (e.g. "od teh", "za vsaki od njih/teh", "za te produkte/trgovine/stranke", "in each of these"), put its exact value list into the schema field it names, and set aggregation.group_by, sort_by and limit back to empty/None. Do NOT re-derive a fresh ranking to figure out "these" again - a different sort_by (e.g. switching from sales value to quantity) can surface a completely different top-N than the one already shown, which would silently answer about the wrong items. The goal of such a follow-up is reporting a new metric for the SAME already-identified items, not re-ranking everything from scratch.
        22. Set INTENT (never query_type) to "trend" when the question asks for one row PER time period: "po mesecih/tednih/dnevih/letih", "mesečno", "dnevno", "skozi čas", "po obdobjih". Put that unit in time.granularity (mesec -> month, teden -> week, dan -> day, leto -> year) and leave aggregation.group_by empty unless a further dimension is named ("po kategorijah po mesecih"). Two limits on this rule: (a) it changes intent ONLY - a "zakaj/razloži/pojasni" question still gets query_type "explain" per RULE 16, and a ranking still follows RULE 20; (b) merely NAMING a period is not a breakdown request - "v letu 2023", "januarja 2024", "lani" fill time.start/end per RULE 9 and leave time.granularity alone.
        23. Slovenian input often arrives without diacritics ("igrac" = "igrač", "cokolada" = "čokolada", "oblacila" = "oblačila"). Match c/s/z as possible č/š/ž against the closed sets in RULES 13, 14 and 17.
        """

    def parse(self, user_input):
        messages = [{"role": "system", "content": self.get_system_prompt()}]

        # Get the history if there is any and append it
        if self.history:
            prev_context = json.dumps(self.history[-3:])
            messages.append({"role": "system", "content": f"PREVIOUS_JSON_CONTEXT: {prev_context}"})

        if self.last_result_entities:
            values_json = json.dumps(self.last_result_entities["values"], ensure_ascii=False)
            schema_field = self.last_result_entities["schema_field"]
            messages.append({"role": "system", "content": (
                f"PREVIOUS_RESULT_ENTITIES: the previous answer's rows covered exactly "
                f"these values: {values_json}, from the field {schema_field}. "
                f"See RULE 21 for how to use this."
            )})

        # Add the user query
        messages.append({"role": "user", "content": user_input})
        
        # Try to parse the query
        try:
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1500,
                safety_identifier=SAFETY_IDENTIFIER
            )

            raw_content = response.choices[0].message.content
            parsed_json = json.loads(raw_content)
            
            # Cleaning up the json with preset defaults
            clean_json = finalize_query(parsed_json)
            # "po mesecih" & co. must produce a series, not one total - fix the
            # intent deterministically instead of hoping the LLM labelled it right
            clean_json = apply_period_breakdown(clean_json, user_input)
            # "zakaj / pojasni / razloži" must reach the explain route, which pairs
            # the sales numbers with the anomaly detector (RULE 16)
            clean_json = apply_explain_routing(clean_json, user_input)


            self.history.append(clean_json)
            # Limit the context for testing
            if len(self.history) > 5:
                    self.history.pop(0)

            return clean_json

        except Exception as e:
            return {"error" : f"Parser failed: {str(e)}"}


from datetime import datetime

def normalize_nulls(obj):
    
    # Recursively convert string null-like values to Python None.
    
    if isinstance(obj, dict):
        return {k: normalize_nulls(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [normalize_nulls(v) for v in obj]

    if isinstance(obj, str) and obj.strip().lower() in {
        "none", "null", "", "n/a"
    }:
        return None

    return obj

def apply_period_breakdown(parsed_data, user_input):

    # Force intent "trend" when the question asks for a per-period breakdown.
    #
    # SalesAnalyzer buckets rows into periods ONLY when intent == "trend", so a
    # question like "prodaja igrač v Ljubljani po mesecih" that the LLM labelled
    # "summarize" comes back as one total with no monthly series in it at all -
    # and the summary then tells the user there is no monthly data.
    #
    # Runs after finalize_query, so time.granularity already exists and can just
    # be overwritten with the unit the user actually named.

    if not isinstance(user_input, str):
        return parsed_data

    text = user_input.lower().translate(DIACRITIC_MAP)

    granularity = None
    for pattern, unit in PERIOD_BREAKDOWN_PATTERNS:
        if pattern.search(text):
            granularity = unit
            break

    # Nothing asked for a breakdown over time - leave the LLM's intent alone
    if granularity is None and not PERIOD_TREND_PATTERN.search(text):
        return parsed_data

    parsed_data["intent"] = "trend"
    if granularity is not None:
        parsed_data.setdefault("time", {})["granularity"] = granularity

    return parsed_data


def apply_explain_routing(parsed_data, user_input):

    # Force query_type "explain" for why/interpret questions.
    #
    # RULE 16 says a "zakaj / pojasni / razloži" question is query_type "explain",
    # but the LLM keeps picking the substantive type instead (sales_trend for
    # "razloži gibanje prodaje", customer_analysis for "zakaj stranke ..."). That
    # costs the user the anomaly context, because ResultsAggregator only pairs the
    # sales numbers with the anomaly detector for "explain".
    #
    # promotion_analysis and anomaly_detection are left untouched - they are the
    # two exceptions RULE 16 spells out.

    if not isinstance(user_input, str):
        return parsed_data

    text = user_input.lower().translate(DIACRITIC_MAP)
    if not EXPLAIN_PATTERN.search(text):
        return parsed_data

    if parsed_data.get("query_type") in EXPLAINABLE_QUERY_TYPES:
        parsed_data["query_type"] = "explain"

    return parsed_data


def finalize_query(parsed_data):

    # Applies defaults and normalization to parsed LLM JSON output.
    # Ensures all required sections exist and have safe defaults so downstream
    # modules (parameter extractor, analyzers) can work reliably.
    
    parsed_data = normalize_nulls(parsed_data)

    # 1. Metrics Defaults
    if not parsed_data.get("metrics"):
        parsed_data["metrics"] = ["total_sales"]

    # 2. Time Defaults
    t = parsed_data.setdefault("time", {})
    t.setdefault("start", None)
    t.setdefault("end", None)

    if t.get("start") is None and t.get("end") is None:
        t["type"] = None
    elif t.get("type") is None:
        t["type"] = "absolute"

    if t.get("granularity") is None:
        t["granularity"] = "month"

    # 3. Entities Defaults
    entities = parsed_data.setdefault("entities", {})

    for key in [
        "products",
        "categories",
        "brands",
        "stores",
        "customers",
        "promotions"
    ]:
        if key not in entities or entities[key] is None:
            entities[key] = []

    # 4. Filters Defaults
    filters = parsed_data.setdefault("filters", {})

    filter_keys = [
        # Demographics
        "gender",
        "income_bracket",
        "marital_status",
        "education_level",

        # Customer behavior
        "loyalty_program",
        "churned",
        "purchase_frequency_min",

        # Transaction attributes
        "payment_method",
        "discount_applied",
        "weekend",

        # Temporal
        "season",
        "holiday_season",
        "day_of_week",

        # Geographic
        "store_location",
        "store_state",
        "customer_city",
        "customer_state",

        # Product
        "product_rating_min",
        "product_color"
    ]

    for key in filter_keys:
        if key not in filters:
            filters[key] = None

    # Range filters
    if "age_range" not in filters or filters["age_range"] is None:
        filters["age_range"] = [None, None]

    # 5. Comparison Defaults
    comparison = parsed_data.setdefault("comparison", {})

    if "enabled" not in comparison:
        comparison["enabled"] = False

    if "dimension" not in comparison:
        comparison["dimension"] = None

    if "groups" not in comparison or comparison["groups"] is None:
        comparison["groups"] = []

    # Normalize comparison groups
    for group in comparison["groups"]:
        if "label" not in group:
            group["label"] = None

        if "filters" not in group or group["filters"] is None:
            group["filters"] = {}

        if "entities" not in group or group["entities"] is None:
            group["entities"] = {}

    # 6. Context Defaults
    ctx = parsed_data.setdefault("context_reference", {})

    if "use_previous_query" not in ctx:
        ctx["use_previous_query"] = False

    if "modification_type" not in ctx:
        ctx["modification_type"] = None

    # 7. Aggregation Defaults
    agg = parsed_data.setdefault("aggregation", {})

    if "group_by" not in agg or agg["group_by"] is None:
        agg["group_by"] = []

    if "sort_by" not in agg:
        agg["sort_by"] = None

    if "sort_order" not in agg or agg["sort_order"] is None:
        agg["sort_order"] = "desc"

    if "limit" not in agg:
        agg["limit"] = None

    return parsed_data

# Testing
if __name__ == "__main__":
    parser = RetailQueryParser()
    
    # Round 1
    query_1 = "How was the quantity sold in London for 2019?"
    result_1 = parser.parse(query_1)
    print("Round 1 Output:", json.dumps(result_1, indent=2))
        
    # Round 2 - Context follow-up
    query_2 = "What about in Berlin?"
    result_2 = parser.parse(query_2)
    print("\nRound 2 (Contextual) Output:", json.dumps(result_2, indent=2))