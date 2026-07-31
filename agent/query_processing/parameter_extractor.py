import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParameterExtractionError(Exception):
    # Raised when parameter extraction fails.
    pass


class ValidationError(Exception):
    # Raised when validation fails.
    pass


class ParameterExtractor:
    
    # Extracts and validates parameters from LLM-generated query dictionaries.
    
    # Converts structured dictionaries into pandas-compatible filtering and aggregation
    # instructions for retail analytics operations.
    
    # Known categorical values from dataset

    FUZZY_MATCH_THRESHOLD = 0.75

    VALID_VALUES = {
        "gender": ["Male", "Female", "Other"],
        "income_bracket": ["Low", "Medium", "High"],
        "loyalty_program": ["Yes", "No"],
        "churned": ["Yes", "No"],
        "marital_status": ["Single", "Married", "Divorced"],
        "education_level": ["High School", "Bachelor's", "Master's", "PhD"],
        "payment_method": ["Cash", "Credit Card", "Debit Card", "Mobile Payment"],
        "season": ["Spring", "Summer", "Fall", "Winter"],
        "holiday_season": ["Yes", "No"],
        "weekend": ["Yes", "No"],
        "discount_applied": ["Yes", "No"],
        "product_category": ["Electronics", "Food", "Toys", "Clothing", "Drinks"],
        "promotion_type": ["20% Off", "Flash Sale", "Buy One Get One Free"],
        "customer_city": ["Ljubljana", "Maribor", "Celje", "Koper"],
        "customer_state": ["Osrednjeslovenska", "Podravska", "Štajerska", "Primorska"],
        "store_city": ["Ljubljana", "Maribor", "Celje", "Koper"],
        "store_location_within_city": ['Ljubljana BTC City', 'Ljubljana Citypark',
                                       'Maribor Europark', 'Maribor Qlandia',
                                       'Celje Citycenter', 'Koper Supernova'],
        "store_state": ["Osrednjeslovenska", "Podravska", "Štajerska", "Primorska"]
    }

    VALID_PRODUCT_NAMES = [
        'iPad 10. generacije', 'Milka lešnik čokolada', 'Nerf Elite blaster',
        'Barbie sanjska hiša', 'Oversized blazer', 'Hoodie pulover',
        'LEGO Classic kocke', 'Penne Rigate', 'Paradižnikova juha',
        'Osnovna majica', '7UP', 'LG hladilnik', 'Cargo hlače',
        'Slim fit jeans hlače', 'Vegeta začimba', 'Osnovni top',
        'Fižolova enolončnica', 'Fusilli testenine', 'LEGO Star Wars X-Wing',
        'Jabolčni nektar', "Lay's klasični čips", 'Play-Doh set za modeliranje',
        'Pepsi', 'LG OLED televizor', 'Uno kartna igra', 'iPhone 15',
        'Monopoly klasična izdaja', 'Operacija družabna igra', 'Pesto Genovese omaka',
        'Radenska Classic mineralna voda', 'Radenska Naturelle',
        'Milka piškoti s čokolado', 'Galaxy Tab S9', 'LG zvočna vrstica',
        'AIRism spodnje perilo', 'Milka Alpska mlečna čokolada', 'MacBook Air M2',
        'Pleten pulover', 'Gatorade pomaranča', 'LG pralni stroj', 'Špageti št. 5',
        'Hot Wheels dirkalna steza', 'Ravne jeans hlače', 'Fisher-Price učna igrača',
        'Breskov sok', 'Milka Oreo čokolada', 'Pomarančni sok', 'Sadni napitek',
        'AirPods Pro', 'Galaxy S24', 'Samsung QLED televizor', 'Jeans jakna',
        'Radenska limeta gazirana voda', 'Radenska 1,5 L plastenka',
        'LEGO City policijska postaja', 'LEGO Technic avto', 'Ultra Light Down jakna',
        'Heattech majica', 'Ajvar', 'Galaxy Buds slušalke'
    ]
    
    VALID_BRANDS = [
        'Apple', 'Milka', 'Hasbro', 'Mattel', 'Zara', 'HM', 'LEGO',
        'Barilla', 'Podravka', 'PepsiCO', 'LG', 'Fructal', 'Radenska',
        'Samsung', 'Uniqlo'
    ]
    
    # Boolean fields that need string conversion
    BOOLEAN_FIELDS = [
        "loyalty_program",
        "churned",
        "discount_applied",
        "weekend",
        "holiday_season"
    ]
    
    # Valid metrics
    VALID_METRICS = [
        "total_sales", "total_transactions", "quantity", "unit_price",
        "discount_applied", "product_rating", "promotion_effectiveness"
    ]

    # LLM vcasih naredi malo razliko v imenu, ceprov ima closed set v schemi
    # ovo je backup verifikacija u slucaju da se to desi
    METRIC_SYNONYMS = {
        "revenue": "total_sales",
        "sales": "total_sales",
        "spending": "total_sales",
        "total_spent": "total_sales",
        "total_spending": "total_sales",
        "poraba": "total_sales",
        "prihodki": "total_sales",
        "units": "quantity",
        "units_sold": "quantity",
        "quantity_sold": "quantity",
        "kolicina": "quantity",
        "transactions": "total_transactions",
        "price": "unit_price",
        "rating": "product_rating",
    }
    
    GROUP_BY_COLUMNS = {
        "product_name": "product_name",
        "product": "product_name",
        "products": "product_name",
        "izdelek": "product_name",
        "product_category": "product_category",
        "category": "product_category",
        "kategorija": "product_category",
        "product_brand": "product_brand",
        "brand": "product_brand",
        "znamka": "product_brand",
        "store_city": "store_city",
        "store": "store_city",
        "city": "store_city",
        "trgovina": "store_city",
        "store_location": "store_location",
        "store_location_within_city": "store_location",
        "promotion_type": "promotion_type",
        "promotion": "promotion_type",
        "customer_id": "customer_id",
        "customer": "customer_id",
        "gender": "gender",
        "payment_method": "payment_method",
        "season": "season",
        "day_of_week": "day_of_week",
    }

    # Entity to column mapping
    ENTITY_COLUMNS = {
        "products": "product_name",
        "categories": "product_category",
        "brands": "product_brand",
        "promotions": "promotion_type",
        "stores": "store_city"
    }


    
    def __init__(self, fuzzy_matching: bool = True):
        # Initialize the parameter extractor.
        self.fuzzy_matching = fuzzy_matching
        logger.info("ParameterExtractor initialized")
    
    def extract(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        
        # Extract and validate parameters from query dictionary.
        
        # Args:
        #     query_dict: Dictionary from LLM query parser.
        
        # Returns:
        #     Dictionary with analysis-ready parameters.
        
        # Raises:
        #     ParameterExtractionError: If input is not a dictionary.
        #     ValidationError: If validation fails.
        
        # Validate input type
        if not isinstance(query_dict, dict):
            logger.error(f"Expected dict, got {type(query_dict)}")
            raise ParameterExtractionError(f"Input must be a dictionary, got {type(query_dict)}")
        
        logger.info("Processing query dictionary")
        
        # Extract and validate components
        result = {
            "metrics": self._extract_metrics(query_dict),
            "date_filter": self._extract_date_filter(query_dict),
            "filters": self._extract_filters(query_dict),
            "entity_filters": self._extract_entity_filters(query_dict),
            "group_by": self._extract_aggregation(query_dict, "group_by"),
            "sort_by": self._extract_aggregation(query_dict, "sort_by"),
            "sort_order": self._extract_aggregation(query_dict, "sort_order"),
            "limit": self._extract_aggregation(query_dict, "limit"),
            "comparison": self._extract_comparison(query_dict),
            "metadata": self._extract_metadata(query_dict)
        }
        
        logger.info("Parameters extracted successfully")
        return result
    
    def _extract_metrics(self, data: Dict[str, Any]) -> List[str]:
        
        # Extract and validate metrics.
        
        # Args:
        #     data: Query dictionary.
        
        # Returns:
        #     List of validated metric names.
        
        # Raises:
        #     ValidationError: If metric is unsupported.
        
        metrics = data.get("metrics", ["total_sales"])
        
        if not isinstance(metrics, list):
            metrics = ["total_sales"]
            logger.warning("Metrics not a list, using default: ['total_sales']")
        
        # Validate each metric. Try the synonym map before giving up.
        # A single unknown metric gives a warning instead of failing
        # the whole query.
        valid = []
        for metric in metrics:
            if metric in self.VALID_METRICS:
                valid.append(metric)
            elif str(metric).lower() in self.METRIC_SYNONYMS:
                mapped = self.METRIC_SYNONYMS[str(metric).lower()]
                logger.info(f"Mapped metric synonym '{metric}' -> '{mapped}'")
                valid.append(mapped)
            else:
                logger.warning(f"Dropping unsupported metric: '{metric}'. Valid: {self.VALID_METRICS}")

        if not valid:
            valid = ["total_sales"]

        metrics = list(dict.fromkeys(valid))
        logger.debug(f"Extracted metrics: {metrics}")
        return metrics
    
    def _extract_date_filter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        # Extract date filtering parameters.
        
        # Args:
        #     data: Query dictionary.
        
        # Returns:
        #     Dictionary with date filter instructions.
        
        time_data = data.get("time", {})
        
        start = time_data.get("start")
        end = time_data.get("end")
        granularity = time_data.get("granularity", "month")
        
        # Validate dates if provided
        if start:
            self._validate_date(start, "start")
        if end:
            self._validate_date(end, "end")
        
        # Check date range validity
        if start and end:
            if start > end:
                raise ValidationError(f"Invalid date range: start ({start}) > end ({end})")
        
        result = {
            "column": "transaction_date",
            "start": start,
            "end": end,
            "granularity": granularity
        }
        
        logger.debug(f"Extracted date filter: {result}")
        return result
    
    def _validate_date(self, date_str: str, field_name: str) -> None:
        
        # Validate date format.
        
        # Args:
        #     date_str: Date string to validate.
        #     field_name: Name of field for error messages.
        
        # Raises:
        #     ValidationError: If date format is invalid.
        
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValidationError(
                f"Invalid date format for {field_name}: '{date_str}'. Expected YYYY-MM-DD"
            )

    def _extract_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        # Extract and normalize attribute-based filters.
        
        # Args:
        #     data: Query dictionary.
        
        # Returns:
        #     Dictionary of normalized filters.
        
        filters_data = data.get("filters", {})
        result = {}
        
        # Process each filter
        for key, value in filters_data.items():
            if value is None or value == []:
                continue
            
            # Handle age_range separately
            if key == "age_range":
                if isinstance(value, list) and len(value) == 2:
                    if value[0] is not None:
                        result["age_min"] = value[0]
                    if value[1] is not None:
                        result["age_max"] = value[1]
                continue

            # Every branch below expects a list (filters arrive as lists
            # to support multi-value OR filtering, e.g. "Koper in Celje" ->
            # ["Koper", "Celje"]). V primeru da LLM vrne skalar namest list-a
            # Bez ovoga string bi se iteriro character po character umjesto ko jedna vrijednost
            if not isinstance(value, list):
                value = [value]

            # Skip fields that will be handled as entity filters
            if key in ["purchase_frequency_min", "product_rating_min"]:
                if len(value) > 0:
                    result[key] = value[0]
                continue
            
            # Normalize boolean fields
            if key in self.BOOLEAN_FIELDS:
                if len(value) > 0:
                    normalized_list = []
                    for v in value:
                        normalized = self._normalize_boolean(v)
                        if normalized:
                            normalized_list.append(normalized)
                    if normalized_list:
                        result[key] = normalized_list if len(normalized_list) > 1 else normalized_list[0]
                continue
            
            # Validate categorical values (case-insensitive; store the properly-cased version).
            # Ako jedna vrijednost fali vratimo warning umjesto da sve padne u vodu
            if key in self.VALID_VALUES:
                if len(value) > 0:
                    valid_values = []

                    for val in value:
                        matched = next(
                            (v for v in self.VALID_VALUES[key] if v.lower() == str(val).lower()),
                            None
                        )
                        if matched is None:
                            logger.warning(
                                f"Dropping unrecognized value for {key}: '{val}'. Valid: {self.VALID_VALUES[key]}"
                            )
                            continue
                        valid_values.append(matched)

                    if valid_values:
                        result[key] = valid_values if len(valid_values) > 1 else valid_values[0]
                continue

            # For any other filters
            if len(value) > 0:
                result[key] = value[0] if len(value) == 1 else value
        
        logger.debug(f"Extracted filters: {result}")
        return result
    
    def _normalize_boolean(self, value: Any) -> Optional[str]:
        
        # Convert boolean to dataset-compatible string.
        
        # Args:
        #     value: Boolean or string value.
        
        # Returns:
        #     "Yes", "No", or None.
        
        if value is True:
            return "Yes"
        elif value is False:
            return "No"
        elif isinstance(value, str) and value.lower() == "yes":
            return "Yes"
        elif isinstance(value, str) and value.lower() == "no":
            return "No"
        else:
            logger.warning(f"Unexpected boolean value: {value}")
            return None
    
    def _extract_entity_filters(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        
        # Extract entity-based filters (list filters).
        
        # Args:
        #     data: Query dictionary.
        
        # Returns:
        #     Dictionary mapping column names to value lists.
        
        entities_data = data.get("entities", {})
        result = {}
        
        for schema_key, column_name in self.ENTITY_COLUMNS.items():
            values = entities_data.get(schema_key, [])

            if values and isinstance(values, list):

                # Fuzzy matching for products and brands
                if schema_key == "products":
                    matched_values = self._fuzzy_match_list(
                        values, 
                        self.VALID_PRODUCT_NAMES,
                        "product"
                    )
                elif schema_key == "brands":
                    matched_values = self._fuzzy_match_list(
                        values,
                        self.VALID_BRANDS,
                        "brand"
                    )
                else:
                    matched_values = values
                
                if matched_values:
                    result[column_name] = matched_values
        
        # Handle customer filter (needs special mapping)
        customers = entities_data.get("customers", [])
        if customers:
            result["customer_id"] = customers
        
        logger.debug(f"Extracted entity filters: {result}")
        return result

    def _fuzzy_match_list(self, input_values: List[str], valid_values: List[str], entity_type: str) -> List[str]:
        
        # Fuzzy match a list of values against valid options.
        
        # Args:
        #     input_values: Values from LLM (may have typos)
        #     valid_values: Known valid values from dataset
        #     entity_type: Type of entity (for logging)
        
        # Returns:
        #     List of matched valid values
        
        matched = []
        
        for value in input_values:
            # Try exact match first
            exact_match = self._find_exact_match(value, valid_values)
            
            if exact_match:
                matched.append(exact_match)
                logger.debug(f"Exact match for {entity_type} '{value}': {exact_match}")
            elif self.fuzzy_matching:
                # Fall back to fuzzy matching
                fuzzy_match = self._find_fuzzy_match(value, valid_values)
                
                if fuzzy_match:
                    matched.extend(fuzzy_match)
                    logger.info(f"Fuzzy matched {entity_type} '{value}' → '{fuzzy_match}'")
                else:
                    logger.warning(f"No match found for {entity_type}: '{value}'")
            else:
                logger.warning(f"No exact match for {entity_type}: '{value}' (fuzzy matching disabled)")
        
        return list(set(matched))  # Remove duplicates
    
    def _find_exact_match(self, value: str, valid_values: List[str]) -> Optional[str]:
        
        # Find exact match (case-insensitive).
        
        # Args:
        #     value: Input value
        #     valid_values: List of valid values
        
        # Returns:
        #     Matched value or None
        
        value_lower = value.lower().strip()
        
        for valid in valid_values:
            if valid.lower().strip() == value_lower:
                return valid  # Return the correctly-cased version
        
        return None

    def _find_fuzzy_match(self, value: str, valid_values: List[str]) -> List[str]:
        # Find fuzzy match using rapidfuzz
        
        # Args:
        #     value: Input value (possibly with typos)
        #     valid_values: List of valid values
        
        # Returns:
        #     Best matched value or None

        # Find best match with score
        results = process.extract(
            value,
            valid_values,
            scorer=fuzz.WRatio,  # Handles word order
            score_cutoff=self.FUZZY_MATCH_THRESHOLD * 100,  # rapidfuzz uses 0-100 scale
            limit=None
        )
        if results:
            matches = [match for match, score, _ in results]
            return matches
        
        return []
    
    def _extract_aggregation(self, data: Dict[str, Any], field: str) -> Union[List[str], str, int, None]:
        
        # Extract aggregation parameters.
        
        # Args:
        #     data: Query dictionary.
        #     field: Aggregation field name.
        
        # Returns:
        #     Aggregation value or None.
        
        agg_data = data.get("aggregation", {})
        value = agg_data.get(field)
        
        if field == "group_by":
            if not isinstance(value, list):
                return []
            columns = []
            for item in value:
                mapped = self.GROUP_BY_COLUMNS.get(str(item).lower())
                if mapped is None:
                    logger.warning(f"Dropping unknown group_by value: '{item}'")
                elif mapped not in columns:
                    columns.append(mapped)
            return columns
        elif field == "sort_order":
            return value if value in ["asc", "desc"] else "desc"
        elif field == "limit":
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid limit value: {value}")
                    return None
        
        return value
    
    def _extract_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        # Extract comparison configuration using group-based comparisons.
        
        comp_data = data.get("comparison", {})

        groups = comp_data.get("groups", [])

        normalized_groups = []
        for group in groups:
            normalized_groups.append({
                "label": group.get("label"),
                "filters": group.get("filters", {}),
                "entities": group.get("entities", {})
            })

        return {
            "enabled": bool(comp_data.get("enabled", False)),
            "dimension": comp_data.get("dimension"),
            "groups": normalized_groups
        }
    
    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, str]:
        
        # Extract query metadata.
        
        # Args:
        #     data: Query dictionary.
        
        # Returns:
        #     Metadata dictionary.
        
        return {
            "query_type": data.get("query_type", "sales_trend"),
            "intent": data.get("intent", "summarize")
        }
