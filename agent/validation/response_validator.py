import re
from itertools import chain

NUMBER_PATTERN = re.compile(
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"    # 1.234.567,89
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # 1,234,567.89
    r"|\d+,\d+"                        # 6,9
    r"|\d+\.\d+"                       # 6.9
    r"|\d+"                            # 42
)

SCALES = (1, 1_000, 1_000_000, 1_000_000_000)

IGNORE_BELOW = 11

class ResponseValidator:

    def validate(self, summary_text: str, analysis: dict, params: dict = None) -> dict:
        known = collect_numbers(analysis)
        if params is not None:
            # Validated inputs count as known values too. A summary that echoes the
            # period or age range the user asked about ("v letu 2024") is not making
            # anything up, but the year only appears in the analysis when the result
            # happens to carry _period labels - so on a summarize query it would be
            # reported as unverified.
            known |= collect_numbers(params)
        found = extract_numbers(summary_text)

        unverified = []
        checked = 0
        for raw_text, value in found:
            if value < IGNORE_BELOW and float(value).is_integer():
                continue
            checked += 1
            if not self._is_grounded(value, raw_text, known):
                unverified.append(raw_text)

        return {
            "valid": len(unverified) == 0,
            "numbers_checked": checked,
            "numbers_unverified": len(unverified),
            "unverified": unverified,
        }

    def _is_grounded(self, value: float, raw_text: str, known: set) -> bool:
        decimals = count_decimals(raw_text)
        for known_value in known:
            for scale in SCALES:
                if round(abs(known_value) / scale, decimals) == value:
                    return True
        return False


def count_decimals(raw_text: str) -> int:
    normalized = normalize_number_text(raw_text)
    if "." in normalized:
        return len(normalized.split(".")[1])
    return 0


def normalize_number_text(raw_text: str) -> str:
    text = raw_text
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", text):
        text = text.replace(".", "").replace(",", ".")   # 1.234,56 -> 1234.56
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")                     # 1,234.56 -> 1234.56
    else:
        text = text.replace(",", ".")                    # 6,9 -> 6.9
    return text


def extract_numbers(text: str):
    results = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group()
        try:
            results.append((raw, abs(float(normalize_number_text(raw)))))
        except ValueError:
            continue
    return results


def collect_numbers(obj) -> set:
    numbers = set()
    if isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        numbers.add(float(obj))
    elif isinstance(obj, str):
        for _, value in extract_numbers(obj):
            numbers.add(value)
    elif isinstance(obj, dict):
        for value in chain(obj.keys(), obj.values()):
            numbers.update(collect_numbers(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            numbers.update(collect_numbers(value))
    return numbers