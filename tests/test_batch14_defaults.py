import pytest
from tests.helpers import load_golden, import_query_utils, assert_invariants

_normalize_nulls, _finalize_query = import_query_utils()

_entries = load_golden("batch_14_defaults.json")

params = [
    pytest.param(e, id=e["id"],
                 marks=pytest.mark.representative if e["representative"] else ())
    for e in _entries
]


@pytest.mark.batch14
@pytest.mark.parametrize("entry", params)
def test_batch14_defaults(entry):
    if entry["func"] == "normalize_nulls":
        result = _normalize_nulls(entry["input"])
        assert result == entry["expected"], (
            f"normalize_nulls({entry['input']!r}) → {result!r}, expected {entry['expected']!r}"
        )
    else:
        result = _finalize_query(entry["input"].copy() if isinstance(entry["input"], dict) else entry["input"])
        assert_invariants(result, entry["expect"])
