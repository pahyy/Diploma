import pytest
from tests.helpers import load_golden, run_pipeline, assert_invariants

_entries = load_golden("batch_02_comparison.json")

params = [
    pytest.param(e, id=e["prompt"][:60],
                 marks=pytest.mark.representative if e["representative"] else ())
    for e in _entries
]


@pytest.mark.batch02
@pytest.mark.parametrize("entry", params)
def test_batch02_comparison(entry, parser, extractor):
    result = run_pipeline(parser, extractor, entry["prompt"])
    assert_invariants(result, entry["expect"])
