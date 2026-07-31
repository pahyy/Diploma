import pytest
from tests.helpers import load_golden, run_pipeline, assert_invariants

_entries = load_golden("batch_11_explanation.json")

params = [
    pytest.param(e, id=e["prompt"][:60],
                 marks=pytest.mark.representative if e["representative"] else ())
    for e in _entries
]


@pytest.mark.batch11
@pytest.mark.parametrize("entry", params)
def test_batch11_explanation(entry, parser, extractor):
    result = run_pipeline(parser, extractor, entry["prompt"])
    assert_invariants(result, entry["expect"])
