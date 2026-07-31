import pytest
from tests.helpers import load_golden, run_pipeline_with_parsed, assert_invariants

_sequences = load_golden("batch_12_iterative_context.json")

params = [
    pytest.param(seq, id=seq["id"],
                 marks=pytest.mark.representative if seq["representative"] else ())
    for seq in _sequences
]


@pytest.mark.batch12
@pytest.mark.parametrize("seq", params)
def test_batch12_iterative_context(seq, persistent_parser, extractor):
    for turn in seq["turns"]:
        result, parsed = run_pipeline_with_parsed(persistent_parser, extractor, turn["prompt"])
        if "expect_parsed" in turn:
            assert_invariants(parsed, turn["expect_parsed"])
        if "expect" in turn:
            assert_invariants(result, turn["expect"])
