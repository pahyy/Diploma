import importlib
import json
import os
import pathlib
import re
import sys
from unittest.mock import patch

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
DATE_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Golden file loading
# ---------------------------------------------------------------------------

def load_golden(filename: str):
    with open(GOLDEN_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_internal(parser, extractor, prompt):
    parsed = parser.parse(prompt)
    if "error" in parsed:
        raise RuntimeError(f"Parser failed: {parsed['error']}")
    result = extractor.extract(parsed)
    return result, parsed


def run_pipeline(parser, extractor, prompt):
    result, _ = _run_internal(parser, extractor, prompt)
    return result


def run_pipeline_with_parsed(parser, extractor, prompt):
    return _run_internal(parser, extractor, prompt)


def assert_loose_date_filter(df):
    assert df["column"] == "transaction_date", f"column: {df['column']!r}"
    if df["start"] is None and df["end"] is None:
        return
    assert DATE_PAT.match(str(df["start"])), f"start not YYYY-MM-DD: {df['start']!r}"
    assert DATE_PAT.match(str(df["end"])), f"end not YYYY-MM-DD: {df['end']!r}"
    assert df["start"] <= df["end"], f"start > end: {df['start']} > {df['end']}"


def _get_at_path(obj, path: str):
    for part in path.split("."):
        if not isinstance(obj, dict):
            raise KeyError(f"Cannot traverse '{part}' — got {type(obj).__name__}")
        if part not in obj:
            raise KeyError(f"Key '{part}' not found (path '{path}')")
        obj = obj[part]
    return obj



def assert_invariants(result: dict, expect_list: list) -> None:

    for a in expect_list:
        path = a.get("path", "")

        if "any_of" in a:
            last_err = AssertionError("any_of: empty list")
            for sub in a["any_of"]:
                try:
                    assert_invariants(result, [sub])
                    last_err = None
                    break

                except (AssertionError, KeyError) as e:
                    last_err = e
            if last_err is not None:
                raise AssertionError(f"any_of: none passed — last: {last_err}")
            continue

        if a.get("loose_date"):
            assert_loose_date_filter(_get_at_path(result, path) if path else result)
            continue

        if "try_paths" in a:
            target = a.get("eq") if "eq" in a else a.get("contains")
            op = "eq" if "eq" in a else "contains"
            for p in a["try_paths"]:
                try:
                    v = _get_at_path(result, p)
                    if op == "eq" and v == target:
                        break
                    if op == "contains" and target in v:
                        break
                except (KeyError, TypeError):
                    pass
            else:
                raise AssertionError(
                    f"try_paths {a['try_paths']}: none matched {target!r}"
                )
            continue

        value = _get_at_path(result, path)

        if "eq" in a:
            assert value == a["eq"], f"[{path}] expected {a['eq']!r}, got {value!r}"

        elif "eq_icase" in a:
            assert str(value).lower() == str(a["eq_icase"]).lower(), (
                f"[{path}] expected {a['eq_icase']!r} (icase), got {value!r}"
            )

        elif "contains" in a:
            assert a["contains"] in value, (
                f"[{path}] {a['contains']!r} not in {value!r}"
            )

        elif "contains_icase" in a:
            target_lo = str(a["contains_icase"]).lower()
            if isinstance(value, list):
                members = [str(x).lower() for x in value]
                assert any(target_lo in m or m == target_lo for m in members), (
                    f"[{path}] {a['contains_icase']!r} not found (icase) in {value!r}"
                )
            else:
                assert target_lo in str(value).lower(), (
                    f"[{path}] {a['contains_icase']!r} not in {value!r}"
                )

        elif "contains_any_icase" in a:
            candidates = [str(x).lower() for x in a["contains_any_icase"]]
            if isinstance(value, list):
                members = [str(x).lower() for x in value]
                ok = any(c in m for c in candidates for m in members)
            else:
                val_lo = str(value).lower()
                ok = any(c in val_lo for c in candidates)
            assert ok, (
                f"[{path}] none of {a['contains_any_icase']!r} found in {value!r}"
            )

        elif "not_key" in a:
            assert a["not_key"] not in value, (
                f"[{path}] unexpected key {a['not_key']!r}"
            )

        elif "eq_set" in a:
            assert set(value) == set(a["eq_set"]), (
                f"[{path}] expected set {sorted(a['eq_set'])!r}, got {sorted(value)!r}"
            )

        elif "subset" in a:
            missing = set(a["subset"]) - set(value)
            assert not missing, f"[{path}] missing: {sorted(missing)!r}"

        elif "subset_labels" in a:
            
            actual_labels = [g.get("label") or "" for g in value]
            for lbl in a["subset_labels"]:
                assert any(lbl in al for al in actual_labels), (
                    f"[{path}] label '{lbl}' not in group labels {actual_labels!r}"
                )

        elif "count" in a:
            assert len(value) == a["count"], (
                f"[{path}] expected len {a['count']}, got {len(value)}"
            )

        elif "truthy" in a:
            assert bool(value) is a["truthy"], (
                f"[{path}] expected truthy={a['truthy']}, got {value!r}"
            )

        elif "date_format" in a:
            assert value is not None, f"[{path}] is None, expected YYYY-MM-DD"
            assert DATE_PAT.match(str(value)), f"[{path}] {value!r} is not YYYY-MM-DD"

        else:
            raise ValueError(f"Unknown assertion op in: {a!r}")


def import_query_parser_module():
    mod_key = "agent.query_processing.query_parser"
    if mod_key not in sys.modules:
        has_real_credentials = all([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("LLM_MODEL"),
            os.getenv("SAFETY_IDENTIFIER"),
        ])
        if has_real_credentials:
            importlib.import_module(mod_key)
        else:
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "dummy",
                "LLM_MODEL": "dummy-model",
                "SAFETY_IDENTIFIER": "dummy-id",
            }):
                importlib.import_module(mod_key)
    return sys.modules[mod_key]


def import_query_utils():
    mod = import_query_parser_module()
    return mod.normalize_nulls, mod.finalize_query
