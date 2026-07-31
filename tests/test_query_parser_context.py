
import pytest

from agent.query_processing.query_parser import RetailQueryParser

pytestmark = pytest.mark.query_parser_context


@pytest.fixture
def parser():
    return RetailQueryParser(api_key="test-key", llm_model="test-model")


def test_no_group_by_clears_remembered_entities(parser):
    parser.last_result_entities = {"schema_field": "entities.products", "values": ["stale"]}

    parser.remember_result_entities([], {"results": [{"product_name": "iPhone 15"}]})

    assert parser.last_result_entities is None


def test_top_products_ranking_is_remembered_under_entities_products(parser):
    analysis_result = {"results": [
        {"product_name": "iPhone 15", "total_sales": 15900000.0},
        {"product_name": "Galaxy S24", "total_sales": 11900000.0},
        {"product_name": "MacBook Air M2", "total_sales": 11600000.0},
    ]}

    parser.remember_result_entities(["product_name"], analysis_result)

    assert parser.last_result_entities == {
        "schema_field": "entities.products",
        "values": ["iPhone 15", "Galaxy S24", "MacBook Air M2"],
    }


def test_store_city_groups_map_to_a_filter_field_not_an_entity(parser):
    analysis_result = {"results": [
        {"store_city": "Ljubljana", "total_sales": 100.0},
        {"store_city": "Maribor", "total_sales": 90.0},
    ]}

    parser.remember_result_entities(["store_city"], analysis_result)

    assert parser.last_result_entities["schema_field"] == "filters.store_city"
    assert parser.last_result_entities["values"] == ["Ljubljana", "Maribor"]


def test_duplicate_values_are_deduped_keeping_first_occurrence_order(parser):
    analysis_result = {"results": [
        {"product_category": "Electronics", "total_sales": 100.0},
        {"product_category": "Food", "total_sales": 80.0},
        {"product_category": "Electronics", "total_sales": 20.0},
    ]}

    parser.remember_result_entities(["product_category"], analysis_result)

    assert parser.last_result_entities["values"] == ["Electronics", "Food"]


def test_unmapped_group_by_column_is_not_remembered(parser):
    parser.remember_result_entities(["some_unmapped_column"], {"results": [{"some_unmapped_column": "x"}]})

    assert parser.last_result_entities is None


def test_empty_results_is_not_remembered(parser):
    parser.remember_result_entities(["product_name"], {"results": []})

    assert parser.last_result_entities is None


def test_missing_analysis_result_does_not_crash(parser):
    parser.remember_result_entities(["product_name"], None)

    assert parser.last_result_entities is None


def test_rows_missing_the_group_by_key_are_skipped_not_crashed(parser):
    analysis_result = {"results": [
        {"product_name": "iPhone 15", "total_sales": 100.0},
        {"total_sales": 50.0},
    ]}

    parser.remember_result_entities(["product_name"], analysis_result)

    assert parser.last_result_entities["values"] == ["iPhone 15"]


def test_parse_injects_previous_result_entities_message(parser, monkeypatch):

    captured = {}

    class _FakeResponse:
        class _Choice:
            class _Message:
                content = '{"query_type": "sales_trend", "metrics": ["total_sales"]}'
            message = _Message()
        choices = [_Choice()]

    def _fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse()

    monkeypatch.setattr(parser.client.chat.completions, "create", _fake_create)

    parser.last_result_entities = {"schema_field": "entities.products", "values": ["iPhone 15", "Galaxy S24"]}
    parser.parse("za vsaki produkt od teh mi napiši še prodano količino")

    system_messages = " ".join(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert "PREVIOUS_RESULT_ENTITIES" in system_messages
    assert "iPhone 15" in system_messages
    assert "entities.products" in system_messages
