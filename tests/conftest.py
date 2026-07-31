import os
import pytest
from agent.query_processing.parameter_extractor import ParameterExtractor


def _has_credentials() -> bool:
    return all([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("LLM_MODEL"),
        os.getenv("SAFETY_IDENTIFIER"),
    ])


@pytest.fixture
def parser():
    if not _has_credentials():
        pytest.skip("LLM credentials not configured "
                    "(set OPENAI_API_KEY, LLM_MODEL, SAFETY_IDENTIFIER)")
    from agent.query_processing.query_parser import RetailQueryParser
    return RetailQueryParser()


@pytest.fixture
def persistent_parser():
    if not _has_credentials():
        pytest.skip("LLM credentials not configured "
                    "(set OPENAI_API_KEY, LLM_MODEL, SAFETY_IDENTIFIER)")
    from agent.query_processing.query_parser import RetailQueryParser
    return RetailQueryParser()


@pytest.fixture
def extractor():
    return ParameterExtractor()
