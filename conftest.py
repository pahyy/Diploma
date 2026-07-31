import os
import sys
from dotenv import load_dotenv

# Add the project root to sys.path so tests can import agent.*
sys.path.insert(0, os.path.dirname(__file__))

# Load the real .env first so its values take precedence.
# setdefault below only kicks in when running without a .env (e.g. CI).
load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder-key")
os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")
os.environ.setdefault("SAFETY_IDENTIFIER", "test-placeholder")
