import ollama
import pytest

from app.llm_service import MODEL_NAME, generate_response


def ollama_has_model():
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        return False
    return any(name == MODEL_NAME or name.split(":")[0] == MODEL_NAME for name in names)


pytestmark = pytest.mark.skipif(
    not ollama_has_model(),
    reason=f"Ollama isn't running or model '{MODEL_NAME}' isn't pulled",
)


def test_model_returns_non_empty_text():
    response = generate_response("Reply with one word: hello")

    assert isinstance(response, str)
    assert response.strip()
