from types import SimpleNamespace
from unittest.mock import patch

from app import llm_service


def fake_response(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_generate_response_returns_message_content():
    with patch.object(llm_service, "chat", return_value=fake_response("Hi there")):
        assert llm_service.generate_response("Hello") == "Hi there"


def test_generate_response_sends_prompt_as_user_message():
    with patch.object(llm_service, "chat", return_value=fake_response("")) as chat:
        llm_service.generate_response("What is Python?")

    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == llm_service.MODEL_NAME
    assert kwargs["messages"] == [{"role": "user", "content": "What is Python?"}]
