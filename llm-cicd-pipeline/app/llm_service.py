import os

from ollama import chat

# Set MODEL_NAME to switch models without changing code
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3")


def generate_response(prompt):
    response = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        # qwen3 is a reasoning model; keep its thinking out of the answer
        think=False,
    )

    return response.message.content
