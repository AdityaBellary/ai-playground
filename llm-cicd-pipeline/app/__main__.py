import sys

from app.llm_service import generate_response


def main():
    # One-shot: python -m app "your prompt"
    if len(sys.argv) > 1:
        print(generate_response(" ".join(sys.argv[1:])))
        return

    print("LLM app is running. Type 'exit' to stop.")
    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break
        if prompt.strip().lower() == "exit":
            break
        print(f"LLM: {generate_response(prompt)}")


if __name__ == "__main__":
    main()
