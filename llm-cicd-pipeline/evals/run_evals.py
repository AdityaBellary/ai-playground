import json
import os
import sys

# Allow `python evals/run_evals.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_service import MODEL_NAME, generate_response

TEST_CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json")

# Fail the pipeline if fewer than 80% of cases pass
THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", "0.8"))

with open(TEST_CASES_PATH) as f:
    test_cases = json.load(f)


def run_evals():
    passed = 0

    for case in test_cases:
        response = generate_response(case["input"])
        keyword = case["expected_keyword"].lower()
        ok = keyword in response.lower()
        passed += ok

        print(f"[{'PASS' if ok else 'FAIL'}] {case['input']}  (expects '{keyword}')")

    score = passed / len(test_cases)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Eval score: {score:.2f} (threshold {THRESHOLD:.2f})")

    return score


if __name__ == "__main__":
    score = run_evals()

    if score < THRESHOLD:
        raise SystemExit("LLM evaluation failed")
