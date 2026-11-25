import os
import time
from difflib import SequenceMatcher

import ollama
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "evaluation_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation_results.csv")

# Any model you've pulled with `ollama pull`
MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

PROMPT_VERSIONS = ["v1", "v2"]

# Fixed evaluation set: rerun the same questions after every change
eval_data = pd.DataFrame({
    "question": [
        "What is overfitting?",
        "What is RAG?",
        "Why do we use a validation set?",
        "What is an AI agent?",
    ],
    "context": [
        "Overfitting occurs when a model learns training data too closely and performs poorly on unseen data.",
        "RAG combines information retrieval with generation so an LLM can answer using external knowledge.",
        "A validation set is used to tune model choices and hyperparameters without using the final test set.",
        "An AI agent is a system that can reason about a task and use tools or actions to accomplish it.",
    ],
    "ground_truth": [
        "Overfitting happens when a model fits training data too closely and fails to generalize.",
        "RAG retrieves relevant information and provides it to an LLM to generate a grounded response.",
        "A validation set helps tune the model while keeping the test set unseen.",
        "An AI agent can reason about a task and use tools to take actions.",
    ],
})


def local_model(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


def llm_application(question, context, prompt_version="v1"):
    if prompt_version == "v1":
        prompt = f"""
        Answer the question using the provided context.

        Context:
        {context}

        Question:
        {question}
        """
    else:
        prompt = f"""
        Answer the question using only the provided context.
        If the answer is not supported by the context, say you
        don't have enough information.

        Context:
        {context}

        Question:
        {question}
        """

    return local_model(prompt)


def run_with_latency(question, context, prompt_version):
    start = time.perf_counter()
    answer = llm_application(question, context, prompt_version)
    latency = time.perf_counter() - start

    return answer, latency


def similarity_score(prediction, reference):
    # Character-level overlap: a rough proxy, not a measure of correctness
    return SequenceMatcher(None, prediction.lower(), reference.lower()).ratio()


def evaluate_prompt(prompt_version):
    results = []

    for _, row in eval_data.iterrows():
        answer, latency = run_with_latency(
            row["question"], row["context"], prompt_version
        )
        score = similarity_score(answer, row["ground_truth"])

        print(f"[{prompt_version}] {row['question']}  quality={score:.2f}  latency={latency:.2f}s")

        results.append({
            "prompt": prompt_version,
            "question": row["question"],
            "answer": answer,
            "quality": score,
            "latency": latency,
        })

    return pd.DataFrame(results)


def run_evaluation():
    eval_data.to_csv(DATASET_PATH, index=False)

    # Load the model into memory first, so the first timed call
    # doesn't include the load time and skew v1's latency
    local_model("Hello")

    comparison = pd.concat(
        [evaluate_prompt(version) for version in PROMPT_VERSIONS],
        ignore_index=True,
    )
    comparison.to_csv(RESULTS_PATH, index=False)

    print(f"\nModel: {MODEL}")
    print(comparison.groupby("prompt")[["quality", "latency"]].mean().round(3))
    print(f"\nSaved results to {RESULTS_PATH}")

    return comparison


if __name__ == "__main__":
    run_evaluation()
