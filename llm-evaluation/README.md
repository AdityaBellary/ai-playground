# llm-evaluation

A minimal, fully local evaluation pipeline for a RAG (retrieval-augmented generation) question-answering app. No paid APIs: everything runs on open-source models from Hugging Face.

## What it does

An evaluation pipeline:

1. Runs your LLM app on a fixed dataset.
2. Compares the outputs with the expected results, or scores them with heuristics.
3. Produces metrics you can track over time.

`app.py` does this end to end:

```
question ──► retriever (MiniLM + FAISS) ──► context ──► generator (distilgpt2) ──► prediction
                                                                                      │
                     ground-truth answer ─────────────► metrics ◄─────────────────────┘
```

| Step | Component | Code |
|------|-----------|------|
| 1. Eval dataset | `question`, `context`, `answer` rows in a pandas DataFrame | `data` / `df` |
| 2. Retriever | `all-MiniLM-L6-v2` embeddings + `faiss.IndexFlatL2` | `retrieve()` |
| 3. Generator | `distilgpt2` via `transformers.pipeline` | `generate_answer()` |
| 4. RAG run | retrieve the top-1 context, then generate | `rag_pipeline()` |
| 5. Metrics | similarity, context score, groundedness | `similarity_score()`, `groundedness()` |

## Metrics

| Metric | How it's computed | What it tells you |
|--------|-------------------|-------------------|
| **Similarity** | Cosine similarity of the embeddings of `ground_truth` and `prediction` | Answer quality: how close the generated answer is in meaning to the expected one |
| **Context score** | Cosine similarity of `retrieved_context` and `ground_truth` | Retrieval quality: whether the retriever found context relevant to the answer |
| **Groundedness** | `1` if any word of the prediction appears in the retrieved context, else `0` | A rough check that the answer draws on the context rather than inventing content |

## Setup

Requires Python 3.10+. The repo-level virtualenv is `.playground/`.

```bash
# from the repo root
python3 -m venv .playground. # this is my virtual env.
source .playground/bin/activate
pip install transformers sentence-transformers faiss-cpu pandas scikit-learn torch
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) and `distilgpt2` (~350 MB) to `~/.cache/huggingface`.

## Run

```bash
cd llm-evaluation
python app.py
```

Sample output:

```
Average Similarity: 0.7689917
Average Context Score: 0.89787203
Groundedness Rate: 1.0
```

distilgpt2 samples its output, so the similarity changes between runs. The context score is deterministic.

To see the per-question results, inspect `results_df`, which has the columns `question`, `ground_truth`, `prediction`, `retrieved_context`, `similarity`, `context_score` and `groundedness`.

## Known limitations

This is a learning scaffold, not a production evaluation suite.

- **Two examples are far too few.** Grow the dataset first; it's the most valuable asset.
- **Retrieval is trivial.** The FAISS index holds only the ground-truth contexts, so top-1 retrieval nearly always succeeds. Add distractor documents to make the context score meaningful. A direct check like `retrieved_context == context` (hit rate) is stricter than embedding similarity.
- **Groundedness is weak.** Any shared word counts, even `the` or `a`, so nearly any output passes. Better options: the fraction of non-stopword tokens found in the context, an NLI/entailment model, or an LLM-as-judge.
- **The same embedding model is used for retrieval and for scoring**, which biases the context score upward.
- **distilgpt2 isn't instruction-tuned**, so answers are often repetitive or off-topic. An instruction model such as `Qwen/Qwen2.5-0.5B-Instruct` or `google/flan-t5-base` gives more realistic numbers.
- **No thresholds or tracking.** To use this as a regression gate, save the metrics per run (e.g. CSV/JSON with the commit SHA) and fail when a metric drops below a threshold.
