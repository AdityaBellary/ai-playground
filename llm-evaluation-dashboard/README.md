# llm-evaluation-dashboard

A local dashboard for comparing prompt versions of an LLM app on response quality and speed. It uses Ollama, pandas and Streamlit, and runs entirely on your machine, with no paid APIs.

Asking a model a few questions and checking that the answers look right works for a demo. It stops being reliable once you change prompts, models, retrieval logic or code. A fixed evaluation dataset that you rerun after every change tells you whether a change actually helped.

## What it does

```
evaluation dataset ──► LLM app (prompt v1 / v2) ──► local model (Ollama)
                                                            │
                         ground truth ──► quality + latency ◄┘
                                                │
                                   evaluation_results.csv ──► Streamlit dashboard
```

The LLM app (`llm_application()`) is kept separate from the evaluation code, so you can change one without touching the other.

## Project structure

```
.
├── evaluate.py       # Dataset, LLM app, metrics and evaluation loop; writes the CSVs
├── app.py            # Streamlit dashboard that reads the results CSV
└── requirements.txt  # pandas, ollama, streamlit
```

Running the evaluation creates `evaluation_dataset.csv` and `evaluation_results.csv`. Both are gitignored.

## How it works

| Step | What happens | Code |
|------|--------------|------|
| 1. Dataset | 4 rows of `question`, `context` and `ground_truth` | `eval_data` |
| 2. Model | Sends a prompt to a local Ollama model | `local_model()` |
| 3. LLM app | Builds the prompt from one of two versions and calls the model | `llm_application()` |
| 4. Latency | Times each call with `time.perf_counter()` | `run_with_latency()` |
| 5. Quality | `difflib.SequenceMatcher` ratio between the answer and the ground truth (0 to 1) | `similarity_score()` |
| 6. Evaluate | Runs every dataset row through one prompt version | `evaluate_prompt()` |
| 7. Compare | Runs both versions and saves the combined results | `run_evaluation()` |
| 8. Dashboard | Average quality and latency, a bar chart for each by prompt, and the full results table | `app.py` |

### The two prompts

| Version | Instruction |
|---------|-------------|
| `v1` | Answer the question using the provided context. |
| `v2` | Answer using **only** the provided context. If the context doesn't support the answer, say you don't have enough information. |

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/download). The repo-level virtualenv is `.playground/`.

```bash
# from the repo root
source .playground/bin/activate
pip install -r llm-evaluation-dashboard/requirements.txt

ollama pull llama3
```

The model defaults to `llama3`. To use another model you've pulled, set `OLLAMA_MODEL`:

```bash
export OLLAMA_MODEL=mistral
```

## Run

Make sure Ollama is running (the desktop app, or `ollama serve`).

```bash
cd llm-evaluation-dashboard
python evaluate.py        # run the evaluation and save the results
streamlit run app.py      # open the dashboard at http://localhost:8501
```

You can also click **Run evaluation** in the dashboard to rerun it.

Sample output from `mistral` on an M-series Mac:

```
[v1] What is overfitting?  quality=0.30  latency=1.63s
...
[v2] What is an AI agent?  quality=0.74  latency=0.47s

Model: mistral
        quality  latency
prompt
v1        0.421    1.034
v2        0.500    0.782
```

Your numbers will be different: model output varies between runs, and latency depends on your hardware.

## Known limitations

This is a learning scaffold, not a production evaluation suite.

- **Four examples are far too few.** One unusual answer moves the average a lot. Grow the dataset first.
- **`SequenceMatcher` measures character overlap, not correctness.** A correct answer in different words scores low, and a long, verbose answer scores low even when it's right. Better options: embedding similarity, an LLM-as-judge with a local model, task-specific checks or human review.
- **v2 isn't tested on what makes it different.** Every question is answerable from its context, so the "say you don't have enough information" instruction never applies. Add questions whose context doesn't contain the answer, and check that v2 refuses.
- **Each prompt is run once per question.** Output varies between runs, so small differences between v1 and v2 may be noise. Run each question several times and compare the averages.
- **Results are overwritten on every run.** To track changes over time, save each run with a timestamp, model name and commit SHA.
