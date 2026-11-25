# llm-cicd-pipeline

A CI/CD pipeline for a small LLM application using GitHub Actions, Docker, pytest and a local Ollama model, with no paid APIs.

Keeping an LLM app reliable after every change to code, prompts, models or dependencies is harder than building it. Model outputs are probabilistic, so unit tests alone can't tell you whether the app still behaves correctly. This pipeline adds task-specific evals to the usual tests, and only builds the image when both pass.

## What it does

```
push / PR ──► unit tests ──► integration tests + LLM evals ──► Docker build (tagged with commit SHA)
              (mocked LLM)    (real model in Ollama, gated            
                               on a 0.80 eval score)
```

Each stage runs only if the one before it passes.

## Project structure

```
.
├── app/
│   ├── __init__.py
│   ├── __main__.py          # CLI: python -m app ["prompt"]
│   └── llm_service.py       # generate_response(): one call to Ollama
├── tests/
│   ├── test_unit.py         # Mocks Ollama; checks the request and response handling
│   └── test_integration.py  # Calls the real model; skipped if it isn't available
├── evals/
│   ├── test_cases.json      # Prompts and the keyword each answer must contain
│   └── run_evals.py         # Scores the model and exits non-zero below the threshold
├── Dockerfile
├── pytest.ini
└── requirements.txt         # ollama, pytest
```

The workflow is at [`.github/workflows/llm-cicd-pipeline.yml`](../.github/workflows/llm-cicd-pipeline.yml) in the repo root, because GitHub only runs workflows from there.

## How it works

### Testing layers

| Layer | What it checks | Needs a model? | File |
|-------|----------------|----------------|------|
| Unit tests | `generate_response()` sends the prompt as a user message and returns the reply text | No, Ollama is mocked | `tests/test_unit.py` |
| Integration test | The real model returns non-empty text | Yes | `tests/test_integration.py` |
| LLM evals | The answers contain the expected keyword in at least 80% of cases | Yes | `evals/run_evals.py` |

Unit tests catch ordinary bugs such as broken imports or a wrong response field. They can't tell you whether the model gives useful answers, which is what the evals are for.

### GitHub Actions jobs

| Job | Steps | Runs after |
|-----|-------|------------|
| `test` | Install dependencies, run unit tests | — |
| `evaluate` | Install Ollama, pull `qwen3:0.6b`, run integration tests and evals | `test` |
| `build` | `docker build -t llm-app:<commit SHA> .` | `evaluate` |

The workflow runs on pushes and pull requests to `main` that change `llm-cicd-pipeline/` or the workflow file. CI uses `qwen3:0.6b` (about 520 MB) so the model download stays quick. The tag with the commit SHA lets you trace any image back to the code it was built from.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_NAME` | `qwen3` | Ollama model the app calls |
| `EVAL_THRESHOLD` | `0.8` | Minimum eval score for the pipeline to pass |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server (read by the `ollama` client) |

Never put API keys or passwords in the repo. If you switch to a hosted model, store the key as a GitHub secret and read it in the workflow with `API_KEY: ${{ secrets.API_KEY }}`. For real deployments, set up `staging` and `production` [GitHub environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment), which can limit deploys to certain branches and require manual approval.

## Setup

Requires Python 3.10+, [Ollama](https://ollama.com/download) and Docker. The repo-level virtualenv is `.playground/`.

```bash
# from the repo root
source .playground/bin/activate
pip install -r llm-cicd-pipeline/requirements.txt

ollama pull qwen3        # or qwen3:0.6b, which is smaller and faster
```

## Run locally

```bash
cd llm-cicd-pipeline

pytest                        # unit + integration tests
python evals/run_evals.py     # LLM evals
python -m app                 # chat with the app
```

To use a different model, set `MODEL_NAME`, for example `MODEL_NAME=qwen3:0.6b pytest`.

Sample eval output:

```
[PASS] Explain machine learning in one sentence.  (expects 'data')
[PASS] What is Python?  (expects 'programming')
[PASS] What does CPU stand for?  (expects 'processing')
[PASS] What is the capital of France? Answer in one word.  (expects 'paris')
[PASS] Name the language used to style web pages, alongside HTML.  (expects 'css')

Model: qwen3:0.6b
Eval score: 1.00 (threshold 0.80)
```

### Docker

```bash
docker build -t llm-app .

# The container talks to Ollama on your machine
docker run -it --rm \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e MODEL_NAME=qwen3:0.6b \
  llm-app
```

The image contains only the app. The model runs in Ollama, outside the container.

## Known limitations

This is a learning scaffold, not a production pipeline.

- **Keyword evals are brittle.** "Explain machine learning" fails if the answer says "examples" instead of "data". Real evals check factual correctness, structured output validity, retrieval quality, tool selection or task completion, often with an LLM-as-judge.
- **Evals can be flaky.** Outputs vary between runs: in testing, `qwen3:0.6b` scored 0.80 once and 1.00 twice. With five cases, one miss puts the score right at the threshold. Add more cases, run each several times, or set a fixed `seed` and `temperature` in the Ollama options.
- **CI evaluates a smaller model than the default.** CI uses `qwen3:0.6b` and local runs default to `qwen3`, so a pass in CI doesn't guarantee the same results with the larger model.
- **The model is downloaded on every CI run.** Caching `~/.ollama/models` with `actions/cache` would speed this up.
- **Nothing is deployed.** The `build` job builds the image but doesn't push it. Next steps: push to a registry such as GHCR, then add a `deploy` job that uses a `production` environment with required reviewers.
