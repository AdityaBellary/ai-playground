# dockerize-ai-agent

A small command-line AI agent that runs inside Docker. It uses Hugging Face Transformers with [`distilgpt2`](https://huggingface.co/distilbert/distilgpt2), a smaller and faster version of GPT-2.

Writing the model code is only the first step. An agent that runs on your laptop can still break on a teammate's machine or a cloud server, usually because of a different Python version or missing libraries. Docker packages the code, dependencies and runtime into one image, so the agent runs the same way everywhere.

## Project structure

```
.
├── app.py            # The agent: a chat loop around a text-generation pipeline
├── requirements.txt  # Python dependencies (transformers, torch)
└── Dockerfile        # How to build the container image
```

## How it works

### `app.py`: the agent

```python
generator = pipeline("text-generation", model="distilgpt2")
```

`run_agent()` runs a simple loop:

1. Read a line of user input.
2. Stop if the input is `exit`.
3. Send the input to `distilgpt2` and print the generated text.

### `requirements.txt`: dependencies

| Package        | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `transformers` | Model interface: pipelines, tokenizers, weights |
| `torch`        | PyTorch, the tensor engine that runs the model  |

### `Dockerfile`: the build steps

| Instruction                                    | What it does                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| `FROM python:3.10-slim`                        | Starts from the official lightweight Python 3.10 image            |
| `WORKDIR /app`                                 | Creates `/app` in the container and makes it the working directory |
| `COPY . /app`                                  | Copies the project files into the container                       |
| `RUN pip install --no-cache-dir -r requirements.txt` | Installs dependencies without saving pip's cache, which keeps the image smaller |
| `CMD ["python", "app.py"]`                     | Starts the agent when the container runs                          |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running

## Quick start

### 1. Build the image

```bash
docker build -t ai-agent .
```

The `.` tells Docker to use the current directory as the build context. The first build takes a few minutes because it downloads the Python base image and installs PyTorch.

### 2. Run the container

```bash
docker run -it ai-agent
```

The agent reads from your terminal, so you need the `-it` flags. `-i` keeps STDIN open and `-t` attaches a terminal. Without them, `input()` fails right away.

On the first run, the pipeline downloads the `distilgpt2` weights (about 350 MB). After that, you get the prompt:

```
AI Agent is running. Type 'exit' to stop.

You: The future of artificial intelligence is
Agent: The future of artificial intelligence is ...

You: exit
```

> `distilgpt2` is a small model that continues text. It isn't tuned for chat, so expect replies that are creative and sometimes off topic.

### 3. (Optional) Keep downloaded weights between runs

Each new container downloads the model again. To keep the weights, store the Hugging Face cache in a named volume:

```bash
docker run -it -v hf-cache:/root/.cache/huggingface ai-agent
```

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Troubleshooting

| Problem                                   | Fix                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `EOFError` when the container starts      | Run with `-it`: `docker run -it ai-agent`                           |
| Image is very large (several GB)          | The default `torch` package includes GPU (CUDA) libraries. For a CPU-only image, install from PyTorch's CPU index: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Every run downloads the model again       | Mount a volume for the Hugging Face cache (see step 3)              |
| Replies are cut short                     | `max_length=50` counts the prompt too. Use `max_new_tokens=50` to limit only the reply |

## Next steps

- Pin dependency versions in `requirements.txt` so builds are reproducible.
- Add a `.dockerignore` so files like `.git` and `.venv` stay out of the image.
- Download the model during `docker build`, so containers start without fetching it.
- Push the image to a registry and deploy it to AWS, Google Cloud or Azure.
