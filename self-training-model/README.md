# self-training-model

An ML pipeline that watches a model's performance and retrains it when accuracy drops. It uses scikit-learn's `RandomForestClassifier` on synthetic data, so it runs locally in a few seconds with no downloads.

A model that works at launch can degrade over time. Customers change their behavior, preferences shift, and data collection changes. This is called **data drift** (the inputs change) or **concept drift** (the relationship between inputs and labels changes). Training is only the start: a production system also needs to notice when the model gets worse and respond.

## What it does

```
load current model ──► score on new production data ──► accuracy ≥ 0.80? ──yes──► keep model
                                                              │
                                                              no
                                                              ▼
                                         retrain on production data ──► validate
                                                                            │
                                    new accuracy > current accuracy? ──yes──► deploy (overwrite model)
                                                    │
                                                    no ──► keep current model
```

## Project structure

```
.
├── train.py          # Trains the initial model and saves it to model/current_model.joblib
├── pipeline.py       # Monitor → retrain → validate → deploy loop
├── requirements.txt  # scikit-learn, joblib
└── model/            # Created on first run (gitignored)
```

## How it works

| Step | What happens | Code |
|------|--------------|------|
| 1. Initial model | Train a random forest (150 trees) on an easy dataset (`class_sep=1.5`) and save it with `joblib` | `train.py`: `train_initial_model()` |
| 2. Production data | Generate a harder dataset (`class_sep=0.5`, different seed) to simulate drift | `generate_production_data()` |
| 3. Monitor | Score the current model on the production data | `evaluate_model()` |
| 4. Decide | Retrain if accuracy is below `RETRAIN_THRESHOLD = 0.80` | `should_retrain()` |
| 5. Retrain | Train a new random forest (200 trees) on 80% of the production data, validate on the other 20% | `retrain_model()` |
| 6. Deploy | Overwrite the saved model only if the new one scores higher than the current one | `deploy_model()`, `run_pipeline()` |

Retraining and deploying are separate decisions. Finishing a retrain doesn't mean the new model is better, so the pipeline checks before it replaces the production model.

In a real project, `generate_production_data()` would read from a database, data warehouse, object store or event stream. You need the true labels to measure accuracy. Without labels you can still watch for input drift, but you can't compute accuracy, precision or recall.

## Setup

Requires Python 3.10+. The repo-level virtualenv is `.playground/`.

```bash
# from the repo root
source .playground/bin/activate
pip install -r self-training-model/requirements.txt
```

## Run

```bash
cd self-training-model
python train.py      # train and save the initial model
python pipeline.py   # run the monitoring / retraining pipeline
```

Sample output:

```
$ python train.py
Initial model accuracy: 0.9550

$ python pipeline.py
Production accuracy: 0.5360
Performance dropped. Starting retraining...
New model validation accuracy: 0.8250
New model deployed successfully.
Retrained model is better. Deployment complete.
```

The model trained on easy data falls to about 54% (close to a coin flip) on the drifted data. The retrained model reaches about 82%, so the pipeline deploys it.

Run `python train.py` again to reset to the original model.

## Known limitations

This is a learning scaffold, not a production retraining system.

- **The comparison isn't like for like.** The new model's score comes from a 20% validation split. The current model's score comes from the full production set. Compare both models on the same held-out data.
- **Running the pipeline twice looks healthy for the wrong reason.** The deployed model was trained on 80% of the same production data, so a second run scores it at about 96% and skips retraining. Real pipelines evaluate on new data each time.
- **The retrained model forgets the original data.** It trains only on the 1,000 production rows. Combining historical and recent data, or weighting recent rows, is usually more robust.
- **Accuracy isn't always the right metric.** For imbalanced problems such as fraud detection, use precision, recall, F1 or a business KPI. You can also track calibration and latency.
- **Deploying overwrites the only copy.** There's no versioning or rollback. A model registry such as MLflow, or timestamped files, lets you roll back a bad deploy.
- **Nothing runs it on a schedule.** Use cron, Airflow or a CI job to run `pipeline.py` regularly, and log each run's metrics so you can see trends.
