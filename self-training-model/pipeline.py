import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from train import MODEL_PATH

# Retrain when production accuracy falls below this value
RETRAIN_THRESHOLD = 0.80


def generate_production_data():
    # Simulates drifted production data: the classes overlap more (class_sep=0.5).
    # In a real system this would read labeled data from a database, warehouse,
    # object store or event stream.
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        class_sep=0.5,
        random_state=100,
    )

    return X, y


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Production accuracy: {accuracy:.4f}")

    return accuracy


def should_retrain(accuracy):
    return accuracy < RETRAIN_THRESHOLD


def retrain_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    new_model = RandomForestClassifier(n_estimators=200, random_state=42)
    new_model.fit(X_train, y_train)

    predictions = new_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, predictions)

    print(f"New model validation accuracy: {validation_accuracy:.4f}")

    return new_model, validation_accuracy


def deploy_model(model):
    joblib.dump(model, MODEL_PATH)
    print("New model deployed successfully.")


def run_pipeline():
    current_model = joblib.load(MODEL_PATH)

    X_prod, y_prod = generate_production_data()

    current_accuracy = evaluate_model(current_model, X_prod, y_prod)

    if not should_retrain(current_accuracy):
        print("Performance is healthy. No retraining needed.")
        return

    print("Performance dropped. Starting retraining...")

    new_model, new_accuracy = retrain_model(X_prod, y_prod)

    # Retraining and deployment are separate decisions:
    # only replace the production model if the new one is better
    if new_accuracy > current_accuracy:
        deploy_model(new_model)
        print("Retrained model is better. Deployment complete.")
    else:
        print("New model did not improve performance.")
        print("Keeping the current production model.")


if __name__ == "__main__":
    run_pipeline()
