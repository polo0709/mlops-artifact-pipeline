 
import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_config(path="config/config.json"):
    with open(path, "r") as file:
        return json.load(file)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target

    # Load config
    config = load_config()

    # Train model
    model = train_model(X, y, config)

    # Evaluate
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training accuracy: {acc:.4f}")

    # Save model
    joblib.dump(model, "model_train.pkl")
    print("Model saved as model_train.pkl")
