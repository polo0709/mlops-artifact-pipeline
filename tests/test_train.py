 
import json
import os
import pytest
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

def test_config_loading():
    config = load_config("config/config.json")
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

def test_model_training():
    config = load_config("config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy():
    config = load_config("config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.9  # Check if training worked
