# ML Assignment 2 - training script
# Dataset: Breast Cancer Wisconsin (Diagnostic), UCI
# Target: diagnosis (1 = Malignant, 0 = Benign)
#
# Steps:
# 1. load data (569 rows, 30 features)
# 2. 80/20 stratified train test split
# 3. train 5 models with GridSearchCV (5 fold) on train data
# 4. find Accuracy, AUC, Precision, Recall, F1, MCC on test data
# 5. save models in model folder and test data as test_data.csv
#
# run: python train_models.py

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 2611
TARGET = "diagnosis"
MODEL_DIR = "model"


def load_data() -> pd.DataFrame:
    raw = load_breast_cancer()
    cols = [c.replace(" ", "_") for c in raw.feature_names]
    df = pd.DataFrame(raw.data, columns=cols)
    # in sklearn 0 = malignant and 1 = benign, changing it so malignant = 1
    df[TARGET] = 1 - raw.target
    return df


def build_candidates():
    # pipeline used so that scaler is saved along with the model
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [3, 4, 5, 6, 8, None],
             "clf__min_samples_leaf": [1, 3, 5, 10],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "Support Vector Machine": (
            Pipeline([("scaler", StandardScaler()),
                      # calibrated SVC to get probabilities for AUC
                      ("clf", CalibratedClassifierCV(SVC(random_state=SEED),
                                                     ensemble=False))]),
            {"clf__estimator__C": [0.1, 1, 10, 100],
             "clf__estimator__kernel": ["linear", "rbf"],
             "clf__estimator__gamma": ["scale", 0.01]},
        ),
    }


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features")
    print("Class count:", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    # save test data with target column, this file is used in the app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv("test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, best_params = {}, {}

    for name, (pipe, grid) in build_candidates().items():
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        joblib.dump(best, os.path.join(MODEL_DIR, file_name(name)))
        results[name] = {k: round(float(v), 4) for k, v in evaluate(best, X_test, y_test).items()}
        best_params[name] = {k.replace("clf__estimator__", "").replace("clf__", ""): v for k, v in search.best_params_.items()}
        print(f"{name:<24} best={best_params[name]}  test={results[name]}")

    meta = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic), UCI",
        "target": TARGET,
        "positive_class": "Malignant (1)",
        "features": list(X.columns),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": SEED,
        "model_files": {n: file_name(n) for n in results},
        "best_params": best_params,
        "test_metrics": results,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    table = pd.DataFrame(results).T
    print("\nTest set results:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
