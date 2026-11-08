"""
Trains 5 classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes,
SVM) on the UCI Adult / Census Income dataset, evaluates them on a held-out test
split, and saves:
  - fitted sklearn Pipelines (preprocessing + classifier) to model/*.joblib
  - evaluation metrics to model/metrics.json
  - a raw held-out test sample (features + target) to test_data.csv, used by the
    Streamlit app so it can score without retraining.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MODEL_DIR = "model"
TEST_DATA_PATH = "test_data.csv"
TEST_SAMPLE_SIZE = 500
RAW_DATA_PATH = "adult.data"

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print(f"Loading Adult/Census Income dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
    )
    df = df.dropna(subset=["income"])

    y = df["income"].astype(str).str.strip().str.rstrip(".").apply(
        lambda v: 1 if v.startswith(">50K") else 0
    )
    X = df.drop(columns=["income"])

    return X, y


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "svm": SVC(probability=True, random_state=RANDOM_STATE),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
    }


def main():
    X, y = load_data()
    print(f"Dataset shape: {X.shape}, positive class rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)
    models = get_models()
    all_metrics = {}

    for name, estimator in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        all_metrics[name] = metrics
        print(f"  {name}: {metrics}")

        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {MODEL_DIR}/metrics.json")

    # Export a raw test sample (features + true target) for the Streamlit app / grading
    test_export = X_test.copy()
    test_export["income"] = y_test.values
    test_sample = test_export.sample(
        n=min(TEST_SAMPLE_SIZE, len(test_export)), random_state=RANDOM_STATE
    )
    test_sample.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved {len(test_sample)} rows of raw test data to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
