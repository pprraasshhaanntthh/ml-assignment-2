"""
Streamlit app for the ML Assignment 2 demo: loads 5 pre-trained classifiers
(Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) trained on
the UCI Adult / Census Income dataset, scores an uploaded CSV of test data
against the selected model, and displays evaluation metrics, a confusion
matrix, and a classification report. Models are loaded from disk and are
never retrained in this app.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "model"
DEFAULT_TEST_DATA = "test_data.csv"
TARGET_COLUMN = "income"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Income Classifier Demo", layout="wide")


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_saved_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


st.title("Census Income Classification — Model Demo")
st.markdown(
    """
This app loads **5 pre-trained classifiers** (Logistic Regression, Decision Tree,
KNN, Gaussian Naive Bayes, SVM) trained on the **UCI Adult / Census Income**
dataset. Upload a labelled test CSV (must include the `income` target column
with values `0`/`1`), pick a model, and see its evaluation metrics, confusion
matrix, and classification report — computed on your uploaded data, with
**no retraining**.
"""
)

st.sidebar.header("1. Upload test data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded uploaded file: {uploaded_file.name} ({len(data)} rows)")
else:
    data = pd.read_csv(DEFAULT_TEST_DATA)
    st.sidebar.info(f"No file uploaded — using bundled `{DEFAULT_TEST_DATA}` ({len(data)} rows)")

if TARGET_COLUMN not in data.columns:
    st.error(f"Uploaded CSV must contain the target column `{TARGET_COLUMN}` (0/1).")
    st.stop()

X = data.drop(columns=[TARGET_COLUMN])
y_true = data[TARGET_COLUMN].astype(int)

st.sidebar.header("2. Select model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
metrics = compute_metrics(y_true, y_pred, y_proba)

st.header(f"Results — {model_name}")

cols = st.columns(6)
labels = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
keys = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
for col, label, key in zip(cols, labels, keys):
    col.metric(label, f"{metrics[key]:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticklabels(["<=50K", ">50K"])
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_true, y_pred, target_names=["<=50K", ">50K"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4))

st.header("Compare all models on this data")
compare_rows = []
for name, filename in MODEL_FILES.items():
    m = load_model(filename)
    pred = m.predict(X)
    proba = m.predict_proba(X)[:, 1]
    row = {"Model": name}
    row.update(compute_metrics(y_true, pred, proba))
    compare_rows.append(row)

compare_df = pd.DataFrame(compare_rows).set_index("Model").round(4)
compare_df.columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
st.dataframe(compare_df, width="stretch")

with st.expander("Training-time metrics (from original train/test split)"):
    st.json(load_saved_metrics())
