# ML Assignment 2 - Streamlit app
# Upload the test csv, select a model and check the results.
# Models are loaded from the model folder. No training happens here.

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_CSV = BASE / "test_data.csv"
LABELS = {0: "Benign", 1: "Malignant"}

st.set_page_config(page_title="Breast Cancer Classification",
                   layout="wide")


# load files
@st.cache_resource
def load_meta() -> dict:
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


def clean_target(series: pd.Series) -> pd.Series:
    # target can be 0/1 or M/B
    if not pd.api.types.is_numeric_dtype(series):
        mapped = series.astype(str).str.strip().str.upper().str[0].map({"M": 1, "B": 0})
        if mapped.isna().any():
            raise ValueError("Target column has values other than M/B or Malignant/Benign.")
        return mapped.astype(int)
    return series.astype(int)


def score(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob) if y.nunique() == 2 else float("nan")
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

meta = load_meta()
FEATURES = meta["features"]
TARGET = meta["target"]
MODEL_FILES = meta["model_files"]

# sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    st.caption(f"File should have all {len(FEATURES)} feature columns and the "
               f"{TARGET} column (1 or M for Malignant, 0 or B for Benign).")
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv",
                           SAMPLE_CSV.read_bytes(), "test_data.csv", "text/csv")
    use_sample = st.checkbox("Use test_data.csv from repo", value=uploaded is None)

    st.divider()
    st.markdown("**Best parameters (GridSearchCV)**")
    st.json(meta["best_params"].get(model_name, {}))

# main page
st.title("Breast Cancer Classification")
st.markdown(
    "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI. 30 features. "
    "The app predicts if a tumour is Malignant or Benign using 5 saved models "
    "and shows the metrics on the uploaded test data."
)

# read data
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = f"Uploaded file: {uploaded.name}"
elif use_sample and SAMPLE_CSV.exists():
    data = pd.read_csv(SAMPLE_CSV)
    source = "Using test_data.csv from repo"
else:
    st.info("Please upload the test CSV from the sidebar.")
    st.stop()

data.columns = [c.strip().replace(" ", "_") for c in data.columns]
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=FEATURES + [TARGET])
try:
    y_true = clean_target(data[TARGET])
except ValueError as err:
    st.error(str(err))
    st.stop()
X = data[FEATURES]

c1, c2, c3 = st.columns(3)
c1.metric("Total rows", len(data))
c2.metric("Malignant", int(y_true.sum()))
c3.metric("Benign", int((y_true == 0).sum()))
st.caption(source)
with st.expander("View data"):
    st.dataframe(data.head(20), use_container_width=True)

tab_model, tab_compare = st.tabs([f"{model_name} results", "Model comparison"])

# selected model
with tab_model:
    model = load_model(MODEL_FILES[model_name])
    res = score(model, X, y_true)

    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, key in zip(cols, METRIC_KEYS):
        col.metric(key, f"{res[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, res["y_pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="Blues")
        names = [LABELS[0], LABELS[1]]
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"False negatives: {fn}, False positives: {fp}")

    with right:
        st.subheader("ROC curve")
        if y_true.nunique() == 2:
            fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
            fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
            ax2.plot(fpr, tpr, lw=2, label=f"AUC = {res['AUC']:.3f}")
            ax2.plot([0, 1], [0, 1], ls="--", color="grey")
            ax2.set_xlabel("False positive rate")
            ax2.set_ylabel("True positive rate")
            ax2.legend(loc="lower right")
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.warning("ROC needs both classes in the uploaded data.")

    st.subheader("Classification report")
    report = classification_report(y_true, res["y_pred"], labels=[0, 1],
                                   target_names=names, output_dict=True,
                                   zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    out = data.copy()
    out["predicted"] = [LABELS[p] for p in res["y_pred"]]
    out["prob_malignant"] = np.round(res["y_prob"], 4)
    st.download_button("Download predictions", out.to_csv(index=False),
                       f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       "text/csv")

# all models
with tab_compare:
    st.subheader("Metrics for all 5 models")
    rows = {}
    for name, fname in MODEL_FILES.items():
        r = score(load_model(fname), X, y_true)
        rows[name] = {k: r[k] for k in METRIC_KEYS}
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
                 use_container_width=True)

    metric_pick = st.radio("Select metric for chart", METRIC_KEYS, index=5, horizontal=True)
    st.bar_chart(table[metric_pick])
    best = table["MCC"].idxmax()
    st.success(f"Best model based on MCC: {best} "
               f"(MCC {table.loc[best, 'MCC']:.4f}, F1 {table.loc[best, 'F1']:.4f})")

st.caption("ML Assignment 2, BITS Pilani WILP")
