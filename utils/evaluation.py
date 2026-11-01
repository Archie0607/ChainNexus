import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .prediction import predict_dataframe

TARGET_COLUMN = "Late_delivery_risk"


def evaluate_labeled_dataframe(df, model, bundle, threshold):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"ไฟล์ประเมินต้องมีคอลัมน์ผลจริงชื่อ {TARGET_COLUMN}")
    truth = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if truth.isna().any() or not set(truth.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} ต้องมีเฉพาะค่า 0 หรือ 1 และห้ามเว้นว่าง")

    results = predict_dataframe(df, model, bundle, threshold)
    y_true = truth.astype("int32").to_numpy()
    probabilities = results["Late Risk Probability"].to_numpy()
    y_pred = results["Risk Classification"].to_numpy()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "roc_auc": None,
        "average_precision": None,
        "roc_curve": None,
        "pr_curve": None,
    }
    if len(np.unique(y_true)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
        metrics["average_precision"] = average_precision_score(y_true, probabilities)
        metrics["roc_curve"] = (fpr, tpr, roc_thresholds)
        metrics["pr_curve"] = (precision, recall, pr_thresholds)
    return results, metrics
