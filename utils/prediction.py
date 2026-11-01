import numpy as np
import pandas as pd
from .preprocessing import prepare_inputs

def display_high_threshold(threshold):
    return min(1.0, max(0.50, float(threshold) + 0.15))


def predict_dataframe(df, model, bundle, threshold, high_threshold=None):
    if high_threshold is None:
        high_threshold = display_high_threshold(threshold)
    inputs = prepare_inputs(df, bundle)
    probabilities = np.asarray(model.predict(inputs, verbose=0)).reshape(-1)
    if len(probabilities) != len(df) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid prediction values.")
    if not 0 <= float(threshold) < float(high_threshold) <= 1:
        raise ValueError("Risk-level thresholds must satisfy 0 <= saved threshold < high threshold <= 1.")
    result = df.copy()
    result["Late Risk Probability"] = probabilities
    result["Risk Classification"] = (probabilities >= float(threshold)).astype(int)
    result["Risk Level"] = np.select(
        [probabilities >= float(high_threshold), probabilities >= float(threshold)],
        ["สูง", "กลาง"],
        default="ต่ำ",
    )
    return result
