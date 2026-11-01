from pathlib import Path
import json
import joblib
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

@st.cache_resource(show_spinner="Loading trained ChainNexus model…")
def load_model():
    path = ARTIFACTS / "models" / "chainnexus_mlp_embedding.keras"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_bundle():
    path = ARTIFACTS / "preprocessing_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing bundle not found: {path}")
    return joblib.load(path)

@st.cache_data
def load_metrics():
    path = ARTIFACTS / "metrics" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
