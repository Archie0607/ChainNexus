# ChainNexus AI

Streamlit dashboard for late delivery risk prediction using the pretrained ChainNexus MLP + Embedding model. The app reads the supplied model, preprocessing bundle, and exported metrics; it does not train or alter model artifacts.

## Run locally

Python 3.10–3.12 is recommended for compatibility with TensorFlow.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

streamlit run app.py
```

If starting from the original ZIP, extract `chainnexus_artifacts.zip` in this project root. The resulting `artifacts/` directory must include `preprocessing_bundle.joblib`, `models/chainnexus_mlp_embedding.keras`, and `metrics/metrics.json`.

## Pages

- **ภาพรวม** predicts uploaded orders, shows risk summaries, and lists orders to review in descending risk order. It uses an order ID from the upload where available.
- **ทำนายความเสี่ยง** supports a single order form and batch CSV upload with downloadable predictions.
- **ผลโมเดล** shows exported test metrics, baseline comparisons, tuning details, and feature importance. Upload an independent labeled CSV containing `Late_delivery_risk` to calculate metrics, a confusion matrix, ROC and precision–recall curves.
- **วิธีใช้** explains required inputs, risk levels, and evaluation data.

The saved binary alert threshold is read from `metrics.json` (0.34 in the supplied export). Three displayed risk bands are provisional presentation bands: low below the saved threshold, medium between it and the high-band cutoff, and high above that cutoff. They are not separately trained classes and need calibration against operational costs and an independent labeled dataset. Predicted risk is an estimate, not an observed delivery outcome.

## Input schema

The required columns are defined at runtime by `preprocessing_bundle.joblib`; the included `artifacts/batch_prediction_template.csv` is a starting template. Actual delivery outcomes and delivery timing fields are not prediction inputs.
