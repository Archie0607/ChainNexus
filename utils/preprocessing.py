import numpy as np
import pandas as pd

def validate_columns(df, features):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("CSV contains no data rows.")

def prepare_inputs(df, bundle):
    features = bundle["features"]
    categorical = bundle["categorical_features"]
    numerical = bundle["numerical_features"]
    validate_columns(df, features)
    source = df.loc[:, features].copy()
    output = {}
    maps = bundle["category_maps"]
    for col in categorical:
        raw = source[col]
        unknown = sorted(set(raw.dropna().astype(str)) - set(maps[col]))
        if unknown:
            raise ValueError(f"Unknown value(s) in {col}: {', '.join(unknown[:8])}" + (" …" if len(unknown) > 8 else ""))
        # Bundle mappings use IDs starting at 1; zero is reserved for the embedding OOV index.
        encoded = raw.map(maps[col]).fillna(0).astype("int32").to_numpy().reshape(-1, 1)
        output[f"cat_{col.replace(' ', '_')}"] = encoded
    numbers = source[numerical].apply(pd.to_numeric, errors="coerce")
    invalid = source[numerical].notna() & numbers.isna()
    invalid_columns = invalid.columns[invalid.any()].tolist()
    if invalid_columns:
        raise ValueError("Numeric columns contain non-numeric values: " + ", ".join(invalid_columns))
    medians = np.asarray(bundle["num_imputer"].statistics_, dtype="float32")
    if numbers.shape[1] != len(medians):
        raise ValueError("Numeric feature layout does not match the saved imputer.")
    numbers = numbers.fillna(pd.Series(medians, index=numerical)).astype("float32")
    if not np.isfinite(numbers.to_numpy()).all():
        raise ValueError("Numeric features contain non-finite values.")
    scaler = bundle.get("num_scaler")
    if scaler is None:
        raise ValueError("Saved numeric scaler is missing from the preprocessing bundle.")
    output["numerical_features"] = scaler.transform(numbers.to_numpy(dtype="float32")).astype("float32")
    return output
