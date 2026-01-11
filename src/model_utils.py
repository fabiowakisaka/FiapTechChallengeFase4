import joblib
import os

def save_artifacts(model, threshold, features, base_path="../modelo"):
    os.makedirs(base_path, exist_ok=True)

    joblib.dump(model, f"{base_path}/xgb.joblib")
    joblib.dump(threshold, f"{base_path}/threshold.joblib")
    joblib.dump(features, f"{base_path}/features.joblib")
