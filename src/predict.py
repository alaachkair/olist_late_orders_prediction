import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_model_from_file(model_path: str):
    """Load model from a local .joblib file (fallback)."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(path)


def load_model_from_registry(
    model_name: str,
    stage: str = "Staging",
    tracking_uri: str = "sqlite:///mlflow.db"
):
    """
    Load the model from MLflow Model Registry.
    """
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict(model, processed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Make a prediction on the already preprocessed data.
    """
    probability = model.predict_proba(processed_df)[0, 1]
    prediction = model.predict(processed_df)[0]
    label = "late" if prediction == 1 else "on_time"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "label": label
    }