import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_model(model_path: str):
    """
    Load the trained model.
    Never train again — only load the saved object.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(path)
    return model


def predict(model, processed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Make a prediction on the already preprocessed data.
    
    Returns:
        - prediction: 0 (on time) or 1 (late)
        - probability: probability of being late
        - label: "late" or "on_time"
    """
    # Get probability of the positive class (late = 1)
    probability = model.predict_proba(processed_df)[0, 1]

    # Get the class prediction
    prediction = model.predict(processed_df)[0]

    label = "late" if prediction == 1 else "on_time"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "label": label
    }