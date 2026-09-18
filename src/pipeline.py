import pandas as pd
from typing import Dict, Any

from src.data import load_city_mapping, apply_city_mapping
from src.features import (
    create_city_features,
    create_time_features,
    create_review_features,
    create_log_features
)
from src.preprocessing import load_preprocessor, apply_preprocessor
from src.predict import load_model, predict


def run_inference(
    order: Dict[str, Any],
    city_mapping_path: str,
    preprocessor_path: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Full inference pipeline for one order.

    Steps:
    1. Convert order dict to DataFrame
    2. Apply city mapping
    3. Create all features (city, time, review, log)
    4. Apply the fitted preprocessor
    5. Make prediction with the trained model
    """

    # 1. Convert single order to DataFrame
    df = pd.DataFrame([order])

    # 2. Load and apply city mapping
    city_mapping = load_city_mapping(city_mapping_path)
    df = apply_city_mapping(df, city_mapping)

    # 3. Create features (exactly the same order as the notebook)
    df = create_city_features(df)
    df = create_time_features(df)
    df = create_review_features(df)
    df = create_log_features(df)

    # 4. Load and apply the fitted preprocessor
    preprocessor = load_preprocessor(preprocessor_path)
    processed_df = apply_preprocessor(df, preprocessor)

    # 5. Load model and predict
    model = load_model(model_path)
    result = predict(model, processed_df)

    return result