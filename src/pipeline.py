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
from src.validation import validate_order


def run_inference(
    order: Dict[str, Any],
    city_mapping_path: str,
    preprocessor_path: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Full inference pipeline for one order.

    Steps:
    1. Validate the input
    2. Convert order dict to DataFrame
    3. Apply city mapping
    4. Create all features (city, time, review, log)
    5. Apply the fitted preprocessor
    6. Make prediction with the trained model
    """

    # 1. Validate input
    validate_order(order)

    # 2. Convert single order to DataFrame
    df = pd.DataFrame([order])

    # 3. Load and apply city mapping
    city_mapping = load_city_mapping(city_mapping_path)
    df = apply_city_mapping(df, city_mapping)

    # 4. Create features (exactly the same order as the notebook)
    df = create_city_features(df)
    df = create_time_features(df)
    df = create_review_features(df)
    df = create_log_features(df)

    # 5. Load and apply the fitted preprocessor
    preprocessor = load_preprocessor(preprocessor_path)
    processed_df = apply_preprocessor(df, preprocessor)

    # 6. Load model and predict
    model = load_model(model_path)
    result = predict(model, processed_df)

    return result