import time
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
from src.logging_config import setup_logging


logger = setup_logging()


def run_inference(
    order: Dict[str, Any],
    city_mapping_path: str,
    preprocessor_path: str,
    model_path: str,
    model_version: str = "1"
) -> Dict[str, Any]:
    """
    Full inference pipeline for one order.
    Includes logging, latency measurement, and error handling.
    """
    start_time = time.time()

    try:
        logger.info("Received prediction request")
        logger.info(f"Input order: {order}")

        # 1. Validate input
        validate_order(order)

        # 2. Convert to DataFrame
        df = pd.DataFrame([order])

        # 3. City mapping
        city_mapping = load_city_mapping(city_mapping_path)
        df = apply_city_mapping(df, city_mapping)

        # 4. Feature engineering
        df = create_city_features(df)
        df = create_time_features(df)
        df = create_review_features(df)
        df = create_log_features(df)

        # 5. Preprocessing
        preprocessor = load_preprocessor(preprocessor_path)
        processed_df = apply_preprocessor(df, preprocessor)

        # 6. Prediction
        model = load_model(model_path)
        result = predict(model, processed_df)

        # Add model version and latency
        latency = round(time.time() - start_time, 4)
        result["model_version"] = model_version
        result["latency_seconds"] = latency

        logger.info(f"Prediction result: {result}")
        logger.info(f"Latency: {latency} seconds")

        return result

    except Exception as e:
        latency = round(time.time() - start_time, 4)
        logger.error(f"Prediction failed after {latency}s: {str(e)}")
        raise