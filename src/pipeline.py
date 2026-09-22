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
from src.ge_validation import validate_with_great_expectations
from src.logging_config import setup_logging
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"   # or any number of cores you have

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
    Includes:
    - Basic validation
    - Great Expectations validation
    - Logging
    - Latency
    - Error handling (reject on bad data)
    """
    start_time = time.time()

    try:
        logger.info("Received prediction request")
        logger.info(f"Input order: {order}")

        # 1. Basic required fields check
        validate_order(order)

        # 2. Great Expectations validation
        validate_with_great_expectations(order)

        # 3. Convert to DataFrame
        df = pd.DataFrame([order])

        # 4. City mapping
        city_mapping = load_city_mapping(city_mapping_path)
        df = apply_city_mapping(df, city_mapping)

        # 5. Feature engineering
        df = create_city_features(df)
        df = create_time_features(df)
        df = create_review_features(df)
        df = create_log_features(df)

        # 6. Preprocessing
        preprocessor = load_preprocessor(preprocessor_path)
        processed_df = apply_preprocessor(df, preprocessor)

        # 7. Prediction
        model = load_model(model_path)
        result = predict(model, processed_df)

        # Add metadata
        latency = round(time.time() - start_time, 4)
        result["model_version"] = model_version
        result["latency_seconds"] = latency

        logger.info(f"Prediction result: {result}")
        logger.info(f"Latency: {latency} seconds")

        return result

    except Exception as e:
        latency = round(time.time() - start_time, 4)
        logger.error(f"Prediction failed after {latency}s: {str(e)}")
        # We re-raise so the caller (API later) can return a proper error
        raise