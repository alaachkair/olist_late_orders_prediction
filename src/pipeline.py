import time
from typing import Any, Dict

import pandas as pd
import yaml

from src.data import apply_city_mapping, load_city_mapping
from src.features import (
    create_city_features,
    create_log_features,
    create_review_features,
    create_time_features,
)
from src.ge_validation import validate_with_great_expectations
from src.logging_config import setup_logging
from src.predict import load_model_from_file, load_model_from_registry, predict
from src.preprocessing import apply_preprocessor, load_preprocessor
from src.validation import validate_order

logger = setup_logging()


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_inference(order: Dict[str, Any], use_registry: bool = True) -> Dict[str, Any]:
    """
    Full inference pipeline for one order.
    Can load the model either from MLflow Model Registry or from a local file.
    """
    start_time = time.time()
    config = load_config()

    try:
        logger.info("Received prediction request")
        logger.info(f"Input order: {order}")

        # 1. Basic validation
        validate_order(order)

        # 2. Great Expectations style validation
        validate_with_great_expectations(order)

        # 3. Convert to DataFrame
        df = pd.DataFrame([order])

        # 4. City mapping
        city_mapping = load_city_mapping(config["paths"]["city_mapping"])
        df = apply_city_mapping(df, city_mapping)

        # 5. Feature engineering
        df = create_city_features(df)
        df = create_time_features(df)
        df = create_review_features(df)
        df = create_log_features(df)

        # 6. Preprocessing
        preprocessor = load_preprocessor(config["paths"]["preprocessor"])
        processed_df = apply_preprocessor(df, preprocessor)

        # 7. Load model (from registry or local file)
        if use_registry:
            logger.info(
                f"Loading model from MLflow Model Registry ({config['model']['stage']})"
            )
            model = load_model_from_registry(
                model_name=config["model"]["registered_name"],
                stage=config["model"]["stage"],
                tracking_uri=config["mlflow"]["tracking_uri"],
            )
            model_version = config["model"]["stage"]
        else:
            logger.info("Loading model from local file")
            model = load_model_from_file(config["paths"]["model"])
            model_version = config["model"]["version"]

        # 8. Predict
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
        raise
