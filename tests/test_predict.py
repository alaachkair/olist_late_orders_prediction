import pytest
import pandas as pd
from pathlib import Path
from src.predict import load_model_from_file, predict
from src.preprocessing import load_preprocessor, apply_preprocessor
from src.features import (
    create_city_features,
    create_time_features,
    create_review_features,
    create_log_features
)
from src.data import load_city_mapping, apply_city_mapping
import yaml


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def artifacts_exist() -> bool:
    """Check if the required model and data files exist."""
    config = load_config()
    model_path = Path(config["paths"]["model"])
    mapping_path = Path(config["paths"]["city_mapping"])
    preprocessor_path = Path(config["paths"]["preprocessor"])
    return model_path.exists() and mapping_path.exists() and preprocessor_path.exists()


@pytest.mark.skipif(not artifacts_exist(), reason="Model and data artifacts not available in CI")
def test_model_loads():
    config = load_config()
    model = load_model_from_file(config["paths"]["model"])
    assert model is not None


@pytest.mark.skipif(not artifacts_exist(), reason="Model and data artifacts not available in CI")
def test_predict_returns_expected_keys():
    config = load_config()

    order = {
        "order_purchase_timestamp": "2018-05-10 14:30:00",
        "order_approved_at": "2018-05-10 15:00:00",
        "order_estimated_delivery_date": "2018-05-25 00:00:00",
        "customer_zip_code": 13056,
        "customer_state": "SP",
        "customer_city": "campinas",
        "item_count": 1,
        "total_item_price": 89.90,
        "total_freight_value": 15.30,
        "unique_products": 1,
        "unique_sellers": 1,
        "payment_count": 1,
        "total_payment_value": 105.20,
        "max_installments": 1,
        "unique_payment_types": 1,
        "review_count": 1,
        "avg_review_score": 4.0,
        "min_review_score": 4.0
    }

    df = pd.DataFrame([order])

    city_mapping = load_city_mapping(config["paths"]["city_mapping"])
    df = apply_city_mapping(df, city_mapping)
    df = create_city_features(df)
    df = create_time_features(df)
    df = create_review_features(df)
    df = create_log_features(df)

    preprocessor = load_preprocessor(config["paths"]["preprocessor"])
    processed_df = apply_preprocessor(df, preprocessor)

    model = load_model_from_file(config["paths"]["model"])
    result = predict(model, processed_df)

    assert "prediction" in result
    assert "probability" in result
    assert "label" in result
    assert result["label"] in ["late", "on_time"]
    assert 0 <= result["probability"] <= 1