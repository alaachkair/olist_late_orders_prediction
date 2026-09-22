import pandas as pd
import pytest
from src.features import (
    create_city_features,
    create_time_features,
    create_review_features,
    create_log_features
)


def test_create_city_features():
    df = pd.DataFrame({
        "customer_city": ["sao paulo", "campinas", "unknown city"]
    })

    result = create_city_features(df)

    assert "is_large_city" in result.columns
    assert "is_capital" in result.columns
    assert "customer_city" not in result.columns
    assert result.loc[0, "is_large_city"] == 1
    assert result.loc[1, "is_large_city"] == 1
    assert result.loc[2, "is_large_city"] == 0


def test_create_time_features():
    df = pd.DataFrame({
        "order_purchase_timestamp": ["2018-05-10 14:30:00"],
        "order_estimated_delivery_date": ["2018-05-25 00:00:00"],
        "order_approved_at": ["2018-05-10 15:00:00"]
    })

    result = create_time_features(df)

    assert "purchase_month" in result.columns
    assert "purchase_dayofweek" in result.columns
    assert "estimated_delivery_days" in result.columns
    assert "order_purchase_timestamp" not in result.columns
    assert result.loc[0, "purchase_month"] == 5


def test_create_review_features():
    df = pd.DataFrame({
        "review_count": [1, None],
        "avg_review_score": [4.5, None],
        "min_review_score": [4.0, None]
    })

    result = create_review_features(df)

    assert "has_review" in result.columns
    assert result.loc[0, "has_review"] == 1
    assert result.loc[1, "has_review"] == 0
    assert result.loc[1, "review_count"] == 0


def test_create_log_features():
    df = pd.DataFrame({
        "item_count": [1],
        "total_item_price": [100.0],
        "total_freight_value": [15.0],
        "unique_products": [1],
        "unique_sellers": [1],
        "payment_count": [1],
        "total_payment_value": [115.0]
    })

    result = create_log_features(df)

    assert "log_item_count" in result.columns
    assert "log_total_item_price" in result.columns
    assert result.loc[0, "log_item_count"] > 0