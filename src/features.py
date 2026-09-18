import pandas as pd
import numpy as np


# List of large cities (from your notebook)
LARGE_CITIES = [
    "sao paulo",
    "rio d janeiro",
    "salvador",
    "belo horizonte",
    "fortaleza",
    "brasilia",
    "curitiba",
    "recife",
    "porto alegre",
    "manaus",
    "belem",
    "guarulhos",
    "goiania",
    "campinas",
    "sao goncalo",
    "nova iguacu",
    "sao luis",
    "maceio",
    "duque d caxias",
    "sao bernardo d campo"
]

CAPITAL_CITY = "brasilia"

LOG_COLS = [
    "item_count",
    "total_item_price",
    "total_freight_value",
    "unique_products",
    "unique_sellers",
    "payment_count",
    "total_payment_value"
]


def create_city_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create is_large_city and is_capital features."""
    df = df.copy()

    df["is_large_city"] = df["customer_city"].isin(LARGE_CITIES).astype(int)
    df["is_capital"] = (df["customer_city"] == CAPITAL_CITY).astype(int)

    # Remove original high-cardinality city
    df = df.drop(columns=["customer_city"], errors="ignore")

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from purchase and estimated delivery dates."""
    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_weekofyear"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
    df["purchase_quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Remove raw datetime columns
    df = df.drop(
        columns=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date"
        ],
        errors="ignore"
    )

    return df


def create_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create has_review indicator and fill missing review values."""
    df = df.copy()

    df["has_review"] = df["review_count"].notna().astype(int)

    df["review_count"] = df["review_count"].fillna(0)
    df["avg_review_score"] = df["avg_review_score"].fillna(0)
    df["min_review_score"] = df["min_review_score"].fillna(0)

    return df


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log1p versions of selected numeric columns."""
    df = df.copy()

    for col in LOG_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df