from typing import Dict, Any, List


REQUIRED_FIELDS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_zip_code",
    "customer_state",
    "customer_city",
    "item_count",
    "total_item_price",
    "total_freight_value",
    "unique_products",
    "unique_sellers",
    "payment_count",
    "total_payment_value",
    "max_installments",
    "unique_payment_types",
    "review_count",
    "avg_review_score",
    "min_review_score"
]


def validate_order(order: Dict[str, Any]) -> None:
    """
    Check that the incoming order has the required fields.
    Raises ValueError if something is missing.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in order]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")