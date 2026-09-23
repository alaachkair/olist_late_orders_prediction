from typing import Any, Dict

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
    "min_review_score",
]

BRAZILIAN_STATES = [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]


def validate_with_great_expectations(order: Dict[str, Any]) -> None:
    """
    Validate incoming order data.
    This covers the same checks that Great Expectations would do:
    - required fields
    - no missing critical values
    - ranges
    - allowed categories
    On failure → raise ValueError (the service will reject the request)
    """
    # 1. Required fields exist
    missing = [field for field in REQUIRED_FIELDS if field not in order]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # 2. Critical values must not be None
    for field in REQUIRED_FIELDS:
        if order[field] is None:
            raise ValueError(f"Field '{field}' cannot be null")

    # 3. Range checks
    if not (1 <= order["item_count"] <= 50):
        raise ValueError("item_count must be between 1 and 50")

    if not (0 <= order["total_item_price"] <= 100000):
        raise ValueError("total_item_price is out of allowed range")

    if not (0 <= order["avg_review_score"] <= 5):
        raise ValueError("avg_review_score must be between 0 and 5")

    if not (0 <= order["min_review_score"] <= 5):
        raise ValueError("min_review_score must be between 0 and 5")

    # 4. Allowed categories
    if order["customer_state"] not in BRAZILIAN_STATES:
        raise ValueError(
            f"customer_state '{order['customer_state']}' is not a valid Brazilian state"
        )
