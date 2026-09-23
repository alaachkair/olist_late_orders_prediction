import pytest

from src.ge_validation import validate_with_great_expectations
from src.validation import validate_order


def test_validate_order_success():
    order = {
        "order_purchase_timestamp": "2018-05-10 14:30:00",
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
        "min_review_score": 4.0,
    }
    # Should not raise any error
    validate_order(order)


def test_validate_order_missing_field():
    order = {
        "order_purchase_timestamp": "2018-05-10 14:30:00",
        "customer_state": "SP",
        # many fields missing
    }
    with pytest.raises(ValueError, match="Missing required fields"):
        validate_order(order)


def test_ge_validation_success():
    order = {
        "order_purchase_timestamp": "2018-05-10 14:30:00",
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
        "min_review_score": 4.0,
    }
    # Should not raise
    validate_with_great_expectations(order)


def test_ge_validation_invalid_state():
    order = {
        "order_purchase_timestamp": "2018-05-10 14:30:00",
        "order_estimated_delivery_date": "2018-05-25 00:00:00",
        "customer_zip_code": 13056,
        "customer_state": "XX",  # invalid state
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
        "min_review_score": 4.0,
    }
    with pytest.raises(ValueError, match="customer_state"):
        validate_with_great_expectations(order)


def test_ge_validation_out_of_range():
    order = {
        "order_purchase_timestamp": "2018-05-10 14:30:00",
        "order_estimated_delivery_date": "2018-05-25 00:00:00",
        "customer_zip_code": 13056,
        "customer_state": "SP",
        "customer_city": "campinas",
        "item_count": 100,  # out of range
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
        "min_review_score": 4.0,
    }
    with pytest.raises(ValueError, match="item_count"):
        validate_with_great_expectations(order)
