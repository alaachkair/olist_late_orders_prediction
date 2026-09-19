import pandas as pd
from typing import Dict, Any
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest


def validate_with_great_expectations(order: Dict[str, Any]) -> None:
    """
    Validate a single order using Great Expectations.
    Raises an exception if validation fails.
    """
    # Convert order to DataFrame
    df = pd.DataFrame([order])

    # Create a minimal in-memory context
    context = gx.get_context()

    # Define a simple suite of expectations
    suite_name = "order_validation_suite"

    try:
        suite = context.get_expectation_suite(suite_name)
    except:
        suite = context.add_expectation_suite(suite_name)

    # Clear old expectations (to keep it clean)
    suite.expectations = []

    # === Basic expectations ===
    # Required columns exist
    required_columns = [
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

    for col in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # Numeric columns should not be null
    numeric_columns = [
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
        "customer_zip_code"
    ]

    for col in numeric_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # Simple range checks
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="item_count", min_value=1, max_value=50
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_item_price", min_value=0, max_value=100000
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="avg_review_score", min_value=0, max_value=5
        )
    )

    # Allowed states (Brazilian states)
    brazilian_states = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ]
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="customer_state",
            value_set=brazilian_states
        )
    )

    # Save the suite
    context.save_expectation_suite(suite)

    # Run validation
    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name="order_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default"}
    )

    # For simplicity we use a basic validator approach
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    results = validator.validate()

    if not results["success"]:
        # Collect failed expectations
        failed = [
            exp["expectation_config"]["expectation_type"]
            for exp in results["results"]
            if not exp["success"]
        ]
        raise ValueError(f"Data validation failed. Failed expectations: {failed}")