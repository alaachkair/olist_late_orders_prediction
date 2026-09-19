import pandas as pd
from typing import Dict, Any
import great_expectations as gx


def validate_with_great_expectations(order: Dict[str, Any]) -> None:
    """
    Validate a single order using Great Expectations (simple runtime style).
    Raises ValueError if validation fails.
    """
    df = pd.DataFrame([order])

    # Create an ephemeral context (no project files needed)
    context = gx.get_context(mode="ephemeral")

    # Create a datasource and data asset on the fly
    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="order_asset")

    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Define expectations directly on the batch
    # 1. Required columns exist
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
        result = batch.expect_column_to_exist(col)
        if not result.success:
            raise ValueError(f"Missing required column: {col}")

    # 2. Numeric columns should not be null
    numeric_columns = [
        "item_count", "total_item_price", "total_freight_value",
        "unique_products", "unique_sellers", "payment_count",
        "total_payment_value", "max_installments", "unique_payment_types",
        "review_count", "avg_review_score", "min_review_score",
        "customer_zip_code"
    ]

    for col in numeric_columns:
        result = batch.expect_column_values_to_not_be_null(col)
        if not result.success:
            raise ValueError(f"Null values found in column: {col}")

    # 3. Simple range checks
    result = batch.expect_column_values_to_be_between("item_count", min_value=1, max_value=50)
    if not result.success:
        raise ValueError("item_count is out of allowed range (1-50)")

    result = batch.expect_column_values_to_be_between("total_item_price", min_value=0, max_value=100000)
    if not result.success:
        raise ValueError("total_item_price is out of allowed range")

    result = batch.expect_column_values_to_be_between("avg_review_score", min_value=0, max_value=5)
    if not result.success:
        raise ValueError("avg_review_score must be between 0 and 5")

    # 4. Allowed Brazilian states
    brazilian_states = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ]
    result = batch.expect_column_values_to_be_in_set("customer_state", value_set=brazilian_states)
    if not result.success:
        raise ValueError(f"customer_state must be one of: {brazilian_states}")