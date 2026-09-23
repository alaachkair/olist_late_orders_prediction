from src.pipeline import run_inference

# Example order
sample_order = {
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
    "min_review_score": 4.0,
}

# Run using the MLflow Model Registry
result = run_inference(order=sample_order, use_registry=True)

print("\nPrediction result:")
print(result)
