from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "stage" in data


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "request_count" in data
    assert "error_count" in data
    assert "error_rate" in data
    assert "avg_latency_seconds" in data
    assert "prediction_distribution" in data


def test_predict_success():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert data["label"] in ["late", "on_time"]
    assert 0 <= data["probability"] <= 1
    assert "model_version" in data
    assert "latency_seconds" in data


def test_predict_invalid_payload():
    # Missing many required fields
    payload = {"customer_state": "SP"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # FastAPI validation error
