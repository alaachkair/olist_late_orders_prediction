from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class OrderRequest(BaseModel):
    order_purchase_timestamp: str
    order_approved_at: Optional[str] = None
    order_estimated_delivery_date: str
    customer_zip_code: int
    customer_state: str
    customer_city: str
    item_count: int = Field(..., ge=1, le=50)
    total_item_price: float = Field(..., ge=0)
    total_freight_value: float = Field(..., ge=0)
    unique_products: int = Field(..., ge=1)
    unique_sellers: int = Field(..., ge=1)
    payment_count: int = Field(..., ge=1)
    total_payment_value: float = Field(..., ge=0)
    max_installments: int = Field(..., ge=1)
    unique_payment_types: int = Field(..., ge=1)
    review_count: float = 0
    avg_review_score: float = Field(0, ge=0, le=5)
    min_review_score: float = Field(0, ge=0, le=5)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    probability: float
    label: str
    model_version: str
    latency_seconds: float


class BatchOrderRequest(BaseModel):
    orders: List[OrderRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: str
    stage: str