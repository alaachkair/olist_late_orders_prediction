import yaml
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchOrderRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderRequest,
    PredictionResponse,
)
from src.pipeline import run_inference


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()

app = FastAPI(
    title="Olist Late Orders Prediction API",
    description="Predict whether an order will be delivered late or on time",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {"status": "ok", "model_version": config["model"]["stage"]}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    return {
        "model_name": config["model"]["registered_name"],
        "model_version": config["model"]["version"],
        "stage": config["model"]["stage"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(order: OrderRequest):
    try:
        order_dict = order.model_dump()
        result = run_inference(order=order_dict, use_registry=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchOrderRequest):
    try:
        results = []
        for order in batch.orders:
            order_dict = order.model_dump()
            result = run_inference(order=order_dict, use_registry=False)
            results.append(result)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
