# Olist Late Orders Prediction Service

Inference service that predicts whether an order will be delivered **late** or **on time**.

This project turns the trained model from the notebooks into a production-ready inference pipeline with:

- Clean Python modules
- FastAPI service
- Docker & Docker Compose
- MLflow model registry
- Data validation
- Logging & monitoring
- CI/CD with GitHub Actions
- Pre-commit hooks

---

## Project Structure

```text
olist_late_orders_prediction/
├── app/                  # FastAPI application
│   ├── main.py
│   └── schemas.py
├── config/
│   └── config.yaml       # All paths and parameters
├── data/                 # Data files (DVC tracked)
├── models/               # Model & preprocessor (DVC tracked)
├── src/                  # Core business logic
│   ├── data.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── predict.py
│   ├── pipeline.py
│   ├── validation.py
│   ├── ge_validation.py
│   ├── logging_config.py
│   └── metrics.py
├── tests/                # Unit tests
├── logs/                 # Prediction logs & service logs
├── notebooks/            # Original notebooks (not used in production)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── MONITORING.md
└── README.md
