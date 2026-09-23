\# Olist Late Orders Prediction



Inference service that predicts whether an order will be delivered late or on time.



\## Project Structure

olist\_late\_orders\_prediction/

├── app/                  # FastAPI application

├── config/               # Configuration files

├── data/                 # Data (tracked later with DVC)

├── models/               # Saved models \& transformers

├── notebooks/            # Original notebooks

├── src/                  # Core Python modules

├── tests/                # Unit \& integration tests

├── requirements.txt      # Runtime dependencies

├── requirements-dev.txt  # Development dependencies

└── README.md



\## Setup (from zero)



```bash

\# 1. Clone the repository

git clone <your-repo-url>

cd olist\_late\_orders\_prediction



\# 2. Create virtual environment

python -m venv .venv

source .venv/Scripts/activate      # Windows (Git Bash)

\# source .venv/bin/activate        # Mac/Linux



\# 3. Install dependencies

pip install -r requirements-dev.txt

How to run

(Coming in later steps)

Notes



No hardcoded paths or parameters — everything comes from config/config.yaml

Training stays in notebooks. This repo only contains the inference pipeline.
