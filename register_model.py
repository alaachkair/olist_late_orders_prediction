import mlflow
import mlflow.sklearn
import joblib
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    model_path = config["paths"]["model"]
    preprocessor_path = config["paths"]["preprocessor"]
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    registered_name = config["model"]["registered_name"]
    model_version = config["model"]["version"]

    # Load objects
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # MLflow setup
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"hist_gradient_boosting_v{model_version}") as run:

        mlflow.log_param("model_type", "HistGradientBoosting")
        mlflow.log_param("version", model_version)

        # Log and register the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_name,
            skops_trusted_types=[
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ]
        )

        # Log preprocessor
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        print("Run ID:", run.info.run_id)
        print(f"Model registered successfully as '{registered_name}'")


if __name__ == "__main__":
    main()