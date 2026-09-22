import mlflow
from mlflow.tracking import MlflowClient
import yaml


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    tracking_uri = config["mlflow"]["tracking_uri"]
    model_name = config["model"]["registered_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Transition version 1 to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Staging"
    )

    print(f"Model '{model_name}' version 1 moved to Staging")


if __name__ == "__main__":
    main()