import mlflow
import os
from dotenv import load_dotenv, find_dotenv
from utils.params_helper import load_params

load_dotenv(find_dotenv())


def mlflow_connect():
    """
    Connects to local MLflow tracking server.
    """

    host = os.getenv("MLFLOW_HOST", "localhost")
    port = os.getenv("MLFLOW_PORT", "5000")

    mlflow.set_tracking_uri(f"http://{host}:{port}")


def get_mlflow_experiment_name() -> str:
    """
    Retrieves the MLflow experiment name from params.yaml.
    Returns:
        str: The name of the MLflow experiment.
    """
    extract_params = load_params("experiment")
    name: str = extract_params.get("name", "Default Experiment")
    return name
