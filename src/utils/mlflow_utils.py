import mlflow
import os
from ruamel.yaml import YAML
from dotenv import load_dotenv, find_dotenv

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
    yaml = YAML(typ='safe')
    params = yaml.load(open('params.yaml', encoding='utf-8'))
    return params['experiment']['name']