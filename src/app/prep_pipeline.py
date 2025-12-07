import mlflow
import logging
from utils.mlflow_helper import get_mlflow_experiment_name
from utils.mlflow_helper import mlflow_connect

_log = logging.getLogger(__name__)


def main():
    experiment_name = get_mlflow_experiment_name()
    mlflow_connect()

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        _log.info(
            f"Created new MLflow experiment: {experiment_name} with ID: {experiment_id}"
        )
    except mlflow.exceptions.MlflowException as e:
        _log.info(f"Experiment {experiment_name} already exists. Error: {e}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        _log.info(
            f"Using existing MLflow experiment: {experiment_name} with ID: {experiment_id}"
        )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="prep_pipeline"):
        mlflow.log_artifact("dvc.lock")


if __name__ == "__main__":
    main()
