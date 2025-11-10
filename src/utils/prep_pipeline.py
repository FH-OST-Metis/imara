import mlflow
from utils.mlflow_utils import get_mlflow_experiment_name
from utils.mlflow_utils import mlflow_connect

def main():    
    experiment_name = get_mlflow_experiment_name()
    mlflow_connect()

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new MLflow experiment: {experiment_name} with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Experiment {experiment_name} already exists. Error: {e}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing MLflow experiment: {experiment_name} with ID: {experiment_id}")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="prep_pipeline"):
        mlflow.log_artifact("dvc.lock")
    

if __name__ == "__main__":
    main()