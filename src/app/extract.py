import shutil
import argparse
import mlflow
from pathlib import Path
import dvc.api
from utils.mlflow_utils import mlflow_connect
from utils.mlflow_utils import get_mlflow_experiment_name

def main():
    # Inject src and dst from command line arguments
    parser = argparse.ArgumentParser(description="Extract files from demo to data directory.")
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument("--output", type=str, help="Output file path", required=True)

    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    # Inject MLflow Tracking Server by .env variable
    mlflow_connect()
    # Inject MLflow Experiment Name by params.yaml
    mlflow.set_experiment(get_mlflow_experiment_name())

    # All logged parameter, metrics and artifacts will be associated to: experiment (pipeline) => run (stage)
    with mlflow.start_run(run_name="extract"):
        total_size = 0
        total_files = 0

        for file in src.rglob("*"):
            if not file.is_file():
                continue
            
            total_files += 1
            file_size = file.stat().st_size
            total_size += file_size
            mlflow.log_metric("file_version", dvc.api.read(f"{file}.dvc"), step=total_files)
            mlflow.log_metric("file_size", file_size, step=total_files)
            try:
                with open(file, "r", encoding="utf-8") as f:
                    file_length = sum(1 for _ in f)
                mlflow.log_metric("file_length", file_length, step=total_files)
            except Exception as e:
                pass  # Skip logging file_length if not a text file

            target = dst / file.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(file), str(target))

        mlflow.log_param("total_files_extracted", total_files)
        mlflow.log_param("total_size_extracted", total_size)
        print("✅ Done: all files extracted recursively.")

if __name__ == "__main__":
    main()
