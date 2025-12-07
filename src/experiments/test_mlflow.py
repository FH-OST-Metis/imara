#!/usr/bin/env python3
"""
Test MLflow integration with Supabase PostgreSQL and S3 storage.

This script tests:
1. Parameter logging to Supabase PostgreSQL backend
2. Artifact saving to Supabase S3 bucket
3. Metric logging

Run this script after starting your MLflow server:
    1. Terminal 1: uv run start_mlflow.py
    2. Terminal 2: uv run src/eperiments/test_mlflow.py
"""

import os
import sys
import tempfile
import json
import random
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

try:
    import mlflow
except ImportError as e:
    print(f"Missing required packages. Please install mlflow: {e}")
    sys.exit(1)


def setup_mlflow():
    """Set up MLflow with Supabase configuration."""
    print("=== Setting up MLflow ===")

    # Set MLflow tracking URI to connect to our server
    mlflow_host = os.getenv("MLFLOW_HOST", "127.0.0.1")
    mlflow_port = os.getenv("MLFLOW_PORT", "5000")
    tracking_uri = f"http://{mlflow_host}:{mlflow_port}"

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Set experiment name
    experiment_name = "supabase_integration_test"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        mlflow.set_experiment(experiment_name)
        print(f"Using existing experiment: {experiment_name}")

    return tracking_uri


def test_parameter_logging():
    """Test parameter logging to Supabase PostgreSQL."""
    print("\n=== Testing Parameter Logging ===")

    with mlflow.start_run(run_name="parameter_test") as run:
        # Log various types of parameters
        test_params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "model_type": "random_forest",
            "test_mode": True,
            "random_seed": 42,
        }

        for param_name, param_value in test_params.items():
            mlflow.log_param(param_name, param_value)
            print(f"✓ Logged parameter: {param_name} = {param_value}")

        # Log some metrics
        test_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
        }

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"✓ Logged metric: {metric_name} = {metric_value}")

        return run.info.run_id


def test_artifact_storage():
    """Test artifact storage to Supabase S3."""
    print("\n=== Testing Artifact Storage ===")

    with mlflow.start_run(run_name="artifact_test") as run:
        # Create various test artifacts

        # 1. JSON file
        json_data = {
            "experiment_config": {
                "model_type": "test_model",
                "parameters": {"param1": "value1", "param2": 42},
                "timestamp": "2024-11-09T12:00:00Z",
            },
            "results": {"accuracy": 0.95, "loss": 0.05},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f, indent=2)
            json_file = f.name

        mlflow.log_artifact(json_file, "config")
        os.unlink(json_file)
        print("✓ Logged JSON artifact")

        # 2. CSV file with sample data
        csv_content = "feature1,feature2,target\n"
        for i in range(100):
            csv_content += f"{random.gauss(0, 1):.4f},{random.gauss(0, 1):.4f},{random.randint(0, 1)}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_file = f.name

        mlflow.log_artifact(csv_file, "data")
        os.unlink(csv_file)
        print("✓ Logged CSV artifact")

        # 3. Text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test artifact for MLflow Supabase integration.\n")
            f.write("Testing artifact storage to S3 bucket.\n")
            f.write(f"Run ID: {run.info.run_id}\n")
            text_file = f.name

        mlflow.log_artifact(text_file, "logs")
        os.unlink(text_file)
        print("✓ Logged text artifact")

        # 4. Log a directory of artifacts
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create multiple files in a directory
            (temp_dir / "file1.txt").write_text("Content of file 1")
            (temp_dir / "file2.txt").write_text("Content of file 2")
            (temp_dir / "subdir").mkdir()
            (temp_dir / "subdir" / "nested_file.txt").write_text("Nested content")

            mlflow.log_artifacts(str(temp_dir), "batch_upload")
            print("✓ Logged directory of artifacts")
        finally:
            # Cleanup
            import shutil

            shutil.rmtree(temp_dir)

        return run.info.run_id


def test_comprehensive_logging():
    """Test comprehensive logging including params, metrics, and artifacts."""
    print("\n=== Testing Comprehensive Logging ===")

    with mlflow.start_run(run_name="comprehensive_test") as run:
        # Log various types of parameters
        test_params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "model_type": "test_model",
            "test_mode": True,
            "random_seed": 42,
            "dataset": "sample_data",
            "preprocessing": "standard_scaler",
        }

        for param_name, param_value in test_params.items():
            mlflow.log_param(param_name, param_value)
            print(f"✓ Logged parameter: {param_name} = {param_value}")

        # Log multiple metrics over "epochs"
        print("\nLogging metrics over time:")
        for epoch in range(1, 11):
            # Simulate training metrics
            train_loss = 1.0 / epoch + random.uniform(-0.1, 0.1)
            val_loss = 1.1 / epoch + random.uniform(-0.1, 0.1)
            accuracy = min(0.95, 0.5 + (epoch * 0.05) + random.uniform(-0.02, 0.02))

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        print("✓ Logged time-series metrics")

        # Log final summary metrics
        summary_metrics = {
            "final_accuracy": 0.94,
            "final_loss": 0.12,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
            "training_time_seconds": 125.5,
            "num_parameters": 1_000_000,
        }

        for metric_name, metric_value in summary_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"✓ Logged metric: {metric_name} = {metric_value}")

        # Create and log various artifacts
        print("\nCreating and logging artifacts:")

        # 1. Model configuration JSON
        model_config = {
            "architecture": {
                "type": "neural_network",
                "layers": [128, 64, 32, 2],
                "activation": "relu",
                "dropout": 0.2,
            },
            "training": {
                "optimizer": test_params["optimizer"],
                "learning_rate": test_params["learning_rate"],
                "batch_size": test_params["batch_size"],
                "epochs": test_params["epochs"],
            },
            "performance": summary_metrics,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_config, f, indent=2)
            config_file = f.name

        mlflow.log_artifact(config_file, "config")
        os.unlink(config_file)
        print("✓ Logged model configuration")

        # 2. Training log
        training_log = f"""Training Log - Run {run.info.run_id}
{"=" * 60}
Model Type: {test_params["model_type"]}
Dataset: {test_params["dataset"]}
Random Seed: {test_params["random_seed"]}

Hyperparameters:
  - Learning Rate: {test_params["learning_rate"]}
  - Batch Size: {test_params["batch_size"]}
  - Epochs: {test_params["epochs"]}
  - Optimizer: {test_params["optimizer"]}

Final Results:
  - Accuracy: {summary_metrics["final_accuracy"]:.4f}
  - F1 Score: {summary_metrics["f1_score"]:.4f}
  - Training Time: {summary_metrics["training_time_seconds"]:.2f}s
  
Status: Completed Successfully
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(training_log)
            log_file = f.name

        mlflow.log_artifact(log_file, "logs")
        os.unlink(log_file)
        print("✓ Logged training log")

        # 3. Sample predictions CSV
        csv_content = "sample_id,prediction,confidence,ground_truth\n"
        for i in range(50):
            pred = random.randint(0, 1)
            conf = random.uniform(0.7, 0.99)
            truth = random.randint(0, 1)
            csv_content += f"{i},{pred},{conf:.4f},{truth}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_file = f.name

        mlflow.log_artifact(csv_file, "predictions")
        os.unlink(csv_file)
        print("✓ Logged predictions CSV")

        # 4. Metrics summary
        metrics_summary = {
            "classification_metrics": {
                "accuracy": summary_metrics["final_accuracy"],
                "precision": summary_metrics["precision"],
                "recall": summary_metrics["recall"],
                "f1_score": summary_metrics["f1_score"],
            },
            "training_info": {
                "total_epochs": test_params["epochs"],
                "final_loss": summary_metrics["final_loss"],
                "training_time": summary_metrics["training_time_seconds"],
                "convergence": True,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics_summary, f, indent=2)
            summary_file = f.name

        mlflow.log_artifact(summary_file, "results")
        os.unlink(summary_file)
        print("✓ Logged metrics summary")

        # 5. Directory with multiple files
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create multiple files in a directory
            (temp_dir / "readme.txt").write_text("Additional experiment files")
            (temp_dir / "data_stats.txt").write_text(
                "Mean: 0.5\nStd: 0.2\nSamples: 1000"
            )
            (temp_dir / "notes.txt").write_text("This was a successful training run.")

            # Create subdirectory
            (temp_dir / "plots").mkdir()
            (temp_dir / "plots" / "plot_info.txt").write_text("Placeholder for plots")

            mlflow.log_artifacts(str(temp_dir), "additional_files")
            print("✓ Logged directory of artifacts")
        finally:
            # Cleanup
            import shutil

            shutil.rmtree(temp_dir)

        return run.info.run_id


def verify_run_data(run_id):
    """Verify that the run data was properly stored."""
    print("\n=== Verifying Run Data ===")

    try:
        run = mlflow.get_run(run_id)

        print("\n📊 Run Information:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"  Status: {run.info.status}")
        print(f"  Experiment ID: {run.info.experiment_id}")

        if run.data.params:
            print(f"\n🔧 Parameters ({len(run.data.params)}):")
            for key, value in sorted(run.data.params.items())[:5]:  # Show first 5
                print(f"  {key}: {value}")
            if len(run.data.params) > 5:
                print(f"  ... and {len(run.data.params) - 5} more")

        if run.data.metrics:
            print(f"\n📈 Metrics ({len(run.data.metrics)}):")
            for key, value in sorted(run.data.metrics.items())[:5]:  # Show first 5
                print(f"  {key}: {value}")
            if len(run.data.metrics) > 5:
                print(f"  ... and {len(run.data.metrics) - 5} more")

        # List artifacts
        try:
            artifacts = mlflow.artifacts.list_artifacts(run_id)
            if artifacts:
                print("\n📁 Artifacts:")
                for artifact in artifacts:
                    artifact_type = "📂" if artifact.is_dir else "📄"
                    print(f"  {artifact_type} {artifact.path}")
        except Exception as e:
            print(f"  ⚠️  Error listing artifacts: {e}")

        return True

    except Exception as e:
        print(f"❌ Error retrieving run data: {e}")
        return False


def main():
    """Main test function."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MLflow Supabase Integration Test" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")

    try:
        # Setup MLflow
        tracking_uri = setup_mlflow()

        # Run parameter logging test
        param_run_id = test_parameter_logging()
        verify_run_data(param_run_id)

        # Run artifact storage test
        artifact_run_id = test_artifact_storage()
        verify_run_data(artifact_run_id)

        # Run comprehensive test
        comprehensive_run_id = test_comprehensive_logging()
        verify_run_data(comprehensive_run_id)

        # Final summary
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 10 + "✅ All Tests Completed Successfully!" + " " * 11 + "║")
        print("╚" + "═" * 58 + "╝")

        print(f"\n📊 MLflow UI: {tracking_uri}")
        print("�️  Database: Supabase PostgreSQL")
        print("☁️  Storage: Supabase S3 Bucket")

        print("\n🔍 Run IDs created:")
        print(f"  1️⃣  Parameter Test:     {param_run_id}")
        print(f"  2️⃣  Artifact Test:      {artifact_run_id}")
        print(f"  3️⃣  Comprehensive Test: {comprehensive_run_id}")

        print("\n💡 Next Steps:")
        print("  • Check MLflow UI to view experiments")
        print("  • Verify data in Supabase PostgreSQL dashboard")
        print("  • Check artifacts in Supabase S3 bucket")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
