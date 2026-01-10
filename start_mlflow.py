#!/usr/bin/env python3
"""Cross-platform MLflow server launcher that loads environment variables."""

import os
import sys
import subprocess
from dotenv import load_dotenv, find_dotenv

# Load .env file from workspace root
load_dotenv(find_dotenv())

# Print configuration
print("\n=== MLflow Server Configuration ===")

# Start MLflow server with debug logging
cmd = [
    "mlflow",
    "server",
    "--backend-store-uri",
    os.getenv("MLFLOW_BACKEND_STORE_URI", "postgresql://postgres:[Password]@db.[projectId].supabase.co:5432/postgres"),
    "--default-artifact-root",
    os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "s3://Imara-mlflow/"),
    "--host",
    os.getenv("MLFLOW_HOST", "0.0.0.0"),
    "--port",
    os.getenv("MLFLOW_PORT", "5000"),
    "--workers",
    "1",
    "--gunicorn-opts",
    "--log-level debug --timeout 120",
]

# Ensure output is not buffered
os.environ["PYTHONUNBUFFERED"] = "1"

sys.exit(subprocess.call(cmd))
