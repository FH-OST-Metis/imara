from __future__ import annotations

from pathlib import Path
import json
from typing import List
import sys
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import mlflow

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app
from utils.device_helper import get_device
from utils.mlflow_helper import mlflow_connect, get_mlflow_experiment_name

def get_embeddings(corpus: List[str], target_dim: int):
    device = get_device()
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    mlflow.log_param("graphmert_sbert_model", "all-MiniLM-L6-v2")

    original_dim = sbert_model.get_sentence_embedding_dimension()
    print(f"SBERT embedding dim: {original_dim}")
    mlflow.log_param("graphmert_sbert_dim", original_dim)

    sbert_embeddings = sbert_model.encode(corpus)

    n_components = min(target_dim, sbert_embeddings.shape[0], sbert_embeddings.shape[1])
    pca = PCA(n_components=n_components)
    reduced_sbert_emb = pca.fit_transform(sbert_embeddings)

    print(
        f"Reduced embeddings from {original_dim}D to {n_components}D "
        f"for {len(corpus)} text units"
    )
    mlflow.log_param("graphmert_target_dim", target_dim)
    mlflow.log_metric("graphmert_num_text_units", len(corpus))

    return reduced_sbert_emb, sbert_model, pca


def main() -> None:
    corpus_path = Path("data/graphmert/corpus/corpus.json")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    target_dim = 10  # match training input_dim
    reduced_embeddings, _sbert_model, _pca = get_embeddings(corpus, target_dim)

    output_dir = Path("data/graphmert/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "reduced_embeddings.npy"
    np.save(emb_path, reduced_embeddings)

    print(f"Saved reduced embeddings with shape {reduced_embeddings.shape} to {emb_path}")
    mlflow.log_metric(
        "graphmert_embedding_rows", float(reduced_embeddings.shape[0]), step=0
    )
    mlflow.log_metric(
        "graphmert_embedding_dim", float(reduced_embeddings.shape[1]), step=0
    )


if __name__ == "__main__":
    mlflow_connect()
    experiment_name = get_mlflow_experiment_name()
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"graphmert_embeddings_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        main()
