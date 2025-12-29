from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from transformers import pipeline

# Ensure sibling 'utils' and 'graphmert' packages are importable when running as script via path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app

from utils.params_helper import load_params
from embedder import EmbeddingModel, EmbeddingConfig
from graphmert.corpus import get_corpus_from_chunks
from graphmert.train_model import GraphMERTEncoder
from graphmert.inference import (
    get_reduced_graphmert_embeddings,
    build_graph,
    graphrag_generate,
)


def _load_graphmert_model() -> GraphMERTEncoder:
    """Load the trained GraphMERT model weights.

    We reuse the architecture from ``graphmert.train_model.GraphMERTEncoder`` and
    load the weights saved by the training pipeline under
    ``data/graphmert/model/graphmert_model_weights.pth``.
    """

    input_dim = 10
    embed_dim = 32
    num_heads = 4
    num_layers = 3

    weights_path = Path("data/graphmert/model/graphmert_model_weights.pth")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"GraphMERT weights not found at {weights_path}. "
            "Run the graphmert_train_model DVC stage first."
        )

    model = GraphMERTEncoder(input_dim, embed_dim, num_layers, num_heads)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_graphmert_state(
    chunks_dir: Path,
) -> Tuple[nx.Graph, Any, Any]:
    """Build GraphMERT graph and retrieval state from pipeline chunks.

    Returns the NetworkX graph, the SBERT model, and the PCA transformer used
    for query-time retrieval.
    """

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunks_dir}")

    corpus = get_corpus_from_chunks(chunks_dir)
    model = _load_graphmert_model()

    # Reuse helper from graphmert.inference to get reduced embeddings and
    # the SBERT/PCA objects needed for retrieval.
    reduced_graphmert_emb, sbert_model, pca_text = get_reduced_graphmert_embeddings(
        corpus, model, input_dim=10
    )

    G = build_graph(corpus, reduced_graphmert_emb)
    return G, sbert_model, pca_text


# ----------------------
# GraphMERT retrieval
# ----------------------


def _retrieve_with_ids(
    query: str,
    graph: nx.Graph,
    sbert_model: Any,
    pca_text: Any,
    top_k: int,
) -> List[Tuple[int, str]]:
    """Retrieve top-k most similar nodes and return (node_id, text).

    Mirrors the cosine-similarity logic used in ``graphmert.inference`` but
    also returns the node ids so we can populate ``passage_id`` fields.
    """

    # Encode and PCA-project the query
    query_emb = sbert_model.encode([query])
    query_emb_reduced = pca_text.transform(query_emb)[0]

    scores: List[Tuple[int, float]] = []
    for node in graph.nodes:
        node_emb = graph.nodes[node]["embedding"]
        sim = float(
            (query_emb_reduced @ node_emb)
            / (
                float(torch.linalg.vector_norm(torch.tensor(query_emb_reduced)))
                * float(torch.linalg.vector_norm(torch.tensor(node_emb)))
                + 1e-8
            )
        )
        scores.append((int(node), sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in scores[:top_k]]
    return [(nid, graph.nodes[nid]["text"]) for nid in top_nodes]


# ----------------------
# Naive RAG (SBERT over chunks)
# ----------------------

_naive_generator = pipeline("text-generation", model="gpt2")


def _build_naive_corpus(
    chunks_dir: Path,
    embed_cfg: EmbeddingConfig,
) -> Tuple[List[str], np.ndarray, EmbeddingModel]:
    """Load chunk texts and pre-compute embeddings for naive RAG.

    Returns (texts, embeddings, embedding_model).
    """

    if not chunks_dir.exists() or not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunks_dir}")

    texts: List[str] = []
    for path in sorted(chunks_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)

    if not texts:
        raise RuntimeError(f"No non-empty .txt chunk files found in {chunks_dir}")

    model = EmbeddingModel(embed_cfg)
    embs = model.embed_texts(texts)  # shape (N, D), L2-normalized
    return texts, embs, model


def _naive_retrieve(
    query: str,
    texts: List[str],
    embs: np.ndarray,
    model: EmbeddingModel,
    top_k: int,
) -> List[Tuple[int, str]]:
    """Retrieve top-k chunks via cosine similarity over SBERT embeddings."""

    if not texts:
        return []

    q_emb = model.embed_texts([query])  # (1, D)
    if q_emb.size == 0:
        return []

    q_vec = q_emb[0]  # (D,)
    # Cosine similarity == dot product because embeddings are normalized.
    sims = embs @ q_vec.astype(np.float32)
    if top_k >= len(texts):
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(int(i), texts[i]) for i in top_idx]


def _naive_generate_answer(query: str, contexts: List[str]) -> str:
    """Generate an answer from retrieved contexts using a small local model."""

    context = "\n".join(contexts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    results = _naive_generator(prompt, max_length=150, num_return_sequences=1)
    return results[0]["generated_text"]


def run_graphmert_eval(
    queries_csv: Path,
    output_csv: Path,
    top_k: int,
) -> None:
    """Run GraphMERT-based RAG over queries and export results for Open RAG Eval.

    The output CSV follows the schema expected by ``RAGResultsLoader`` in
    ``open-rag-eval``:

    - query_id
    - query
    - query_run
    - passage_id
    - passage
    - generated_answer
    """

    if not queries_csv.exists():
        raise FileNotFoundError(f"Queries CSV not found: {queries_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    G, sbert_model, pca_text = _build_graphmert_state(Path("data/processed/chunks"))

    with queries_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        queries: List[Dict[str, str]] = list(reader)

    fieldnames = [
        "query_id",
        "query",
        "query_run",
        "passage_id",
        "passage",
        "generated_answer",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(queries):
            raw_query = (row.get("query") or row.get("question") or "").strip()
            if not raw_query:
                # Skip empty-query rows to match Connector.read_queries behavior
                continue

            query_id = (row.get("query_id") or row.get("id") or str(idx)).strip()
            if not query_id:
                query_id = str(idx)

            query_run = 1

            # Retrieve top-k passages
            passages = _retrieve_with_ids(raw_query, G, sbert_model, pca_text, top_k)

            # Generate an answer using the same helper as graphmert.inference.
            # This function internally performs its own retrieval for context,
            # which is acceptable for evaluation; passages we log are for
            # transparency and do not need to exactly mirror that retrieval.
            try:
                answer = graphrag_generate(raw_query, G, sbert_model, pca_text)
            except Exception as exc:  # pragma: no cover - defensive path
                answer = f"Generation error: {exc}"

            if not passages:
                # Still emit a single row so the query participates in eval.
                writer.writerow(
                    {
                        "query_id": query_id,
                        "query": raw_query,
                        "query_run": query_run,
                        "passage_id": "NA",
                        "passage": "",
                        "generated_answer": answer,
                    }
                )
                continue

            for j, (node_id, passage_text) in enumerate(passages):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "query": raw_query,
                        "query_run": query_run,
                        "passage_id": str(node_id),
                        "passage": passage_text,
                        "generated_answer": answer if j == 0 else "",
                    }
                )


def run_naiverag_eval(
    queries_csv: Path,
    output_csv: Path,
    top_k: int,
) -> None:
    """Run naive RAG (SBERT over chunks) and export results for Open RAG Eval."""

    if not queries_csv.exists():
        raise FileNotFoundError(f"Queries CSV not found: {queries_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    chunks_dir = Path("data/processed/chunks")
    embed_cfg = EmbeddingConfig()
    texts, embs, model = _build_naive_corpus(chunks_dir, embed_cfg)

    with queries_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        queries: List[Dict[str, str]] = list(reader)

    fieldnames = [
        "query_id",
        "query",
        "query_run",
        "passage_id",
        "passage",
        "generated_answer",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(queries):
            raw_query = (row.get("query") or row.get("question") or "").strip()
            if not raw_query:
                continue

            query_id = (row.get("query_id") or row.get("id") or str(idx)).strip()
            if not query_id:
                query_id = str(idx)

            query_run = 1

            passages = _naive_retrieve(raw_query, texts, embs, model, top_k)
            try:
                answer = _naive_generate_answer(
                    raw_query, [p for _, p in passages] if passages else []
                )
            except Exception as exc:  # pragma: no cover - defensive
                answer = f"Generation error: {exc}"

            if not passages:
                writer.writerow(
                    {
                        "query_id": query_id,
                        "query": raw_query,
                        "query_run": query_run,
                        "passage_id": "NA",
                        "passage": "",
                        "generated_answer": answer,
                    }
                )
                continue

            for j, (idx_passage, passage_text) in enumerate(passages):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "query": raw_query,
                        "query_run": query_run,
                        "passage_id": str(idx_passage),
                        "passage": passage_text,
                        "generated_answer": answer if j == 0 else "",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAG pipelines in this repo and export results for open-rag-eval. "
            "Currently supports the GraphMERT-based RAG system."
        )
    )
    parser.add_argument(
        "--system",
        choices=["graphmert", "naiverag"],
        default="graphmert",
        help="Which RAG system to evaluate.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval/queries.csv"),
        help="Path to CSV file with at least a 'query' column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/graphmert/generated_answers.csv"),
        help="Where to write the generated answers CSV for open-rag-eval.",
    )

    args = parser.parse_args()

    eval_params: Dict[str, Any] = load_params("eval")
    top_k = int(eval_params.get("top_k", 5))

    if args.system == "graphmert":
        run_graphmert_eval(args.queries.resolve(), args.output.resolve(), top_k)
    elif args.system == "naiverag":
        run_naiverag_eval(args.queries.resolve(), args.output.resolve(), top_k)
    else:  # pragma: no cover - protected by argparse choices
        raise NotImplementedError(f"Unsupported system: {args.system}")


if __name__ == "__main__":
    main()
