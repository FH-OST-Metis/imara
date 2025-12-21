from __future__ import annotations

from pathlib import Path

import numpy as np


def build_graphs(reduced_embeddings, output_path: Path, seq_len: int = 5, target_dim: int = 10) -> None:
    graphs = []
    num_graphs = (len(reduced_embeddings) + seq_len - 1) // seq_len

    for i in range(num_graphs):
        start = i * seq_len
        end = min(len(reduced_embeddings), (i + 1) * seq_len)
        chunk = reduced_embeddings[start:end].tolist()

        # Pad with zero vectors if less than seq_len
        while len(chunk) < seq_len:
            chunk.append([0.0] * target_dim)

        # Simple dummy labels matching the original example pattern
        if i == 0:
            labels = [1, 0, 2, 1, 0]
        else:
            labels = [0, 1, 0, 2, 1]

        graphs.append({"features": chunk, "labels": labels})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graphs, f, indent=2)

    print(f"Saved {len(graphs)} graph examples to {output_path}")


def main() -> None:
    emb_path = Path("data/graphmert/embeddings/reduced_embeddings.npy")
    if not emb_path.exists():
        raise FileNotFoundError(f"Reduced embeddings not found: {emb_path}")

    reduced_embeddings = np.load(emb_path)

    output_dir = Path("data/graphmert/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_path = output_dir / "sentence_transformer_fixed10d_graphs.json"

    build_graphs(reduced_embeddings, graphs_path)


if __name__ == "__main__":
    main()
