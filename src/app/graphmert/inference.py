from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import networkx as nx
from transformers import pipeline
import sys

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app
from utils.device_helper import get_device


class HeteroGraphAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.attention_heads = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    embed_dim=embed_dim // num_heads,
                    num_heads=1,
                    batch_first=True,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.linear(x)
        splits = torch.chunk(x_proj, self.num_heads, dim=-1)
        head_outputs = []
        for split, attn in zip(splits, self.attention_heads):
            out, _ = attn(split, split, split)
            head_outputs.append(out)
        return torch.cat(head_outputs, dim=-1)


class GraphMERTEncoder(torch.nn.Module):
    def __init__(
        self, input_dim: int, embed_dim: int, num_layers: int, num_heads: int
    ) -> None:
        super().__init__()
        self.total_embed_dim = embed_dim * num_heads
        self.embedding = torch.nn.Linear(input_dim, self.total_embed_dim)
        self.hgat_layers = torch.nn.ModuleList(
            [
                HeteroGraphAttentionLayer(self.total_embed_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.total_embed_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for hgat in self.hgat_layers:
            x = hgat(x)
        x = self.transformer_encoder(x)
        return x


verbose = False
DEVICE = torch.device(get_device())


def load_model() -> GraphMERTEncoder:
    """Load the trained GraphMERT model from the weights checkpoint.

    We reconstruct ``GraphMERTEncoder`` and load the state dict from
    ``data/graphmert/graphmert_model_weights.pth``. This avoids the pickling
    issues that arise when loading a full serialized model across modules.
    """

    input_dim = 10
    embed_dim = 32
    num_heads = 4
    num_layers = 3

    model_path = Path("data/graphmert/graphmert_model_weights.pth")
    model = GraphMERTEncoder(input_dim, embed_dim, num_layers, num_heads)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def get_reduced_graphmert_embeddings(
    corpus: List[str], model: GraphMERTEncoder, input_dim: int = 10
):
    """Compute SBERT+PCA embeddings, run GraphMERT, and PCA-reduce again."""

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    device = get_device()
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    original_dim = sbert_model.get_sentence_embedding_dimension()
    if verbose:
        print(f"SBERT embedding dim: {original_dim}")

    sbert_embeddings = sbert_model.encode(corpus)

    n_components = min(input_dim, sbert_embeddings.shape[0], sbert_embeddings.shape[1])
    pca_text = PCA(n_components=n_components)
    reduced_sbert_emb = pca_text.fit_transform(sbert_embeddings)

    if verbose:
        print(
            f"Reduced embeddings from {original_dim}D to {n_components}D "
            f"for {len(corpus)} text units"
        )

    # Prepare inputs for GraphMERT: (1, N, input_dim)
    input_features = torch.tensor(
        [reduced_sbert_emb], dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        graphmert_outputs = model(input_features)  # (1, N, embed_dim * num_heads)
        if verbose:
            print("Inference output shape:", graphmert_outputs.shape)

    graphmert_embeddings = graphmert_outputs[0].cpu().numpy()  # (N, D)

    # PCA over GraphMERT embeddings to get final node representations
    pca_graphmert = PCA(n_components=input_dim)
    reduced_graphmert_emb = pca_graphmert.fit_transform(graphmert_embeddings)
    return reduced_graphmert_emb, sbert_model, pca_text


def build_graph(corpus: List[str], reduced_graphmert_emb: np.ndarray) -> nx.Graph:
    """Build a simple chain graph with text + GraphMERT embeddings as node attrs."""

    G = nx.Graph()
    for idx, (txt, emb) in enumerate(zip(corpus, reduced_graphmert_emb)):
        G.add_node(idx, text=txt, embedding=emb)
    for i in range(len(corpus) - 1):
        G.add_edge(i, i + 1)
    return G


def retrieve_similar_nodes(
    query: str, graph: nx.Graph, sbert_model, pca_text, top_k: int = 3
) -> List[str]:
    """Retrieve top_k most similar nodes using SBERT + PCA cosine similarity."""

    query_emb = sbert_model.encode([query])
    query_emb_reduced = pca_text.transform(query_emb)[0]  # (input_dim,)

    scores = []
    for node in graph.nodes:
        node_emb = graph.nodes[node]["embedding"]
        sim = float(
            np.dot(query_emb_reduced, node_emb)
            / (np.linalg.norm(query_emb_reduced) * np.linalg.norm(node_emb) + 1e-8)
        )
        scores.append((node, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [graph.nodes[n]["text"] for n, _ in scores[:top_k]]


# Simple text-generation model; can be swapped for something else later.
generator = pipeline("text-generation", model="gpt2")


def graphrag_generate(query: str, G: nx.Graph, sbert_model, pca_text) -> str:
    """Generate an answer using retrieved context + a local text generator."""

    retrieved = retrieve_similar_nodes(query, G, sbert_model, pca_text)
    context = "\n".join(retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    results = generator(prompt, max_length=150, num_return_sequences=1)
    return results[0]["generated_text"]


def main() -> None:
    # Use the chunks produced by this repository's pipeline as corpus
    from graphmert.corpus import get_corpus_from_chunks

    chunk_dir = Path("data/processed/chunks")
    corpus = get_corpus_from_chunks(chunk_dir)

    model = load_model()
    reduced_graphmert_emb, sbert_model, pca_text = get_reduced_graphmert_embeddings(
        corpus, model, input_dim=10
    )

    G = build_graph(corpus, reduced_graphmert_emb)

    # Example query; replace with your own
    query = "What is discussed about graphs in this document?"
    answer = graphrag_generate(query, G, sbert_model, pca_text)
    print("GraphRAG answer:\n", answer)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
