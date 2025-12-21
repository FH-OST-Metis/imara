from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import sys
from datetime import datetime

import mlflow

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app
from utils.device_helper import get_device
from utils.mlflow_helper import mlflow_connect, get_mlflow_experiment_name


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


class ChainGraphDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: Path) -> None:
        import json

        with json_path.open("r", encoding="utf-8") as f:
            self.chain_graphs = json.load(f)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.chain_graphs)

    def __getitem__(self, idx: int):  # type: ignore[override]
        example = self.chain_graphs[idx]
        features = torch.tensor(example["features"], dtype=torch.float)
        labels = example.get("labels")
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.full((features.shape[0],), -100, dtype=torch.long)
        return features, labels


def train_graphmert(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> None:
    """Train GraphMERT model on prepared ChainGraphDataset."""
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)  # (batch, seq_len, embed_dim*num_heads)
            logits = outputs.view(-1, outputs.size(-1))
            loss = criterion(logits, labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def main() -> None:
    input_dim = 10
    embed_dim = 32
    num_heads = 4
    num_layers = 3
    batch_size = 8
    num_epochs = 10

    device = torch.device(get_device())

    output_dir = Path("data/graphmert/model")
    graphs_json_path = Path("data/graphmert/graphs/sentence_transformer_fixed10d_graphs.json")
    if not graphs_json_path.exists():
        raise FileNotFoundError(f"Graphs JSON not found: {graphs_json_path}")

    dataset = ChainGraphDataset(graphs_json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GraphMERTEncoder(input_dim, embed_dim, num_layers, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Log hyperparameters
    mlflow.log_param("graphmert_input_dim", input_dim)
    mlflow.log_param("graphmert_embed_dim", embed_dim)
    mlflow.log_param("graphmert_num_heads", num_heads)
    mlflow.log_param("graphmert_num_layers", num_layers)
    mlflow.log_param("graphmert_batch_size", batch_size)
    mlflow.log_param("graphmert_num_epochs", num_epochs)
    mlflow.log_param("graphmert_lr", 1e-3)

    train_graphmert(model, dataloader, optimizer, num_epochs, device)

    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "graphmert_model_weights.pth"
    torch.save(model.state_dict(), weights_path)

    full_model_path = output_dir / "graphmert_entire_model.pth"
    torch.save(model, full_model_path)

    print(f"Saved model weights to {weights_path}")
    print(f"Saved full model to {full_model_path}")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    mlflow_connect()
    experiment_name = get_mlflow_experiment_name()
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"graphmert_train_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        main()
