
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

# ----------------------------
# Model
# ----------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, src, dst):
        return self.mlp(torch.cat([src, dst], dim=-1)).view(-1)

# ----------------------------
# Data loading
# ----------------------------
def load_graph_from_pairs(pairs_tsv: Path, is_undirected: bool, one_hot_cap: int = 128) -> Data:
    """
    Expected TSV columns:
      - '#Drug' : source ID (string or int)
      - 'Gene'  : destination ID (string or int)
      - optional 'label' : if present, only rows with label==1 are used to build graph structure.
    Node features are auto-built: [truncated one-hot (cap), degree].
    """
    df = pd.read_csv(pairs_tsv, sep="\\t")
    if "label" in df.columns:
        pos_df = df[df["label"] == 1].copy()
    else:
        pos_df = df.copy()

    # Build node index mapping
    all_nodes = pd.unique(pd.concat([pos_df["#Drug"].astype(str), pos_df["Gene"].astype(str)], ignore_index=True))
    id2idx = {nid: i for i, nid in enumerate(all_nodes)}

    # Edges
    src = torch.tensor([id2idx[str(x)] for x in pos_df["#Drug"]], dtype=torch.long)
    dst = torch.tensor([id2idx[str(x)] for x in pos_df["Gene"]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    if is_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    num_nodes = len(all_nodes)

    # Node features: truncated one-hot + degree
    deg = torch.bincount(edge_index.view(-1), minlength=num_nodes).float().view(-1, 1)
    eye_dim = min(num_nodes, one_hot_cap)
    x_eye = torch.eye(num_nodes)[:, :eye_dim]
    x = torch.cat([x_eye, deg], dim=1)

    data = Data(x=x, edge_index=edge_index)
    data._node_id_list = list(all_nodes)
    return data

# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(encoder, predictor, data, device, threshold: float = 0.5) -> Dict[str, float]:
    encoder.eval(); predictor.eval()
    z = encoder(data.x.to(device), data.edge_index.to(device))
    s, d = data.edge_label_index
    logits = predictor(z[s], z[d]).sigmoid().cpu().numpy()
    labels = data.edge_label.cpu().numpy().astype(int)

    auroc = roc_auc_score(labels, logits)
    auprc = average_precision_score(labels, logits)

    preds = (logits >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(auroc),
        "auprc": float(auprc),
    }
    return metrics

def train_one_split(data: Data,
                    hidden: int,
                    out_dim: int,
                    dropout: float,
                    lr: float,
                    weight_decay: float,
                    epochs: int,
                    patience: int,
                    device: torch.device) -> Tuple[Dict[str, float], Dict[str, float]]:
    splitter = RandomLinkSplit(
        num_val=0.15,
        num_test=0.15,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = splitter(data)

    in_dim = data.x.size(-1)
    encoder = GCNEncoder(in_dim, hidden, out_dim, dropout).to(device)
    predictor = LinkPredictor(out_dim, dropout).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -np.inf
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        encoder.train(); predictor.train()
        z = encoder(train_data.x.to(device), train_data.edge_index.to(device))
        s, d = train_data.edge_label_index
        logits = predictor(z[s], z[d])
        loss = loss_fn(logits, train_data.edge_label.float().to(device))
        opt.zero_grad(); loss.backward(); opt.step()

        # validation
        val_metrics = evaluate(encoder, predictor, val_data, device)
        score = val_metrics["auprc"]  # monitor AUPRC
        if score > best_val:
            best_val = score
            best_state = (encoder.state_dict(), predictor.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        predictor.load_state_dict(best_state[1])

    test_metrics = evaluate(encoder, predictor, test_data, device)
    val_metrics = evaluate(encoder, predictor, val_data, device)
    return val_metrics, test_metrics

# ----------------------------
# Cross-Validation (Repeated random splits)
# ----------------------------
def run_cv(pairs_path: Path,
           k: int,
           hidden: int,
           out_dim: int,
           dropout: float,
           lr: float,
           weight_decay: float,
           epochs: int,
           patience: int,
           one_hot_cap: int,
           is_undirected: bool,
           seed: int,
           out_dir: Path):

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_graph_from_pairs(pairs_path, is_undirected=is_undirected, one_hot_cap=one_hot_cap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    fold_rows = []
    for fold in range(1, k + 1):
        # Change RNG state each fold for a different RandomLinkSplit
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        val_m, test_m = train_one_split(
            data=data,
            hidden=hidden,
            out_dim=out_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            device=device
        )
        row = {"fold": fold}
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row.update({f"test_{k}": v for k, v in test_m.items()})
        fold_rows.append(row)

    df = pd.DataFrame(fold_rows).set_index("fold")

    # Summary over test_* metrics
    test_cols = [c for c in df.columns if c.startswith("test_")]
    summary = df[test_cols].agg(["mean", "std"]).T

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fold_metrics.csv")
    summary.to_csv(out_dir / "summary_metrics.csv")

    # Concise terminal output
    def get(col): return summary.loc[col, "mean"]
    print(f"test_accuracy => {get('test_accuracy'):.4f}")
    print(f"test_precision => {get('test_precision'):.4f}")
    print(f"test_recall => {get('test_recall'):.4f}")
    print(f"test_f1 => {get('test_f1'):.4f}")
    print(f"test_roc_auc => {get('test_roc_auc'):.4f}")
    # (Optional) AUPRC:
    # print(f\"test_auprc => {get('test_auprc'):.4f}\")

def main():
    ap = argparse.ArgumentParser(description="GNN (GCN) link prediction with cross-validation (repeated random splits)")
    ap.add_argument("--pairs_path", type=str, required=True, help="Path to balanced_pairs.tsv")
    ap.add_argument("--k", type=int, default=5, help="Number of repeated splits")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--one_hot_cap", type=int, default=128, help="Cap for truncated one-hot size")
    ap.add_argument("--is_undirected", action="store_true", help="Treat graph as undirected")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./gnn_cv_runs")

    args = ap.parse_args()
    run_cv(
        pairs_path=Path(args.pairs_path),
        k=args.k,
        hidden=args.hidden,
        out_dim=args.out_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        one_hot_cap=args.one_hot_cap,
        is_undirected=args.is_undirected,
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )

if __name__ == "__main__":
    main()
