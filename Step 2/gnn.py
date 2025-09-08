import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

# ======== تنظیمات ساده ========
PAIRS_TSV = os.path.join(os.path.dirname(__file__), "balanced_pairs.tsv")
HIDDEN = 128
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
# اگر گراف رو بدون جهت می‌دونی، True بذار
IS_UNDIRECTED = True

# ======== مدل‌ها ========
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim*2, in_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(in_dim, 1),
        )
    def forward(self, src, dst):
        return self.mlp(torch.cat([src, dst], dim=-1)).view(-1)

# ======== داده را بساز ========
def load_graph_from_pairs(pairs_tsv):
    df = pd.read_csv(pairs_tsv, sep="\t")
    # اگر ستون label داری، فقط مثبت‌ها را برای ساخت گراف نگه دار
    if "label" in df.columns:
        pos_df = df[df["label"] == 1].copy()
    else:
        pos_df = df.copy()

    # نگاشت شناسه‌های دارو/ژن به ایندکس گره
    # سادگی: یک فضا برای همهٔ نودها می‌سازیم (دارو+ژن)
    # id را تبدیل به str می‌کنیم تا برخورد پیش نیاد
    all_nodes = pd.unique(pd.concat([pos_df["#Drug"].astype(str),
                                     pos_df["Gene"].astype(str)], ignore_index=True))
    id2idx = {nid: i for i, nid in enumerate(all_nodes)}

    # edge_index
    src = torch.tensor([id2idx[str(x)] for x in pos_df["#Drug"]], dtype=torch.long)
    dst = torch.tensor([id2idx[str(x)] for x in pos_df["Gene"]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    if IS_UNDIRECTED:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    num_nodes = len(all_nodes)

    # ویژگی نودها: اگر ویژگی واقعی نداری، One-hot کم‌حجم + degree
    deg = torch.bincount(edge_index.view(-1), minlength=num_nodes).float().view(-1, 1)
    eye_dim = min(num_nodes, 128)
    x_eye = torch.eye(num_nodes)[:, :eye_dim]
    x = torch.cat([x_eye, deg], dim=1)

    data = Data(x=x, edge_index=edge_index)
    data._node_id_list = list(all_nodes)  # برای نگاشت خروجی به شناسه‌های اصلی
    return data

# ======== حلقه‌های آموزش/ارزیابی با RandomLinkSplit ========
@torch.no_grad()
def evaluate(encoder, predictor, data, device):
    encoder.eval(); predictor.eval()
    z = encoder(data.x.to(device), data.edge_index.to(device))

    src, dst = data.edge_label_index
    logits = predictor(z[src], z[dst]).sigmoid().cpu().numpy()
    labels = data.edge_label.cpu().numpy()

    auroc = roc_auc_score(labels, logits)
    auprc = average_precision_score(labels, logits)
    return auroc, auprc, logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph_from_pairs(PAIRS_TSV)

    # Split جدید: خودش نگتیوها را می‌سازد و در فیلدهای edge_label(_index) می‌گذارد
    splitter = RandomLinkSplit(
        num_val=0.15, num_test=0.15,
        is_undirected=IS_UNDIRECTED,
        add_negative_train_samples=True,  # خیلی مهم: برای train هم نگتیو بسازد
        neg_sampling_ratio=1.0,           # به ازای هر مثبت یک منفی
    )
    train_data, val_data, test_data = splitter(data)

    in_dim = data.x.size(-1)
    encoder = GCNEncoder(in_dim, HIDDEN, HIDDEN).to(device)
    predictor = LinkPredictor(HIDDEN).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        encoder.train(); predictor.train()
        opt.zero_grad()

        z = encoder(train_data.x.to(device), train_data.edge_index.to(device))
        s, d = train_data.edge_label_index
        logits = predictor(z[s], z[d])
        loss = loss_fn(logits, train_data.edge_label.float().to(device))
        loss.backward(); opt.step()

        va_roc, va_pr, _ = evaluate(encoder, predictor, val_data, device)
        if va_pr > best_val:
            best_val = va_pr
            best_state = (encoder.state_dict(), predictor.state_dict())

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val AUPRC={va_pr:.4f} AUROC={va_roc:.4f}")

    # ارزیابی نهایی روی تست
    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        predictor.load_state_dict(best_state[1])

    te_roc, te_pr, te_scores = evaluate(encoder, predictor, test_data, device)
    print(f"TEST — AUPRC={te_pr:.4f} | AUROC={te_roc:.4f}")

    # خروجی پیش‌بینی‌های تست
    # لبه‌های تست شامل مثبت و منفی است؛ ما فقط امتیاز و لیبل را ذخیره می‌کنیم
    nid_list = data._node_id_list
    s, d = test_data.edge_label_index
    s = s.cpu().numpy(); d = d.cpu().numpy()
    lbl = test_data.edge_label.cpu().numpy()
    out_df = pd.DataFrame({
        "src_id": [nid_list[i] for i in s],
        "dst_id": [nid_list[j] for j in d],
        "score": te_scores,
        "label": lbl
    })
    out_df.to_csv("predictions_test.csv", index=False)
    print("Saved predictions to predictions_test.csv")

if __name__ == "__main__":
    main()
