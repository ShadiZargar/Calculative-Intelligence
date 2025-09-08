# node2vec_minimal.py
import argparse
from pathlib import Path
import random
import pandas as pd
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Random walks (DeepWalk / Node2Vec with p=q=1) ----------
def random_walk(G, start, walk_length, rng):
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        walk.append(rng.choice(nbrs))
    return walk

def generate_walks(G, num_walks=10, walk_length=5, seed=42):  # ← تغییر به 5
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for n in nodes:
            w = random_walk(G, n, walk_length, rng)
            # به صورت str برای gensim
            walks.append([str(x) for x in w])
    return walks

# ---------- Load bipartite graph ----------
def load_bipartite_graph(tsv_path: Path) -> nx.Graph:
    df = pd.read_csv(tsv_path, sep="\t")
    # normalization of column names
    cols = {c.strip().replace("#","").lower(): c for c in df.columns}
    drug_col = cols.get("drug", "#Drug")
    gene_col = cols.get("gene", "Gene")
    df = df[[drug_col, gene_col]].rename(columns={drug_col:"#Drug", gene_col:"Gene"}).astype(str)

    G = nx.Graph()
    drugs = df["#Drug"].unique()
    genes = df["Gene"].unique()
    G.add_nodes_from(drugs, kind="drug")
    G.add_nodes_from(genes, kind="gene")
    G.add_edges_from(df.itertuples(index=False, name=None))
    return G

# ---------- Train embeddings ----------
def train_embeddings(walks, dimensions=128, window=10, min_count=1, workers=4, epochs=5, sg=1, seed=42):
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,             # skip-gram
        negative=5,
        epochs=epochs,
        seed=seed
    )
    return model

def save_embeddings(model, G, out_path: Path):
    nodes = list(G.nodes())
    vecs = []
    for n in nodes:
        key = str(n)
        if key in model.wv:
            vecs.append(model.wv[key])
        else:
            vecs.append(np.zeros(model.vector_size, dtype=np.float32))
    emb = pd.DataFrame(vecs, index=nodes)
    emb.insert(0, "node", nodes)
    emb.insert(1, "type", [G.nodes[n].get("kind","?") for n in nodes])
    emb.insert(2, "degree", [G.degree[n] for n in nodes])
    emb.to_csv(out_path, sep="\t", index=False)
    return emb

def cosine_topk(df_emb: pd.DataFrame, query: str, k=10, same_type=False):
    if query not in set(df_emb["node"]):
        raise SystemExit(f"[ERROR] Node not found: {query}")
    X = df_emb.drop(columns=["node","type","degree"]).values
    idx = df_emb.index[df_emb["node"] == query][0]
    v = X[idx:idx+1]
    sims = cosine_similarity(v, X)[0]
    s = pd.Series(sims, index=df_emb["node"])
    s = s.drop(index=query, errors="ignore").sort_values(ascending=False)
    if same_type:
        t = df_emb.loc[idx, "type"]
        keep = df_emb.set_index("node").query("type == @t").index
        s = s[s.index.isin(keep)]
    return s.head(k)

def main():
    ap = argparse.ArgumentParser(description="Lightweight Node2Vec-like embeddings (p=q=1) without extra deps.")
    ap.add_argument("--in", dest="in_path", default="ChG-Miner_miner-chem-gene.tsv")
    ap.add_argument("--out-emb", dest="out_emb", default="embeddings.tsv")
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--walk-length", type=int, default=5)   # ← تغییر به 5
    ap.add_argument("--num-walks", type=int, default=10)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--query", nargs="*", default=[])
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"[ERROR] File not found: {in_path}")

    print("[1/3] Loading graph …")
    G = load_bipartite_graph(in_path)
    print(f"   Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

    print("[2/3] Generating random walks …")
    walks = generate_walks(G, num_walks=args.num_walks, walk_length=args.walk_length)
    print(f"   Walks: {len(walks):,}")

    print("[3/3] Training Word2Vec …")
    model = train_embeddings(
        walks, dimensions=args.dims, window=args.window, epochs=args.epochs
    )

    out_path = Path(args.out_emb).resolve()
    emb = save_embeddings(model, G, out_path)
    print(f"[OK] Saved embeddings → {out_path}  | shape: {emb.shape}")

    for q in args.query:
        try:
            res = cosine_topk(emb, q, k=10, same_type=False)
            print(f"\nTop-10 similar to {q}:")
            for n, sc in res.items():
                print(f"{n}\t{sc:.4f}")
        except SystemExit as e:
            print(e)

if __name__ == "__main__":
    main()
