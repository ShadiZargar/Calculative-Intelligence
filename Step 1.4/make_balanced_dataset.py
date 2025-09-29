
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import random

def read_edges(path: Path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    drug_col = "#Drug" if "#Drug" in df.columns else df.columns[0]
    gene_col = "Gene"  if "Gene"  in df.columns else df.columns[1]
    df = df[[drug_col, gene_col]].astype(str).dropna()
    df = df.drop_duplicates()
    return df, drug_col, gene_col

def sample_negatives(drugs, genes, pos_pairs_set, n_neg, seed=42, max_trials=50_000_000):
    rng = random.Random(seed)
    neg = set()
    D, G = list(drugs), list(genes)
    trials = 0
    while len(neg) < n_neg and trials < max_trials:
        d = rng.choice(D)
        g = rng.choice(G)
        pair = (d, g)
        if (pair not in pos_pairs_set) and (pair not in neg):
            neg.add(pair)
        trials += 1
    if len(neg) < n_neg:
        print(f"[WARN] Requested {n_neg} negatives but only got {len(neg)} (likely very dense).")
    return list(neg)

def build_feature_table(pairs_df: pd.DataFrame, emb_df: pd.DataFrame, drug_col: str, gene_col: str):
 
    z_cols = [c for c in emb_df.columns if c != "node"]
    dims = len(z_cols)

    d_map = emb_df.set_index("node")[z_cols]
    g_map = d_map

    left = pairs_df.join(d_map, on=drug_col, how="inner", rsuffix="_D")
    left.columns = list(left.columns[:3]) + [f"d_{i}" for i in range(dims)]
    full = left.join(g_map, on=gene_col, how="inner", rsuffix="_G")
    full.columns = list(full.columns[:(3+dims)]) + [f"g_{i}" for i in range(dims)]
    if full.empty:
        print("[WARN] No overlapping embeddings for pairs; features table will be empty.")
        return full


    d_cols = [f"d_{i}" for i in range(dims)]
    g_cols = [f"g_{i}" for i in range(dims)]
    Xd = full[d_cols].values
    Xg = full[g_cols].values
    X  = np.hstack([Xd, Xg])

    feat_cols = [f"f{i}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=feat_cols)
    out.insert(0, gene_col, full[gene_col].values)
    out.insert(0, drug_col, full[drug_col].values)
    out.insert(2, "label", full["label"].values)
    return out

def main():
    ap = argparse.ArgumentParser(description="Build a balanced positive/negative dataset for drug-gene.")
    ap.add_argument("--edges", default="ChG-Miner_miner-chem-gene.tsv", help="TSV with #Drug, Gene positives")
    ap.add_argument("--emb", default="emb_final.tsv", help="(optional) node embeddings TSV with columns: node, z0..")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="negatives per positive (e.g., 1.0 ⇒ 1:1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-pairs", default="balanced_pairs.tsv")
    ap.add_argument("--out-feats", default="balanced_features.tsv")
    args = ap.parse_args()

    edges_path = Path(args.edges)
    emb_path   = Path(args.emb)
    out_pairs  = Path(args.out_pairs)
    out_feats  = Path(args.out_feats)


    df_pos, drug_col, gene_col = read_edges(edges_path)
    n_pos = len(df_pos)
    drugs = df_pos[drug_col].unique()
    genes = df_pos[gene_col].unique()
    pos_set = set(map(tuple, df_pos[[drug_col, gene_col]].itertuples(index=False, name=None)))

    print(f"[INFO] Drugs={len(drugs)}, Genes={len(genes)}, Positives={n_pos}")


    n_neg_target = int(round(args.neg_ratio * n_pos))
    print(f"[INFO] Sampling negatives: {n_neg_target} (ratio={args.neg_ratio}) …")
    neg_pairs = sample_negatives(drugs, genes, pos_set, n_neg_target, seed=args.seed)
    df_neg = pd.DataFrame(neg_pairs, columns=[drug_col, gene_col])
    df_neg["label"] = 0


    df_pos_lab = df_pos.copy()
    df_pos_lab["label"] = 1
    pairs = pd.concat([df_pos_lab, df_neg], axis=0, ignore_index=True)
    pairs = pairs.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  
    pairs.to_csv(out_pairs, sep="\t", index=False)
    print(f"[OK] Saved pairs → {out_pairs} | shape={pairs.shape} | pos={n_pos}, neg={len(df_neg)}")

   
    if emb_path.exists():
        emb = pd.read_csv(emb_path, sep="\t")
        emb.columns = [c.strip() for c in emb.columns]
        if "node" not in emb.columns:
            print("[WARN] 'emb_final.tsv' has no 'node' column; skip features.")
            return

        z_cols = [c for c in emb.columns if c != "node"]

        emb[z_cols] = emb[z_cols].apply(pd.to_numeric, errors="coerce")
        feats = build_feature_table(pairs, emb, drug_col, gene_col)
        if len(feats):
            feats.to_csv(out_feats, sep="\t", index=False)
            print(f"[OK] Saved features → {out_feats} | shape={feats.shape}")
        else:
            print("[WARN] Features table is empty; check that 'node' ids in emb_final match #Drug and Gene values.")
    else:
        print(f"[INFO] Embeddings file not found ({emb_path}); skipped feature construction.")

if __name__ == "__main__":
    main()
