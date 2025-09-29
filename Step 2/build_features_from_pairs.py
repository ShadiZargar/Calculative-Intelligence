
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os

def infer_feat_cols(df, preferred_prefix=None):
    cols = list(df.columns)
    def num_key(c, pref):
        s = c[len(pref):]
        return int(''.join(ch for ch in s if ch.isdigit()) or 0)
    if preferred_prefix:
        pref_cols = [c for c in cols if c.startswith(preferred_prefix)]
        if pref_cols:
            return sorted(pref_cols, key=lambda x: num_key(x, preferred_prefix))
    for pref in ["p", "z", "e", "fp", "f"]:
        pref_cols = [c for c in cols if c.startswith(pref)]
        if pref_cols:
            return sorted(pref_cols, key=lambda x: num_key(x, pref))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c != "node"]

def main():
    ap = argparse.ArgumentParser(description="Build pair-wise features by joining pairs with node embeddings.")
    ap.add_argument("--pairs", default="balanced_pairs.tsv", help="TSV with #Drug, Gene, label")
    ap.add_argument("--emb",   default="emb_fused_cat_pca64.tsv", help="TSV with node embeddings (node + p*/z*/...)")
    ap.add_argument("--out",   default="balanced_features_pca.tsv", help="Output TSV with features (f*)")
    ap.add_argument("--prefix", default="p", help="Embedding feature prefix (p for PCA, z for raw fused)")
    args = ap.parse_args()

    print(f"[CWD] {os.getcwd()}")
    print(f"[INFO] pairs = {args.pairs}")
    print(f"[INFO] emb   = {args.emb}")
    print(f"[INFO] out   = {args.out}")
    print(f"[INFO] prefix= {args.prefix}")

    pairs_path = Path(args.pairs)
    emb_path   = Path(args.emb)
    if not pairs_path.exists(): raise SystemExit(f"[ERROR] Not found: {pairs_path.resolve()}")
    if not emb_path.exists():   raise SystemExit(f"[ERROR] Not found: {emb_path.resolve()}")

  
    df_pairs = pd.read_csv(pairs_path, sep="\t")
    df_pairs.columns = [c.strip() for c in df_pairs.columns]
    df_emb   = pd.read_csv(emb_path, sep="\t")
    df_emb.columns = [c.strip() for c in df_emb.columns]

    print(f"[INFO] pairs shape: {df_pairs.shape} | cols: {list(df_pairs.columns)[:6]}")
    print(f"[INFO] emb   shape: {df_emb.shape}   | cols: {list(df_emb.columns)[:6]}")


    drug_col = "#Drug" if "#Drug" in df_pairs.columns else ("drug" if "drug" in df_pairs.columns else None)
    gene_col = "Gene"  if "Gene"  in df_pairs.columns else ("gene" if "gene" in df_pairs.columns else None)
    if drug_col is None or gene_col is None:
        raise SystemExit("[ERROR] pairs file must have '#Drug'/'Gene' (or 'drug'/'gene').")
    if "label" not in df_pairs.columns:
        raise SystemExit("[ERROR] pairs file must contain 'label' (0/1).")
    if "node" not in df_emb.columns:
        raise SystemExit("[ERROR] embeddings file must contain a 'node' column.")

    feat_cols = infer_feat_cols(df_emb, preferred_prefix=args.prefix)
    if not feat_cols:
        raise SystemExit("[ERROR] No embedding feature columns found (prefix p/z/e/fp/f).")
    print(f"[INFO] using {len(feat_cols)} embedding cols; first 5: {feat_cols[:5]}")

    
    E = df_emb.set_index("node")[feat_cols]


    left = df_pairs.join(E, on=drug_col, how="inner", rsuffix="_D")
    dcols = [f"d_{i}" for i in range(len(feat_cols))]
    left.columns = list(left.columns[:3]) + dcols
    print(f"[INFO] after join(drug) : {left.shape}")

    full = left.join(E, on=gene_col, how="inner", rsuffix="_G")
    gcols = [f"g_{i}" for i in range(len(feat_cols))]
    full.columns = list(left.columns) + gcols
    print(f"[INFO] after join(gene) : {full.shape}")

    if full.empty:
        raise SystemExit("[ERROR] After join, no rows left. Check that 'node' IDs match #Drug/Gene values.")

    X = np.hstack([full[dcols].values, full[gcols].values])
    feat_out_cols = [f"f{i}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=feat_out_cols)
    out.insert(0, gene_col, full[gene_col].values)
    out.insert(0, drug_col, full[drug_col].values)
    out.insert(2, "label", full["label"].values)

    out.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] Saved → {args.out} | shape={out.shape} | feat_dim={X.shape[1]}")

if __name__ == "__main__":
    main()
