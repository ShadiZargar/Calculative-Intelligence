
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

def pick_feature_cols(df, prefix="z"):
    cols = [c for c in df.columns if c.startswith(prefix)]

    cols = sorted(cols, key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or 0))
    if not cols:
        
        exclude = {"node","drug","gene","label","#Drug","Gene"}
        cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    return cols

def main():
    ap = argparse.ArgumentParser(description="PCA reduction for embeddings with prefix z*.")
    ap.add_argument("--in", dest="in_path", required=True, help="input TSV (emb_fused_cat.tsv / ...)")
    ap.add_argument("--out", dest="out_path", default=None, help="output TSV (default: *_pca.tsv)")
    ap.add_argument("--n-components", type=int, default=None, help="fixed dimension, e.g., 64")
    ap.add_argument("--var", type=float, default=0.95, help="explained variance to keep if n-components not set")
    ap.add_argument("--prefix", default="z", help="feature prefix, default z")
    ap.add_argument("--save-model", default=None, help="optional: save fitted pipeline (joblib)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"[ERROR] File not found: {in_path}")

    df = pd.read_csv(in_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    feat_cols = pick_feature_cols(df, prefix=args.prefix)
    if not feat_cols:
        raise SystemExit("[ERROR] No feature columns found (z*). Use --prefix or check file.")
    X = df[feat_cols].values

   
    if args.n_components is not None:
        pca = PCA(n_components=args.n_components, random_state=42)
    else:
        
        pca = PCA(n_components=args.var, svd_solver="full", random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", pca),
    ])
    Z = pipe.fit_transform(X)
    evr = pipe.named_steps["pca"].explained_variance_ratio_.sum()
    k = Z.shape[1]

    
    out_df = pd.DataFrame(Z, columns=[f"p{i}" for i in range(k)])
    
    meta = [c for c in ["node","drug","#Drug","gene","Gene","label"] if c in df.columns]
    for c in meta[::-1]:
        out_df.insert(0, c, df[c])

    out_path = Path(args.out_path) if args.out_path else in_path.with_name(in_path.stem + "_pca.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved → {out_path} | shape={out_df.shape} | explained_variance={evr:.4f}")

    if args.save_model:
        joblib.dump(pipe, args.save_model)
        print(f"[OK] Saved model → {args.save_model}")

if __name__ == "__main__":
    main()
