# train_svm_cv.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def pick_feature_cols(df):
    p_cols = [c for c in df.columns if c.startswith("p")]
    if p_cols:
        return sorted(p_cols, key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or 0))
    f_cols = [c for c in df.columns if c.startswith("f")]
    if f_cols:
        return sorted(f_cols, key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or 0))
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]

def main():
    ap = argparse.ArgumentParser(description="Train SVM (RBF) with cross validation.")
    ap.add_argument("--in", dest="in_path", default="balanced_features_pca.tsv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-file", default="svm_cv_info.json")
    args = ap.parse_args()

    # --- read data ---
    df = pd.read_csv(Path(args.in_path), sep="\t")
    if "label" not in df.columns:
        raise SystemExit("[ERROR] input must contain 'label' (0/1).")
    feat_cols = pick_feature_cols(df)
    if not feat_cols:
        raise SystemExit("[ERROR] no feature columns (p* or f*).")
    X = df[feat_cols].values
    y = df["label"].astype(int).values

    # --- define pipeline ---
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=args.seed
        ))
    ])

    # --- cross validation ---
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = cross_validate(
        svm_pipe, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc", "average_precision"],
        return_train_score=False
    )

    # --- print results ---
    print("\n=== SVM (RBF) with Cross Validation ===")
    for metric in scores:
        if metric.startswith("test_"):
            mean = scores[metric].mean()
            std  = scores[metric].std()
            print(f"{metric} => {mean:.4f} +/- {std:.4f}")

    # --- save log ---
    log_data = {m: {"mean": float(scores[m].mean()), "std": float(scores[m].std())}
                for m in scores if m.startswith("test_")}
    with open(args.log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[OK] Saved cross-validation info → {args.log_file}")

if __name__ == "__main__":
    main()