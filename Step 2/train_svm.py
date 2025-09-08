
# train_svm.py (enhanced with log file)
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
import joblib

def pick_feature_cols(df):
    p_cols = [c for c in df.columns if c.startswith("p")]
    if p_cols:
        return sorted(p_cols, key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or 0))
    f_cols = [c for c in df.columns if c.startswith("f")]
    if f_cols:
        return sorted(f_cols, key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or 0))
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]

def main():
    ap = argparse.ArgumentParser(description="Train SVM (RBF) on embedding features with logging.")
    ap.add_argument("--in", dest="in_path", default="balanced_features_pca.tsv")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", default="svm_model.joblib")
    ap.add_argument("--log-file", default="svm_train_info.json")
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

    # --- split ---
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed
    )
    print(f"[INFO] Train size = {Xtr.shape[0]}, Test size = {Xte.shape[0]}")


    # ذخیره مجموعه آموزش و تست در فایل جدا
    train_out = "train_split.tsv"
    test_out  = "test_split.tsv"

    df_train = pd.DataFrame(Xtr, columns=feat_cols)
    df_train.insert(0, "label", ytr)
    df_train.to_csv(train_out, sep="\t", index=False)

    df_test = pd.DataFrame(Xte, columns=feat_cols)
    df_test.insert(0, "label", yte)
    df_test.to_csv(test_out, sep="\t", index=False)

    print(f"[OK] Saved train split → {train_out} | shape={df_train.shape}")
    print(f"[OK] Saved test  split → {test_out}  | shape={df_test.shape}")



    # --- model ---
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=args.seed))
    ])
    svm_pipe.fit(Xtr, ytr)

    # --- predict & metrics ---
    y_prob = svm_pipe.predict_proba(Xte)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(yte, y_pred)
    f1  = f1_score(yte, y_pred)
    auc = roc_auc_score(yte, y_prob)
    ap  = average_precision_score(yte, y_prob)
    cm  = confusion_matrix(yte, y_pred)

    print("\n=== SVM (RBF) ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(yte, y_pred, digits=4))

    # --- save model ---
    joblib.dump(svm_pipe, args.save_model)
    print(f"[OK] Saved model → {args.save_model}")

    # --- save log file ---
    log_data = {
        "input_file": str(args.in_path),
        "train_size": int(Xtr.shape[0]),
        "test_size": int(Xte.shape[0]),
        "test_ratio": args.test_size,
        "seed": args.seed,
        "metrics": {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "pr_auc": ap,
            "confusion_matrix": cm.tolist()
        }
    }
    with open(args.log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[OK] Saved training info → {args.log_file}")

if __name__ == "__main__":
    main()

