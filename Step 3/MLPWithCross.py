
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def parse_hidden_layers(s: str) -> List[int]:
    if not s:
        return [256, 128]
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def build_mlp(input_dim: int, hidden: List[int], dropout: float, lr: float, seed: int = 42) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  
    model.add(Dense(hidden[0], activation="relu"))
    if dropout > 0:
        model.add(Dropout(dropout))
    for h in hidden[1:]:
        model.add(Dense(h, activation="relu"))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def load_features(path: Path, drop_cols: List[str], label_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".tsv", ".txt", ""]:
        df = pd.read_csv(path, sep="\\t")
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path, sep="\\t")

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop).values
    y = df[label_col].values
    feature_names = [c for c in df.columns if c not in cols_to_drop]
    return X, y, feature_names

def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics

def run_cv(features_path: Path,
           label_col: str = "label",
           drop_cols: List[str] = None,
           k: int = 5,
           epochs: int = 50,
           batch_size: int = 32,
           patience: int = 5,
           lr: float = 1e-3,
           dropout: float = 0.2,
           hidden_layers: List[int] = None,
           seed: int = 42,
           out_dir: Path = None):

    if drop_cols is None:
        drop_cols = ["#Drug", "Gene", label_col]
    if hidden_layers is None:
        hidden_layers = [256, 128]
    if out_dir is None:
        out_dir = features_path.with_suffix("").parent / "cv_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, feat_names = load_features(features_path, drop_cols=drop_cols, label_col=label_col)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_rows = []
    oof_prob = np.zeros_like(y, dtype=float)
    oof_pred = np.zeros_like(y, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_mlp(X_train.shape[1], hidden_layers, dropout, lr, seed + fold)

        early = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early],
            verbose=0
        )

        y_prob = model.predict(X_test, verbose=0).ravel()
        y_hat = (y_prob >= 0.5).astype(int)

        oof_prob[test_idx] = y_prob
        oof_pred[test_idx] = y_hat

        m = compute_metrics(y_test, y_prob, y_hat)
        m["fold"] = fold
        fold_rows.append(m)

    df_folds = pd.DataFrame(fold_rows).set_index("fold")
    summary = df_folds.agg(["mean", "std"]).T

    df_folds.to_csv(out_dir / "fold_metrics.csv")
    summary.to_csv(out_dir / "summary_metrics.csv")
    pd.DataFrame({"oof_prob": oof_prob, "oof_pred": oof_pred, "y_true": y}).to_csv(out_dir / "oof_predictions.csv", index=False)

    cfg = {
        "features_path": str(features_path),
        "label_col": label_col,
        "drop_cols": drop_cols,
        "k": k,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "lr": lr,
        "dropout": dropout,
        "hidden_layers": hidden_layers,
        "seed": seed,
        "out_dir": str(out_dir),
    }
    Path(out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"test_accuracy => {summary.loc['accuracy','mean']:.4f}")
    print(f"test_precision => {summary.loc['precision','mean']:.4f}")
    print(f"test_recall => {summary.loc['recall','mean']:.4f}")
    print(f"test_f1 => {summary.loc['f1','mean']:.4f}")
    print(f"test_roc_auc => {summary.loc['roc_auc','mean']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="K-Fold CV for Keras MLP on PCA embedding file")
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to features file (TSV/CSV/Parquet) containing label column.")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--drop_cols", type=str, default="#Drug,Gene,label",
                        help="Comma-separated columns to drop (if present).")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_layers", type=str, default="256,128",
                        help="Comma-separated sizes, e.g., '256,128'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="")

    args = parser.parse_args()

    features_path = Path(args.features_path)
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    hidden = parse_hidden_layers(args.hidden_layers)
    out_dir = Path(args.out_dir) if args.out_dir else None

    run_cv(features_path=features_path,
           label_col=args.label_col,
           drop_cols=drop_cols,
           k=args.k,
           epochs=args.epochs,
           batch_size=args.batch_size,
           patience=args.patience,
           lr=args.lr,
           dropout=args.dropout,
           hidden_layers=hidden,
           seed=args.seed,
           out_dir=out_dir)

if __name__ == "__main__":
    main()
