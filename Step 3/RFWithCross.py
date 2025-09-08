import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold


df = pd.read_csv(r"/Users/shadizargarzadeh/Desktop/ci final proj/Step 3/balanced_features_pca.tsv", sep="\t")


X = df.drop(columns=["#Drug", "Gene", "label"]).values
y = df["label"].values

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    class_weight="balanced"
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    rf, X, y, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=False
)


for metric in scores:
    if metric.startswith("test_"):
        print(metric, "=>", scores[metric].mean(), "+/-", scores[metric].std())
