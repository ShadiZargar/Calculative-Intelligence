import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


df = pd.read_csv(r"/Users/shadizargarzadeh/Desktop/ci final proj/Step 2/balanced_features_pca.tsv", sep="\t")
print("ستون‌ها:", df.columns.tolist())

drop_cols = []
if "#Drug" in df.columns:
    drop_cols.append("#Drug")
if "Gene" in df.columns:
    drop_cols.append("Gene")
if "label" in df.columns:
    y = df["label"].values
    drop_cols.append("label")
else:
    raise ValueError("ستونی به نام 'label' در فایل وجود ندارد!")

X = df.drop(columns=drop_cols).values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]


print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
