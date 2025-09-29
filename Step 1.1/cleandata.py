import pandas as pd
import re
import unicodedata
from pathlib import Path
import sys


BASE_DIR = Path(__file__).parent.resolve()
IN_FILE  = "ChG-Miner_miner-chem-gene.tsv"  
OUT_FILE = "clean_dataset.tsv"

in_path  = (BASE_DIR / IN_FILE)
out_path = (BASE_DIR / OUT_FILE)

if not in_path.exists():
    sys.exit(f"[ERROR] Input file not found: {in_path}")


def normalize_text(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = unicodedata.normalize("NFKC", x)
    x = x.replace("\u200c", "")  # ZWNJ
    x = x.replace("\x00", "")    # NULL
    x = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", x)   
    x = re.sub(r"\s+", " ", x).strip()
    return x

def is_valid_drugbank_id(x: str) -> bool:
    return bool(re.fullmatch(r"DB\d{5,6}", x or ""))

def is_valid_uniprot(x: str) -> bool:
    return bool(
        re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z0-9]{6,10}", x or "")
    )

try:
    df = pd.read_csv(in_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
except Exception:
    df = pd.read_csv(in_path, sep="\t", encoding="latin1", on_bad_lines="skip")

df.columns = [normalize_text(c) for c in df.columns]
rename_map = {}
for c in df.columns:
    c_clean = c.replace(" ", "").replace("#", "").lower()
    if c_clean in {"drug", "drugid"}:
        rename_map[c] = "#Drug"
    elif c_clean in {"gene", "target", "uniprot"}:
        rename_map[c] = "Gene"
if rename_map:
    df = df.rename(columns=rename_map)

required_cols = ["#Drug", "Gene"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    sys.exit(f"[ERROR] Expected columns not found: {missing}. Found: {list(df.columns)}")


n0 = len(df)
for col in required_cols:
    df[col] = df[col].astype(str).map(normalize_text)

df = df.replace({"": pd.NA, "NaN": pd.NA, "nan": pd.NA})
df = df.dropna(subset=required_cols)

mask_valid_drug = df["#Drug"].map(is_valid_drugbank_id)
mask_valid_gene = df["Gene"].map(is_valid_uniprot)
df = df[mask_valid_drug & mask_valid_gene]

n_before_dupes = len(df)
df = df.drop_duplicates(subset=required_cols)
n_after_dupes = len(df)


df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

print(f"Input : {in_path}")
print(f"Output: {out_path}")
print("Rows before:", n0)
print("After NaN/invalid:", n_before_dupes)
print("Removed duplicates:", n_before_dupes - n_after_dupes)
print("Final rows:", len(df))
