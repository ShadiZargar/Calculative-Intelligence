import pandas as pd
from pathlib import Path
import itertools


in_file = Path("ChG-Miner_miner-chem-gene.tsv")


df = pd.read_csv(in_file, sep="\t")
df.columns = [c.strip() for c in df.columns]   
drug_col = "#Drug" if "#Drug" in df.columns else df.columns[0]
gene_col = "Gene"  if "Gene"  in df.columns else df.columns[1]

n_pos = len(df)

n_drugs = df[drug_col].nunique()
n_genes = df[gene_col].nunique()


n_total_pairs = n_drugs * n_genes

n_neg = n_total_pairs - n_pos

print(f"تعداد داروها: {n_drugs}")
print(f"تعداد ژن‌ها: {n_genes}")
print(f"تعداد مثبت‌ها (از دیتاست): {n_pos}")
print(f"تعداد منفی‌ها (محاسبه‌شده): {n_neg}")
print(f"نسبت مثبت به منفی: {n_pos} / {n_neg} ≈ {n_pos/n_neg:.5f}")
