import pandas as pd
from pathlib import Path
import itertools

# فایل اولیه (دارو-ژن مثبت‌ها)
in_file = Path("ChG-Miner_miner-chem-gene.tsv")

# خواندن
df = pd.read_csv(in_file, sep="\t")
df.columns = [c.strip() for c in df.columns]   # پاک کردن اسپیس
drug_col = "#Drug" if "#Drug" in df.columns else df.columns[0]
gene_col = "Gene"  if "Gene"  in df.columns else df.columns[1]

# تعداد مثبت‌ها (همون ردیف‌های موجود)
n_pos = len(df)

# کل تعداد داروها و ژن‌ها
n_drugs = df[drug_col].nunique()
n_genes = df[gene_col].nunique()

# کل فضای ممکن دارو-ژن (ضرب کارتیزین)
n_total_pairs = n_drugs * n_genes

# منفی‌ها = کل جفت‌های ممکن - مثبت‌ها
n_neg = n_total_pairs - n_pos

print(f"تعداد داروها: {n_drugs}")
print(f"تعداد ژن‌ها: {n_genes}")
print(f"تعداد مثبت‌ها (از دیتاست): {n_pos}")
print(f"تعداد منفی‌ها (محاسبه‌شده): {n_neg}")
print(f"نسبت مثبت به منفی: {n_pos} / {n_neg} ≈ {n_pos/n_neg:.5f}")
