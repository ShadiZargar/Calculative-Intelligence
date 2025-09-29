
import pandas as pd
import re

smiles_csv = "df_drugbank_smiles.csv"          
edges_tsv  = "ChG-Miner_miner-chem-gene.tsv"   
out_tsv    = "drug_gene_with_smiles.tsv"       


df_smiles = pd.read_csv(smiles_csv)
df_edges  = pd.read_csv(edges_tsv, sep="\t")


df_smiles.columns = [c.strip() for c in df_smiles.columns]
df_edges.columns  = [c.strip() for c in df_edges.columns]

if "drugbank_id" not in df_smiles.columns:
    raise ValueError("ستون 'drugbank_id' در فایل SMILES پیدا نشد.")
if "SMILES" not in df_smiles.columns and "smiles" in df_smiles.columns:
    df_smiles = df_smiles.rename(columns={"smiles": "SMILES"})

if "#Drug" not in df_edges.columns:
    raise ValueError("ستون '#Drug' در فایل یال‌ها پیدا نشد.")


def norm_dbid(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    ext = s.str.extract(r"(DB\d{5})", expand=False)
    return ext.where(ext.notna(), s)

df_smiles["DrugBankID"] = norm_dbid(df_smiles["drugbank_id"])
df_edges["DrugBankID"]  = norm_dbid(df_edges["#Drug"])


df_smiles = df_smiles.drop_duplicates("DrugBankID", keep="first")

df_merged = df_edges.merge(
    df_smiles[["DrugBankID", "SMILES"]],
    on="DrugBankID",
    how="left"
)


df_merged.to_csv(out_tsv, sep="\t", index=False, na_rep="NaN")

print(f"[OK] Saved → {out_tsv}  |  shape: {df_merged.shape}")
print(df_merged.head())
