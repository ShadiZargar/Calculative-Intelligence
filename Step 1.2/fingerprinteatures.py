import sys
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from gensim.models import Word2Vec
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


EDGES_TSV       = Path("ChG-Miner_miner-chem-gene.tsv") 
MERGED_TSV      = Path("drug_gene_with_smiles.tsv")       

OUT_EMB         = Path("emb_deepwalk.tsv")                
OUT_FP          = Path("drug_fp.tsv")                  
OUT_FUSED_SUM   = Path("emb_fused_sum.tsv")             
OUT_FUSED_CAT   = Path("emb_fused_cat.tsv")            

# DeepWalk / Word2Vec
DIMS_GRAPH   = 128
WALK_LENGTH  = 5
NUM_WALKS    = 10
WINDOW       = 10
EPOCHS       = 5
WORKERS      = 4
MIN_COUNT    = 1

# Fingerprint
FP_BITS      = 2048
FP_RADIUS    = 2              

# Fusion
ALPHA        = 0.7            


GLOBAL_SEED  = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=FutureWarning)


def load_edges(tsv_path: Path):
    if not tsv_path.exists():
        raise SystemExit(f"[ERROR] File not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    drug_col = "#Drug" if "#Drug" in df.columns else df.columns[0]
    gene_col = "Gene"  if "Gene"  in df.columns else df.columns[1]
    df = df[[drug_col, gene_col]].astype(str)
    return df, drug_col, gene_col

def build_graph(df_edges, drug_col, gene_col):
    G = nx.Graph()
    G.add_edges_from(df_edges[[drug_col, gene_col]].itertuples(index=False, name=None))
    return G

def _random_walk(G: nx.Graph, start, walk_length, rng: random.Random):
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        walk.append(rng.choice(nbrs))
    return walk

def generate_walks(G: nx.Graph, walk_length=5, num_walks=10, seed=42):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for n in nodes:
            w = _random_walk(G, n, walk_length, rng)
            walks.append([str(x) for x in w]) 
    return walks

def train_deepwalk_embeddings(G: nx.Graph) -> pd.DataFrame:
    walks = generate_walks(G, walk_length=WALK_LENGTH, num_walks=NUM_WALKS, seed=GLOBAL_SEED)
    model = Word2Vec(
        sentences=walks,
        vector_size=DIMS_GRAPH,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,          
        negative=5,
        epochs=EPOCHS,
        seed=GLOBAL_SEED
    )
   
    nodes = list(G.nodes())
    vecs = []
    for n in nodes:
        k = str(n)
        if k in model.wv:
            vecs.append(model.wv[k])
        else:
            vecs.append(np.zeros(DIMS_GRAPH, dtype=np.float32))
    emb = pd.DataFrame(vecs, index=nodes, columns=[f"e{i}" for i in range(DIMS_GRAPH)])
    emb.insert(0, "node", nodes)
    return emb

def load_merged_smiles(tsv_path: Path, drug_col: str):
    if not tsv_path.exists():
        raise SystemExit(f"[ERROR] File not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if drug_col not in df.columns:
        raise SystemExit(f"[ERROR] Column '{drug_col}' not found in {tsv_path.name}")
    if "SMILES" not in df.columns:
        if "smiles" in df.columns:
            df = df.rename(columns={"smiles": "SMILES"})
        else:
            raise SystemExit(f"[ERROR] Column 'SMILES' not found in {tsv_path.name}")
    df_sm = df[[drug_col, "SMILES"]].drop_duplicates(subset=[drug_col])
    return df_sm

def smiles_to_fp(smiles: str, nBits=2048, radius=2):
    try:
        if pd.isna(smiles):
            return None
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

def build_fp_table(df_sm: pd.DataFrame, drug_col: str, fp_bits=2048, radius=2) -> pd.DataFrame:
    rows = []
    for _, row in df_sm.iterrows():
        dbid = str(row[drug_col]).strip()
        s = row["SMILES"]
        arr = smiles_to_fp(s, nBits=fp_bits, radius=radius)
        if arr is None:
            arr = np.zeros((fp_bits,), dtype=np.float32) 
        rows.append((dbid, arr))
    fp_mat = np.vstack([r[1] for r in rows])
    ids = [r[0] for r in rows]
    df_fp = pd.DataFrame(fp_mat, columns=[f"fp{i}" for i in range(fp_mat.shape[1])])
    df_fp.insert(0, "node", ids)
    return df_fp


def fuse_embeddings(emb_graph: pd.DataFrame, df_fp: pd.DataFrame,
                    alpha=0.7, dims_graph=128,
                    out_sum=OUT_FUSED_SUM, out_cat=OUT_FUSED_CAT):
    """
    Join on 'node'; for genes FP is missing → filled with zeros.
    Outputs two files: sum (same dim) & concat (double dim).
    """
    df_all = emb_graph.merge(df_fp, on="node", how="left")

    vec_cols_graph = sorted([c for c in df_all.columns if c.startswith("e")], key=lambda x: int(x[1:]))
    vec_cols_fp    = sorted([c for c in df_all.columns if c.startswith("fp")], key=lambda x: int(x[2:]))

    Xg  = df_all[vec_cols_graph].values
    Xfp = df_all[vec_cols_fp].fillna(0.0).values if vec_cols_fp else np.zeros((len(df_all), FP_BITS), dtype=np.float32)

    Xfp_std = StandardScaler(with_mean=True, with_std=True).fit_transform(Xfp)
    pca = PCA(n_components=dims_graph, random_state=GLOBAL_SEED)
    Xfp_pca = pca.fit_transform(Xfp_std)


    X_sum = alpha * Xg + (1 - alpha) * Xfp_pca
    X_cat = np.hstack([Xg, Xfp_pca])

    X_sum = normalize(X_sum)
    X_cat = normalize(X_cat)

    out_sum_df = pd.DataFrame(X_sum, columns=[f"z{i}" for i in range(X_sum.shape[1])])
    out_sum_df.insert(0, "node", df_all["node"])
    out_sum_df.to_csv(out_sum, sep="\t", index=False, na_rep="NaN")

    out_cat_df = pd.DataFrame(X_cat, columns=[f"z{i}" for i in range(X_cat.shape[1])])
    out_cat_df.insert(0, "node", df_all["node"])
    out_cat_df.to_csv(out_cat, sep="\t", index=False, na_rep="NaN")

    return out_sum_df, out_cat_df


def main():
    print("[1/4] Loading edges and building graph …")
    df_edges, drug_col, gene_col = load_edges(EDGES_TSV)
    G = build_graph(df_edges, drug_col, gene_col)
    print(f"   Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

    print("[2/4] Training DeepWalk (gensim 4) …")
    emb_graph = train_deepwalk_embeddings(G)
    emb_graph.to_csv(OUT_EMB, sep="\t", index=False, na_rep="NaN")
    print(f"   Saved graph embeddings → {OUT_EMB}  | shape: {emb_graph.shape}")

    print("[3/4] Building fingerprints from SMILES …")
    df_sm = load_merged_smiles(MERGED_TSV, drug_col=drug_col)
    df_fp = build_fp_table(df_sm, drug_col=drug_col, fp_bits=FP_BITS, radius=FP_RADIUS)
    df_fp.to_csv(OUT_FP, sep="\t", index=False, na_rep="NaN")
    # coverage report
    nz = (df_fp.drop(columns=["node"]).sum(axis=1) > 0).sum()
    print(f"   Saved drug fingerprints → {OUT_FP}  | shape: {df_fp.shape} | nonzero FPs: {nz}/{len(df_fp)}")

    print("[4/4] Fusing embeddings (weighted-sum & concat) …")
    fused_sum, fused_cat = fuse_embeddings(
        emb_graph, df_fp, alpha=ALPHA, dims_graph=DIMS_GRAPH,
        out_sum=OUT_FUSED_SUM, out_cat=OUT_FUSED_CAT
    )
    print(f"[OK] Saved fused → {OUT_FUSED_SUM} (sum) | {OUT_FUSED_CAT} (concat)")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        m = str(e)
        if "rdkit" in m.lower():
            sys.stderr.write(
                "[ERROR] RDKit not found. In conda env run:\n"
                "  conda install -c conda-forge rdkit\n"
            )
        elif "gensim" in m.lower():
            sys.stderr.write(
                "[ERROR] gensim not found. In conda env run:\n"
                "  conda install -c conda-forge gensim=4.3.3\n"
            )
        else:
            sys.stderr.write(f"[ERROR] Missing module: {m}\n")
        sys.exit(1)


