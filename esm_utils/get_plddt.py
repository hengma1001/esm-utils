import os

import pandas as pd
from tqdm import tqdm
from utils import get_plddt

esmfold_dir = "/homes/heng.ma/Research/BVBRC/results/esmfold"
openfold_dir = "/homes/heng.ma/Research/BVBRC/results/openfold"

df = pd.read_pickle("esm_emb.pkl")

plddt_list = []
pdb_list = []
for _, line in tqdm(df.iterrows(), total=len(df)):
    pdb_file = f"{esmfold_dir}/{line['name']}.pdb"
    if os.path.exists(pdb_file):
        plddt_list.append(get_plddt(pdb_file))
        pdb_list.append(pdb_file)
    else:
        plddt_list.append(0.0)
        pdb_list.append(0)

df["plddt"] = plddt_list
df["pdb"] = pdb_list

df.to_pickle("esm_emb_plddt.pkl")
