import glob

import pandas as pd
import torch
from tqdm import tqdm

from utils import batcher, get_label_seq

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
fasta_files = glob.glob("/lambda_stor/homes/heng.ma/Research/BVBRC/seqs/*.fa")
fasta_data = [(get_label_seq(fasta_file)) for fasta_file in fasta_files[:]]
batch_size = 50

df = []
for data in tqdm(
    batcher(fasta_data, batch_size), total=len(fasta_data) // batch_size + 1
):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    for i, tokens_len in enumerate(batch_lens):
        seq_rep = token_representations[i, 1 : tokens_len - 1].mean(0)
        df.append(
            {
                "name": batch_labels[i],
                "embedding": list(seq_rep.cpu().numpy()),
                "sequence": str(data[i][1]),
            }
        )

df = pd.DataFrame(df)
df.to_pickle("esm_emb.pkl")
