import numpy as np
import pandas as pd

import umap

df = pd.read_pickle("esm_emb.pkl")
fit = umap.UMAP()
u = fit.fit_transform(df.embeddings)
