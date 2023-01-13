import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("mcc_descriptions.csv", dtype=str)

embeddings_descriptions = model.encode(list(df['description'].astype(str)), convert_to_tensor=True)

torch.save(embeddings_descriptions, 'mcc_embeddings_mini_bert.pt')