import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from mangum import Mangum
import boto3
import io
import os

# Get data from AWS S3
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

s3 = session.resource('s3', region_name='eu-central-1')
mcc_descriptions = s3.Object(bucket_name='mccapibucket', key='mcc_descriptions.csv')
mcc_weights = s3.Object(bucket_name='mccapibucket', key='mcc_embeddings_mini_bert.pt')

# Convert to regular Python objects
df = pd.read_csv(mcc_descriptions.get()['Body'])
mcc_embeddings = torch.load(io.BytesIO(mcc_weights.get()['Body'].read()))


# Define input and output structures
class QueryPhrase(BaseModel):
    query: str

class MCCMatch(BaseModel):
    mcc_code: str
    name: str
    mcc_match: float
    mcc_description: str

# Get model from sentence_transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# APP
app = FastAPI()

@app.get("/")
def read_root():
    return {"Curious about what MCC Code to use?": "Check out the POST /predict endpoint!"}

@app.post("/predict")
def predict_mcc(payload: QueryPhrase):
    # Embed query in BERT space
    embeddings_query = model.encode([payload.query], convert_to_tensor=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(mcc_embeddings, embeddings_query)
    # Convert tensor to numpy
    match_df = df.copy()
    match_df['matches_to_query'] = cosine_scores.numpy()    
    # Rename columns
    match_df = match_df.rename(columns={'mcc': 'mcc_code', 'short_name': 'name', 'unaltered_description': 'mcc_description', 'matches_to_query': 'mcc_match'})
    # Get top 5
    output_df = match_df[['mcc_code', 'name', 'mcc_match', 'mcc_description',]] \
        .sort_values('mcc_match', ascending=False) \
        .head(5) \
        .astype({'mcc_match': 'float'}) \
        .round({'mcc_match': 5})
  
    return output_df.to_dict('records')