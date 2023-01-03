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
    return {"Curious about what MCC Code to use?": "Rows: {}".format(len(df)+100)}

@app.get("/path")
async def demo_get():
    return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


@app.post("/path")
async def demo_post(inp: Msg):
    return {"message": inp.msg.upper()}


@app.get("/path/{path_id}")
async def demo_get_path_id(path_id: int):
    return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}