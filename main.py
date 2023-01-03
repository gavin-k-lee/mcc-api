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

app = FastAPI()

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

class Msg(BaseModel):
    msg: str

# APP
@app.get("/")
def read_root():
    return {"Curious about what MCC Code to use?": "Search now at the /predict endpoint!"}

@app.get("/path")
async def demo_get():
    return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


@app.post("/path")
async def demo_post(inp: Msg):
    return {"message": inp.msg.upper()}


@app.get("/path/{path_id}")
async def demo_get_path_id(path_id: int):
    return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}