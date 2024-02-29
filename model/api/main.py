"""Contains REST API definitions built using FastAPI and Uvicorn"""
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pandas import DataFrame
from model.processing import load
from model.processing.train import Train


app = FastAPI()

if __name__ == "__main__":
    df_input = load()
    train = Train(df_input)
    train.train()
    uvicorn.run(app, port=8002)
    
@app.post("/predict")
def predict():
    """API for model inference"""
