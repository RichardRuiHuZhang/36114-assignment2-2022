#uvicorn main:app --host 127.0.0.1 --port 8080
#Test name Crow Peak Brewing

from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

#API description
API_text = 'World'

api_descripion = """
    Body of API Description text

"""

#Load Model
model_pipeline = load('models/test01.joblib')
#model_pipeline = load('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/test02.joblib')
#Load Target value transformer
target_transformer = load('models/target_decoder.joblib')
#target_transformer = load('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/target_decoder.joblib')

# class beer_data_multiple(BaseModel):
#     brewery: list[str]
#     aroma: list[float]                     
#     appearance: list[float] 
#     palate: list[float] 
#     taste: list[float] 
#     alcohol: list[float]

@app.get('/')
def read_root():
    return {'Hello': API_text}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Predictive model loaded'

def format_features_single(brewery: str, aroma: float, appearance: float, palate: float, taste: float, alcohol: float):
    return {
        'brewery_name': [brewery],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv':[alcohol]
    }

def format_features_multiple(brewery, aroma, appearance, palate, taste, alcohol):
    return {
        'brewery_name': [brewery],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv': [alcohol]
    }

@app.get('/beer/type')
def predict_single(brewery: str, aroma: float, appearance: float, palate: float, taste: float, alcohol: float):
    features = format_features_single(brewery,	aroma, appearance, palate, taste, alcohol)
    obs = pd.DataFrame(features)
    pred = model_pipeline.predict(obs)
    pred_name = target_transformer.inverse_transform(pred)
    return JSONResponse(pred_name.tolist())

@app.post('/beers/type')
def predict_multiple(beers_data_csv: UploadFile = File(...)):
    obs = pd.read_csv(beers_data_csv.file)
    pred = model_pipeline.predict(obs)
    pred_name = target_transformer.inverse_transform(pred)
    return JSONResponse(pred_name.tolist())