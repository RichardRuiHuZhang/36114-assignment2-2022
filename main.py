#uvicorn main:app --host 127.0.0.1 --port 8080
#Test name Crow Peak Brewing

from fastapi import FastAPI, File, UploadFile, Query
from typing import List
from pydantic import BaseModel
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
#if __name__ == '__main__':
from src import DeepNeuralNet1

app = FastAPI()

#API description
API_text = """
    Heroku App for beer type prediction.
    
    List of endpoints:
        / (root): Front page of the app
        /health: Displays the condition of the app
        /beer/type: The endpoint for a single input, for a single prediction output
        /beers/type: The endpoint for multiple inputs, for multiple prediction outputs
        
    Input variables:
        brewery: a string of brewery name
        aroma: a 0 to 10 number score for beer aroma
        appearance: a 0 to 10 number score for beer appearance
        palate: a 0 to 10 number score for beer palete
        taste: a 0 to 10 number score for beer taste
        alcohol: a 0 to 100 number for the percentage alcohol content
        
    Output:
        Name of a brew for each input
            
    Github link: RichardRuiHuZhang/36114-assignment2-2022 

"""

# class DeepNeuralNet1(nn.Module):
#     def __init__(self, input_dim):
#       super(DeepNeuralNet1,self).__init__()
#       hidden_1 = 512
#       hidden_2 = 512
#       self.fc1 = nn.Linear(input_dim, 512)
#       self.fc2 = nn.Linear(512,512)
#       self.fc3 = nn.Linear(512,103)
#       self.droput = nn.Dropout(0.2)
        
#     def forward(self,x):
#           x = F.relu(self.fc1(x))
#           x = self.droput(x)
#           x = F.relu(self.fc2(x))
#           x = self.droput(x)
#           x = self.fc3(x)
#           return x
    
#model_ML = DeepNeuralNet1(input_dim=1)

#Load Model
#model_pipeline = load('models/test02.joblib')
model_pipeline = load('models/pipeline.joblib')
#model_pipeline = load('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/test02.joblib')
model_ML = torch.load('models/NN01.pt')
#model_ML = torch.load('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/NN01.pt')
#Load Target value transformer
target_transformer = load('models/target_decoder.joblib')
#target_transformer = load('G:/Data Science/UTS Courses/36114 Advanced Data Science for Innovation/Assignment2/36114-assignment2-2022/models/target_decoder.joblib')


@app.get('/')
def read_root():
    return {'App Description': API_text}

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
        'beer_abv': [alcohol]
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

def check_data_validity(dataframe):
    iErr = 0
    iErr = sum(dataframe.isnull().any(axis=1))
    return iErr
    
@app.get('/beer/type')
def predict_single(brewery: str, 
                   aroma: float= Query(..., title="Beer aroma score", gt=0, le=10), 
                   appearance: float= Query(..., title="Beer appearance score", gt=0, le=10),
                   palate: float= Query(..., title="Beer palate score", gt=0, le=10), 
                   taste: float= Query(..., title="Beer taste score", gt=0, le=10), 
                   alcohol: float= Query(..., title="Alcohol content as percentage", gt=0, le=100)):
    features = format_features_single(brewery, aroma, appearance, palate, taste, alcohol)
    obs = pd.DataFrame(features)
    #pred = model_pipeline.predict(obs)
    obs_fitted = model_pipeline.predict(obs)
    model_ML.eval()
    pred = model_ML(obs_fitted)
    pred = pred.numpy()
    pred_name = target_transformer.inverse_transform(pred)
    return JSONResponse(pred_name.tolist())

@app.post('/beers/type')
def predict_multiple(beers_data_csv: UploadFile = File(...)):
    obs = pd.read_csv(beers_data_csv.file)
    iErr = check_data_validity(obs)
    if iErr == 0:
        #pred = model_pipeline.predict(obs)
        obs_fitted = model_pipeline.predict(obs)
        model_ML.eval()
        pred = model_ML(obs_fitted)
        pred = pred.numpy()
    else:
        pred = pd.zeros(obs.shape[1])
    pred_name = target_transformer.inverse_transform(pred)    
    return JSONResponse(pred_name.tolist())