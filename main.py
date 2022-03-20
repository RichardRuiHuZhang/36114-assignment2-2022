from fastapi import FastAPI, File, UploadFile, Query
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

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

#Load Model
model_pipeline = load('models/pipeline.joblib')
#Load Target value transformer
target_transformer = load('models/target_decoder.joblib')

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
    pred = model_pipeline.predict(obs)
    pred_name = target_transformer.inverse_transform(pred)
    return JSONResponse(pred_name.tolist())

@app.post('/beers/type')
def predict_multiple(beers_data_csv: UploadFile = File(...)):
    obs = pd.read_csv(beers_data_csv.file)
    iErr = check_data_validity(obs)
    if iErr == 0:
        pred = model_pipeline.predict(obs)
    else:
        pred = pd.zeros(obs.shape[1])
    pred_name = target_transformer.inverse_transform(pred)    
    return JSONResponse(pred_name.tolist())