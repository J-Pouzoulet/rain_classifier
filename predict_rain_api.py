#see main.ipynb for the pipeline used in the xgb_model_with_threshold.pkl
#'''uvicorn predict_rain_api:app --reload''' in terminal to lunch the API >>> copy and past the url in google chrome
#query example to add in the browser http://127.0.0.1:8000/?date='2010-01-01 00:00:00'&Latitude=44.830667&Longitude=-0.691333&Altitude=47&pmer=99050.0&ff=9.8&t=9.6&u=81.0&ssfrai=0.0&pres=98410&dd_sin=-0.766044&dd_cos=-0.642788
#http://127.0.0.1:8000/docs to play with the API in a UI

from fastapi import FastAPI
import pickle
from typing import Union
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import numpy as np

app = FastAPI()

@app.get('/')
def prediction(date : Union[str, None] = '2010-01-01 00:00:00',
                Latitude : Union[float, None] = 44.830667,
                Longitude : Union[float, None] = -0.691333,
                Altitude : Union[float, None] = 47,
                pmer : Union[float, None] = 99050.0,
                ff : Union[float, None] = 9.8,
                t : Union[float, None] = 9.6,
                u : Union[float, None] = 81.0,
                ssfrai : Union[float, None] = 0.0,
                pres : Union[float, None] = 98410,
                dd_sin : Union[float, None] = -0.766044,
                dd_cos : Union[float, None] = -0.642788):
    
    # Load the model and the threshold
    model_file = pickle.load(open('xgb_model_with_threshold.pkl', 'rb'))
    model = model_file['model']
    threshold = model_file['threshold']
    
    # Create a DataFrame with the input values
    df = pd.DataFrame(data = {'date' : [date],
                                'Latitude' : [Latitude],
                                'Longitude' : [Longitude],
                                'Altitude' : [Altitude],
                                'pmer' : [pmer],
                                'ff' : [ff],
                                't' : [t],
                                'u' : [u],
                                'ssfrai' : [ssfrai],
                                'pres' : [pres],
                                'dd_sin' : [dd_sin],
                                'dd_cos' : [dd_cos]}
                                )
    
    # We set the date column as datetime                    
    df['date'] = pd.to_datetime(df['date'])
    # We extract the date and hours from the datetime in two columns
    df['time'] = df['date'].dt.time
    df['date_only'] = df['date'].dt.date
    # From the date we can extract convert the month and day to the days number in the year
    df['day'] = df['date'].apply(lambda x: x.timetuple().tm_yday)
    # From the time we can extract the hour
    df['hour'] = df['time'].apply(lambda x: x.hour)
    # We convert day to sin and cos to keep the cyclical nature of the data
    df['day_sin'] = np.sin(df['day'] * (2 * np.pi / 365))
    df['day_cos'] = np.cos(df['day'] * (2 * np.pi / 365))   
     # We convert hour to sin and cos to keep the cyclical nature of the data
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))    
    # We drop the columns that are not needed anymore
    df = df.drop(columns=['date', 'time', 'date_only', 'day', 'hour'])
                        
    proba = model.predict_proba(df)[:, 1]
    pred = (proba > threshold).astype(int)
    if pred == 0:
        return {'weather' : 'Not raining'}
    else:
        return {'weather' : 'Raining'}