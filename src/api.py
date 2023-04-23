from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing



# Load Config and Production Model
config_data = utils.load_config()
model_data = utils.pickle_load(config_data["production_model_path"])

# Load OneHotEncoding
ohe_season =  utils.pickle_load(config_data["ohe_season"])
ohe_weathersit =  utils.pickle_load(config_data["ohe_weathersit"])
ohe_weekday =  utils.pickle_load(config_data["ohe_weekday"])

# OHE Model - The Order of The Dictionary Shouldnt Change
OHE_MODELS = {
    "weekday": ohe_weekday,
    "season": ohe_season,
    "weathersit": ohe_weathersit,
}

class api_data(BaseModel):
    dteday : int
    season : int
    yr : int
    mnth : int
    hr : int
    holiday : int
    weekday : int
    workingday : int
    weathersit : int
    temp : float
    hum : float
    windspeed : float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)
    
    # Encoding all variables that are needed
    for feature, ohe_model in OHE_MODELS.items():
        data = preprocessing.ohe_transform(data=data,
                                           feature=feature,
                                           ohe_model=ohe_model)
        data.rename(config_data[f"{feature}_map"],
                           axis=1,
                           inplace=True)

    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)
    
    result = {"res" : float(y_pred), "error_msg": ""}

    return result

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8088)