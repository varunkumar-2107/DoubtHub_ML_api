from pathlib import Path
from typing import Annotated
import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from training.Clustering import retrain


app = FastAPI()

model = joblib.load('model/domain_classifier.joblib')
vectoriser = joblib.load('model/domain_classifier_vectorizer.joblib')


class User_input(BaseModel):
    Query: Annotated[str, Field(..., description="Enter your query")]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Query Classification API. Use the /predict endpoint to classify your query."}


# data_path = "./ml_service/data/Clustering_data.csv"
@app.post("/recommend")
def recommend_querry(data: User_input):
    result = retrain(new_querry=data.Query)
    return JSONResponse(
        status_code=200,
        content={"recommendations": result},
    )


@app.post("/predict")
def predict_querry(data: User_input):
    input_data = [data.Query]
    input_data = [text.lower() for text in input_data]
    input_vector = vectoriser.transform(input_data).toarray()
    prediction = model.predict(input_vector)[0]
    if prediction.item() == 0:
        predict = 'Backend'
    elif prediction.item() == 1:
        predict = 'Cyber Security'
    elif prediction.item() == 2:
        predict = 'DSA'
    elif prediction.item() == 3:
        predict = 'Frontend'
    elif prediction.item() == 4:
        predict = 'ML/AI'
    elif prediction.item() == 5:
        predict = 'Programming Language'
    else:
        predict = 'Other'
    
    return JSONResponse(
        status_code=200,
        content={"prediction": predict},
    ) 