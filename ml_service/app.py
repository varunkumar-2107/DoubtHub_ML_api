from pathlib import Path
from typing import Annotated
import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from training.recomended import retrain
from model_3.solution import solution


app = FastAPI()

model = joblib.load('model/domain_classifier.joblib')
vectoriser = joblib.load('model/domain_classifier_vectorizer.joblib')


class User_input(BaseModel):
    Query: Annotated[str, Field(..., description="Enter your query")]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Query Classification API. Use the /predict endpoint to classify your query."}



@app.post("/recommend")
def recommend_query(data: User_input):
    domain = get_domain(data)
    result = retrain(new_querry=data.Query , domain=domain)
    return JSONResponse(
        status_code=200,
        content={"recommendations": result},
    )



@app.post("/solution")
def give_solution(data: User_input):
    domain = get_domain(data)
    result = solution(new_querry=data.Query , domain=domain)
    return JSONResponse(
        status_code=200,
        content={result['Question']: result['Answer']},
    )



@app.post("/predict")
def predict_domain(data: User_input):
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


def get_domain(data: User_input):
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
    
    return predict