from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
from schema.pydantic_model import Userinput
import pandas as pd

with open("Models/model.pkl",'rb') as f:
    model = pickle.load(f)

lb = model['label_encoder']
scl = model['Standard_scaler']
lr = model['Linear_Regression_model']
random_forest = model['Randome_Forest_regressor']
knn = model['knn_models']
xgb = model['xgBOOST']


app = FastAPI(title="Laptop Price Prediction.")

@app.get("/")
def default():
    return {"message":"Hello , welcome to the Fast Api of the Laptop Price Prediction.",
            "Predcition":"Put the '/docs' into your current url for check the prediction manu."}

@app.post("/predict")
def predictions(predict:Userinput):
    
    new_df = pd.DataFrame([{
        'Brand' : predict.Brand,
        'Model' : predict.Model,
        'CPU' : predict.CPU,
        'Status':predict.Status,
        'RAM':predict.RAM,
        'Storage':predict.Storage,
        'Storage_type':predict.Storage_type,
        'GPU':predict.GPU,
        'Screen':predict.Screen,
        'Touch':predict.Touch
    }])

    temp1 = ['Brand','Model','CPU','Status','Storage type','GPU','Touch']

    for i in temp1:
        new_df[i] = lb.transform(new_df[i])

    new_scaled_data = scl.tranasform(new_df)

    lr_prediction = lr.predict(new_scaled_data)
    random_forest_prediction = random_forest.predict(new_scaled_data)
    knn_prediction = knn.predict(new_scaled_data)
    xgb_predicted = xgb.predict(new_scaled_data)

    return JSONResponse(status_code=200,content={
        'Linear Regresson Prediction':lr_prediction,
        "Randome Forest Prediction":random_forest_prediction,
        "knn Prediction":knn_prediction,
        "XGBoost Prediction":xgb_predicted
    })
