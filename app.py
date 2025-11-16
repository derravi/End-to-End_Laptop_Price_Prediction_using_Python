from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
from schema.pydantic_model import Userinput
import pandas as pd

with open("Models/laptop_price_prediction.pkl", 'rb') as f:
    model = pickle.load(f)

encoders = model["encoders"]
scl = model["Standard_scaler"]
lr = model["Linear_Regression_model"]
random_forest = model["Random_Forest_regressor"]
knn = model["knn_model"]
xgb = model["xgboost_model"]
column_order = model["columns"] 


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
        'Storage type':predict.Storage_type,
        'GPU':predict.GPU,
        'Screen':predict.Screen,
        'Touch':predict.Touch
    }])
    
    temp1 = ['Brand', 'Model', 'CPU', 'Status', 'Storage type', 'GPU', 'Touch']
    
    for col in temp1:
        encoder = encoders[col]

        if new_df[col][0] in encoder.classes_:
            new_df[col] = encoder.transform(new_df[col])
        else:
            new_df[col] = [-1] 

    new_df = new_df[column_order]
    
    new_scaled_data = scl.transform(new_df) 

    lr_prediction = lr.predict(new_scaled_data)[0]
    rf_prediction = random_forest.predict(new_scaled_data)[0]
    knn_prediction = knn.predict(new_scaled_data)[0]
    xgb_prediction = xgb.predict(new_scaled_data)[0]
    
    indr = 88.69
    
    return JSONResponse(
    status_code=200,
    content={
        "LinearRegression_Price": f"{float(round(lr_prediction * indr, 2))} Rupee with 65.0 % Accuracy.",
        "RandomForest_Price": f"{float(round(rf_prediction * indr, 2))} Rupee with 77.0 % Accuracy.",
        "KNN_Price": f"{float(round(knn_prediction * indr, 2))} Rupee with 83.0 % Accuracy.",
        "XGBoost_Price": f"{float(round(xgb_prediction * indr, 2))} Rupee with 81.0 % Accuracy."
    })