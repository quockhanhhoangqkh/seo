from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from prophet import Prophet

app = FastAPI()

class DataPoint(BaseModel):
    ds: str
    y: float
    impressions: Optional[float] = None
    averagePosition: Optional[float] = None

@app.post("/predict")
def predict(data: List[DataPoint]):
    df = pd.DataFrame([d.dict() for d in data])
    
    model = Prophet()
    # Add regressors
    if "impressions" in df.columns:
        model.add_regressor("impressions")
    if "averagePosition" in df.columns:
        model.add_regressor("averagePosition")
    
    model.fit(df)
    
    # Predict 30 days ahead
    future = model.make_future_dataframe(periods=30)
    
    # Fill regressors for future (here, just using last known value for demo)
    last_row = df.iloc[-1]
    if "impressions" in df.columns:
        future["impressions"] = last_row["impressions"]
    if "averagePosition" in df.columns:
        future["averagePosition"] = last_row["averagePosition"]

    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30).to_dict(orient="records")
