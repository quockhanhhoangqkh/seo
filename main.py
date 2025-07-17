from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet

app = FastAPI()

class DataPoint(BaseModel):
    ds: str
    organic_clicks: float
    impressions: float
    averagePosition: float

class ForecastRequest(BaseModel):
    data: List[DataPoint]

@app.post("/forecast")
def forecast(request: ForecastRequest):
    df = pd.DataFrame([d.dict() for d in request.data])

    model = Prophet()
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")

    model.fit(df.rename(columns={"organic_clicks": "y"}))
    future = df[["ds", "impressions", "averagePosition"]]
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
