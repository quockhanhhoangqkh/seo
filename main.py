from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet

app = FastAPI(
    title="SEO Forecast API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

class DataPoint(BaseModel):
    ds: str
    y: float

@app.post("/forecast")
def forecast(data: List[DataPoint]):
    df = pd.DataFrame([d.dict() for d in data])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(30).to_dict(orient="records")
