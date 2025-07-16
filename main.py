from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet

class Point(BaseModel):
    ds: str
    y: float

app = FastAPI()

@app.post("/forecast")
def forecast(data: List[Point]):
    df = pd.DataFrame([d.dict() for d in data])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    fc = model.predict(future)
    return fc[['ds','yhat']].tail(30).to_dict(orient="records")
