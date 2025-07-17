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

@app.post("/forecast")
def forecast(data: List[DataPoint]):
    df = pd.DataFrame([d.dict() for d in data])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.rename(columns={"organic_clicks": "y"})

    model = Prophet()

    # Add extra regressors (multivariate inputs)
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")

    model.fit(df)

    future = model.make_future_dataframe(periods=30)

    # Fill extra columns with last known value (or set manually if cần thiết)
    for col in ["impressions", "averagePosition"]:
        future[col] = df[col].iloc[-1]

    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30).to_dict(orient="records")
