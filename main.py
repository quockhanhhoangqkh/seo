from fastapi import FastAPI, Request
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

    # Rename target column
    df = df.rename(columns={"organic_clicks": "y"})

    # Initialize model
    model = Prophet()
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")

    # Fit model
    model.fit(df)

    # Prepare future dataframe (30 days ahead)
    last_date = pd.to_datetime(df["ds"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    # Fill future regressor values with median of training set
    median_impressions = df["impressions"].median()
    median_position = df["averagePosition"].median()

    future_df = pd.DataFrame({
        "ds": future_dates,
        "impressions": median_impressions,
        "averagePosition": median_position
    })

    # Predict
    forecast = model.predict(future_df)

    # Return only needed fields
    output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return output.to_dict(orient="records")
