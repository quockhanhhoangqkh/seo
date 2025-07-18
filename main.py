from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from prophet import Prophet

app = FastAPI()


class DataPoint(BaseModel):
    ds: str
    organic_clicks: Optional[float] = 0
    impressions: Optional[float] = 0
    averagePosition: Optional[float] = 0
    position_change: Optional[float] = 0
    ctr: Optional[float] = 0
    new_articles_published: Optional[float] = 0


@app.post("/forecast")
def forecast(data: List[DataPoint]):
    # Convert input to DataFrame
    df = pd.DataFrame([d.dict() for d in data])

    # Replace nulls with 0 (redundant with Pydantic default, but safe)
    df = df.fillna(0)

    # Rename target variable
    df = df.rename(columns={"organic_clicks": "y"})

    # Initialize Prophet model
    model = Prophet()

    # Add extra regressors
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")
    model.add_regressor("position_change")
    model.add_regressor("ctr")
    model.add_regressor("new_articles_published")

    # Fit the model
    model.fit(df)

    # Generate future dataframe (30 days ahead)
    last_date = pd.to_datetime(df["ds"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    # Use median values for regressors
    future_df = pd.DataFrame({
        "ds": future_dates,
        "impressions": df["impressions"].median(),
        "averagePosition": df["averagePosition"].median(),
        "position_change": 0,  # No "previous" future so set to 0
        "ctr": df["ctr"].median(),
        "new_articles_published": df["new_articles_published"].median(),
    })

    # Forecast
    forecast = model.predict(future_df)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
