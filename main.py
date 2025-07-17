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
    # Convert input to DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    df = df.rename(columns={"organic_clicks": "y"})

    # Tạo model Prophet
    model = Prophet()
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")

    model.fit(df)

    # Tạo 30 ngày tiếp theo với giá trị trung bình của các biến
    last_date = pd.to_datetime(df["ds"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    future_df = pd.DataFrame({
        "ds": future_dates,
        "impressions": df["impressions"].median(),
        "averagePosition": df["averagePosition"].median()
    })

    forecast = model.predict(future_df)

    # Trả về kết quả
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
