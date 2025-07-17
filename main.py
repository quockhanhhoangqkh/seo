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

class RequestBody(BaseModel):
    data: List[DataPoint]

@app.post("/forecast")
def forecast(body: RequestBody):
    df = pd.DataFrame([d.dict() for d in body.data])
    
    # Rename y column for Prophet
    df.rename(columns={"organic_clicks": "y"}, inplace=True)

    # Train model
    model = Prophet()
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")
    model.fit(df)

    # ⚠️ Tạo future 30 ngày tiếp theo
    future_dates = pd.date_range(start=df["ds"].max(), periods=30, freq="D")[1:]  # bỏ ngày cuối đã có
    # Giữ nguyên giá trị cuối cùng của các biến đầu vào để dự báo
    last_impr = df["impressions"].iloc[-1]
    last_pos = df["averagePosition"].iloc[-1]

    future = pd.DataFrame({
        "ds": future_dates,
        "impressions": last_impr,
        "averagePosition": last_pos
    })

    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
