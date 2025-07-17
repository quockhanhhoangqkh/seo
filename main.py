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

@app.post("/predict")
def predict(data: List[DataPoint]):
    df = pd.DataFrame([d.dict() for d in data])
    
    # Khai báo các biến đầu vào
    model = Prophet()
    model.add_regressor("impressions")
    model.add_regressor("averagePosition")
    
    model.fit(df.rename(columns={"organic_clicks": "y"}))  # y phải là cột chính

    # Tạo future dataframe, cần giữ nguyên các biến đầu vào!
    future = df[["ds", "impressions", "averagePosition"]]  # KHÔNG dùng make_future_dataframe
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
