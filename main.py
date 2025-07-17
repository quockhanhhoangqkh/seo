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
    avg_position: Optional[float] = None
    post_count: Optional[int] = None
    keyword_count: Optional[int] = None

@app.post("/forecast")
def forecast(data: List[DataPoint]):
    df = pd.DataFrame([d.dict() for d in data])

    model = Prophet()
    for reg in ["impressions", "avg_position", "post_count", "keyword_count"]:
        if reg in df.columns and df[reg].notnull().all():
            model.add_regressor(reg)

    model.fit(df)

    future = model.make_future_dataframe(periods=30)

    # Copy các cột extra regressors nếu có
    for reg in ["impressions", "avg_position", "post_count", "keyword_count"]:
        if reg in df.columns and df[reg].notnull().all():
            last_value = df[reg].iloc[-1]
            future[reg] = last_value  # giữ giá trị cuối cho 30 ngày tới

    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(30).to_dict(orient="records")
