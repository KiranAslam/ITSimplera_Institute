"""
main.py — FastAPI Dashboard & Prediction App
Week 3, Part 2

Routes:
  GET  /          -> welcome page with nav bar
  GET  /dashboard -> EDA visualizations (3 charts)
  GET  /predict   -> prediction form
  POST /predict   -> runs the loaded pipeline and shows the prediction

Before running this app, generate model.joblib and the dashboard charts:
    python train_and_save_model.py

Then start the server:
    uvicorn main:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="Steel Industry Energy Consumption Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_path = os.path.join("models", "model.joblib")
artifact = joblib.load(model_path)
pipeline = artifact["pipeline"]
FEATURE_NAMES = artifact["feature_names"]
LOAD_TYPE_CATEGORIES = artifact["load_type_categories"]
DAY_OF_WEEK_CATEGORIES = artifact["day_of_week_categories"]
NUMERIC_RANGES = artifact["numeric_ranges"]

# Raw numeric inputs the user fills in directly.
# (Power_Factor_Ratio is NOT collected directly -- it is a derived feature,
#  computed automatically from Lagging/Leading Power Factor below, so the
#  user can't accidentally submit an inconsistent value.)
# Each entry: (HTML form field name, original dataset column, display label)
RAW_NUMERIC_FIELDS = [
    ("lagging_reactive_power", "Lagging_Current_Reactive.Power_kVarh", "Lagging Reactive Power (kVarh)"),
    ("leading_reactive_power", "Leading_Current_Reactive_Power_kVarh", "Leading Reactive Power (kVarh)"),
    ("co2", "CO2(tCO2)", "CO2 (tCO2)"),
    ("lagging_power_factor", "Lagging_Current_Power_Factor", "Lagging Power Factor (%)"),
    ("leading_power_factor", "Leading_Current_Power_Factor", "Leading Power Factor (%)"),
    ("nsm", "NSM", "Seconds Since Midnight (NSM)"),
    ("hour", "Hour", "Hour of Day (0-23)"),
    ("month", "Month", "Month (1-12)"),
    ("is_weekend", "Is_Weekend", "Is Weekend? (0=No, 1=Yes)"),
]


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/predict")
def predict_form(request: Request):
    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "raw_numeric_fields": RAW_NUMERIC_FIELDS,
            "numeric_ranges": NUMERIC_RANGES,
            "load_types": LOAD_TYPE_CATEGORIES,
            "days": DAY_OF_WEEK_CATEGORIES,
            "prediction": None,
        },
    )


@app.post("/predict")
def predict(
    request: Request,
    lagging_reactive_power: float = Form(...),
    leading_reactive_power: float = Form(...),
    co2: float = Form(...),
    lagging_power_factor: float = Form(...),
    leading_power_factor: float = Form(...),
    nsm: float = Form(...),
    hour: int = Form(...),
    month: int = Form(...),
    is_weekend: int = Form(...),
    load_type: str = Form(...),
    day_of_week: str = Form(...),
):
    # Derived feature: same formula used in Week 2 feature engineering.
    power_factor_ratio = leading_power_factor / lagging_power_factor

    row = {
        "Lagging_Current_Reactive.Power_kVarh": lagging_reactive_power,
        "Leading_Current_Reactive_Power_kVarh": leading_reactive_power,
        "CO2(tCO2)": co2,
        "Lagging_Current_Power_Factor": lagging_power_factor,
        "Leading_Current_Power_Factor": leading_power_factor,
        "NSM": nsm,
        "Hour": hour,
        "Month": month,
        "Is_Weekend": is_weekend,
        "Power_Factor_Ratio": power_factor_ratio,
    }

    # One-hot encode Load_Type / DayOfWeek exactly as in training
    # (drop_first=True dropped "Light_Load" and "Friday" as baselines).
    for cat in LOAD_TYPE_CATEGORIES:
        if cat == "Light_Load":
            continue
        row[f"Load_Type_{cat}"] = 1 if load_type == cat else 0

    for day in DAY_OF_WEEK_CATEGORIES:
        if day == "Friday":
            continue
        row[f"DayOfWeek_{day}"] = 1 if day_of_week == day else 0

    X = pd.DataFrame([row])[FEATURE_NAMES]  # enforce exact training column order
    prediction = float(pipeline.predict(X)[0])

    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "raw_numeric_fields": RAW_NUMERIC_FIELDS,
            "numeric_ranges": NUMERIC_RANGES,
            "load_types": LOAD_TYPE_CATEGORIES,
            "days": DAY_OF_WEEK_CATEGORIES,
            "prediction": round(prediction, 3),
            "submitted": row,
        },
    )