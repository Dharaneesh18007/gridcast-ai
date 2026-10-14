# GridCast AI — Electricity Demand Forecasting

A hackathon-ready prototype that compares two regression models and forecasts the next 24 hours of electricity demand.

## Features
- Built-in synthetic demo data, so the app runs without a dataset
- CSV upload for historical hourly demand
- Random Forest vs Hist Gradient Boosting comparison
- Chronological 80/20 train/test split
- MAE, RMSE and MAPE metrics
- Actual vs predicted chart
- Next 24-hour forecast CSV download
- Permutation feature-importance chart

## Run on Windows
1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Extract this ZIP to a folder.
3. Open Command Prompt inside the extracted `GridCast_AI_Project` folder.
4. Create a virtual environment:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the website:
   ```bash
   streamlit run app.py
   ```
7. A browser tab opens. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## CSV format
Required columns:
- `timestamp`: date and hour, such as `2026-01-01 00:00:00`
- `demand`: electricity demand/consumption numeric value

Optional columns:
- `temperature`
- `humidity`
- `is_holiday` (1 for holiday, 0 otherwise)

Example:
```csv
timestamp,demand,temperature,humidity,is_holiday
2026-01-01 00:00:00,102.4,27.5,70,1
2026-01-01 01:00:00,98.7,27.0,72,1
```
Use hourly data. At least 72 valid hourly rows are required; 30+ days is recommended. The app also recognizes common alternative names such as `datetime`, `load_kw`, `demand_kwh`, and `temp`.

## How it works
1. Parses and sorts timestamped demand data.
2. Builds calendar features, lag-1, lag-24 and previous-24-hour rolling mean.
3. Uses the first 80% of time for training and the final 20% for testing.
4. Compares Random Forest and Hist Gradient Boosting using MAE, RMSE and MAPE.
5. Selects the model with the lowest test MAE.
6. Recursively predicts the next 24 hours and visualizes feature importance.

## Important limitations
- The built-in data is synthetic and only demonstrates the workflow; do not present demo metrics as real-world performance.
- The forecast currently carries forward the latest observed temperature and humidity. Add a weather forecast source for a stronger real-world system.
- Replace the simple holiday flag with an accurate local/campus holiday calendar.
- Before deployment, test against actual campus/grid readings and check for data gaps, meter resets and outliers.
