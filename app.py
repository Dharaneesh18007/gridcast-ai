import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="GridCast AI", page_icon="⚡", layout="wide")
st.title("⚡ GridCast AI — Electricity Demand Forecasting")
st.caption("24-hour demand forecasting • Model comparison • Explainable AI")

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "temperature",
            "humidity", "is_holiday", "lag_1", "lag_24", "rolling_mean_24"]

def make_demo_data(days=120, seed=7):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=days*24, freq="h")
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    day_num = np.arange(len(timestamps))
    temperature = 29 + 5*np.sin(2*np.pi*(hour-7)/24) + 2*np.sin(2*np.pi*day_num/(24*30)) + rng.normal(0, 1.4, len(timestamps))
    humidity = np.clip(68 - 0.7*(temperature-29) + rng.normal(0, 5, len(timestamps)), 25, 98)
    holiday = ((timestamps.dayofweek == 6) & (timestamps.day <= 7)).astype(int)
    morning = 22*np.exp(-((hour-9)/2.5)**2)
    evening = 38*np.exp(-((hour-19)/3.0)**2)
    workday = np.where(dow < 5, 22, 7)
    demand = 95 + morning + evening + workday + 1.7*(temperature-29) - 0.12*(humidity-65) - 10*holiday
    demand = demand + 6*np.sin(2*np.pi*day_num/(24*7)) + rng.normal(0, 5, len(timestamps))
    demand = np.maximum(demand, 25)
    return pd.DataFrame({"timestamp": timestamps, "demand": demand,
                         "temperature": temperature, "humidity": humidity,
                         "is_holiday": holiday})

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c in ["datetime", "date_time", "time", "date", "timestamp"]:
            rename[c] = "timestamp"
        elif c in ["load", "load_kw", "demand_kw", "demand_kwh", "electricity_demand", "consumption", "target"]:
            rename[c] = "demand"
        elif c in ["temp", "temperature_c", "temperature_celsius"]:
            rename[c] = "temperature"
        elif c in ["relative_humidity", "humidity_percent"]:
            rename[c] = "humidity"
        elif c in ["holiday", "holiday_flag"]:
            rename[c] = "is_holiday"
    df = df.rename(columns=rename)
    required = {"timestamp", "demand"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain timestamp and demand columns (e.g., timestamp,demand,temperature,humidity,is_holiday).")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    for c, default in [("temperature", 29.0), ("humidity", 65.0), ("is_holiday", 0)]:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    df = df.dropna(subset=["timestamp", "demand"]).sort_values("timestamp").drop_duplicates("timestamp")
    if len(df) < 72:
        raise ValueError("Please provide at least 72 hourly records; 30 days or more is recommended.")
    df["hour"] = df.timestamp.dt.hour
    df["dayofweek"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    df["lag_1"] = df.demand.shift(1)
    df["lag_24"] = df.demand.shift(24)
    df["rolling_mean_24"] = df.demand.shift(1).rolling(24).mean()
    return df.dropna(subset=FEATURES + ["demand"]).reset_index(drop=True)

def metrics(y, p):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    denom = np.maximum(np.abs(np.asarray(y)), 1e-6)
    mape = np.mean(np.abs((np.asarray(y)-np.asarray(p))/denom))*100
    return mae, rmse, mape

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload hourly CSV (optional)", type=["csv"])
    st.write("No CSV? Use built-in synthetic demo data.")
    st.caption("Recommended: at least 30 days of hourly readings.")

try:
    raw = pd.read_csv(upload) if upload else make_demo_data()
    data = prepare_data(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

if not upload:
    st.info("Demo mode: results use synthetic sample data, not real campus electricity readings.")

st.subheader("1. Historical data")
c1, c2, c3 = st.columns(3)
c1.metric("Hourly records", f"{len(data):,}")
c2.metric("Average demand", f"{data.demand.mean():.1f}")
c3.metric("Latest recorded demand", f"{data.demand.iloc[-1]:.1f}")
st.line_chart(data.set_index("timestamp")["demand"].tail(24*7), y_label="Demand (dataset units)")

if len(data) < 100:
    st.warning("A larger dataset usually gives more reliable results. Use 30+ days if available.")

X = data[FEATURES]
y = data["demand"]
split = int(len(data)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Random Forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.08, l2_regularization=1.0, random_state=42)
}
predictions, rows = {}, []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    mae, rmse, mape = metrics(y_test, pred)
    rows.append({"Model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE (%)": round(mape, 2)})

scores = pd.DataFrame(rows).sort_values("MAE")
best_name = scores.iloc[0]["Model"]
best_model = models[best_name]

st.subheader("2. Compare machine-learning models")
st.dataframe(scores, use_container_width=True, hide_index=True)
st.success(f"Best model on the chronological test set by MAE: {best_name}")
st.caption("Lower MAE, RMSE and MAPE generally indicate better predictions. This is a time-ordered 80/20 split, not a random split.")

st.subheader("3. Actual vs predicted demand")
plot_df = pd.DataFrame({"Actual": y_test.to_numpy(), **predictions}, index=data["timestamp"].iloc[split:])
st.line_chart(plot_df.tail(24*7), y_label="Demand (dataset units)")

st.subheader("4. Next 24-hour forecast")
last = data.iloc[-1]
future_times = pd.date_range(last.timestamp + pd.Timedelta(hours=1), periods=24, freq="h")
history = list(data["demand"].astype(float).to_numpy())
future_rows = []
for ts in future_times:
    row = {
        "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "temperature": float(last.temperature),
        "humidity": float(last.humidity),
        "is_holiday": int(ts.dayofweek == 6 and ts.day <= 7),
        "lag_1": history[-1],
        "lag_24": history[-24] if len(history) >= 24 else history[-1],
        "rolling_mean_24": float(np.mean(history[-24:]))
    }
    future_rows.append(row)
    history.append(float(best_model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
forecast = pd.DataFrame({"timestamp": future_times, "predicted_demand": history[-24:]})
st.dataframe(forecast.assign(predicted_demand=forecast.predicted_demand.round(2)), use_container_width=True, hide_index=True)
st.line_chart(forecast.set_index("timestamp")["predicted_demand"], y_label="Predicted demand (dataset units)")
st.download_button("Download 24-hour forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                   file_name="gridcast_24_hour_forecast.csv", mime="text/csv")

st.subheader("5. Explainable AI — feature importance")
try:
    sample_n = min(500, len(X_test))
    sample = X_test.tail(sample_n)
    result = permutation_importance(best_model, sample, y_test.tail(sample_n), n_repeats=3,
                                    random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"Feature": FEATURES, "Importance": result.importances_mean})
    imp = imp.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp["Feature"], imp["Importance"])
    ax.set_xlabel("Permutation importance (higher = more useful)")
    ax.set_title(f"Feature importance — {best_name}")
    fig.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not calculate feature importance for this dataset: {e}")

st.markdown("---")
st.caption("Prototype note: future temperature and humidity currently use the latest observed values. For deployment, connect a weather forecast API and validate with real campus/grid data.")
