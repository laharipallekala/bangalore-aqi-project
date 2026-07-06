"""
Bangalore AQI & PM2.5 Dashboard
--------------------------------
Pages: Home | Trends | Predictions | Model Comparison
Run locally with:  streamlit run app.py
Deploy free at:    https://share.streamlit.io  (connect this GitHub repo)
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Bangalore AQI Dashboard", layout="wide", page_icon="🌫️")

# ────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────
DATA_PATH = "processed/bangalore_master_aqi.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # format='mixed' handles ISO ("2015-01-01") and any other consistent
    # string layout without hard-coding a single pattern that can break
    # the whole app if the source data ever changes format.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    # Lag / rolling features (used by the what-if predictor below)
    df["AQI_lag1"] = df["AQI"].shift(1)
    df["PM25_lag1"] = df["PM2.5"].shift(1)
    df["AQI_roll7"] = df["AQI"].rolling(7).mean()

    return df


if not os.path.exists(DATA_PATH):
    st.error(
        f"Couldn't find `{DATA_PATH}`. Make sure the CSV is committed to your "
        "repo at that exact path (case-sensitive on Streamlit Cloud's Linux servers)."
    )
    st.stop()

master = load_data(DATA_PATH)

HAS_STATION = "station" in master.columns
HAS_SEASON = "season" in master.columns

# ────────────────────────────────────────────────────────────────────────────
# SIMPLE LIVE MODEL (for the what-if sliders on the Predictions page)
# Trained once per session on the available features — this is intentionally
# lightweight so the app doesn't need a pre-pickled model file committed to
# the repo. Swap this out for joblib.load("model.pkl") if you export one
# from your notebook.
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_whatif_model(df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression

    feature_cols = [c for c in ["temp", "wind_speed", "humidity", "rainfall", "AQI_lag1"] if c in df.columns]
    model_df = df[feature_cols + ["AQI"]].dropna()

    X = model_df[feature_cols]
    y = model_df["AQI"]
    model = LinearRegression().fit(X, y)
    return model, feature_cols


whatif_model, whatif_features = train_whatif_model(master)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — navigation + global filters
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ Bangalore AQI")
    page = st.radio("Navigate", ["Home", "Trends", "Predictions", "Model Comparison"])

    st.markdown("---")
    st.subheader("Filters")

    min_date, max_date = master["Date"].min().date(), master["Date"].max().date()
    date_range = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)

    season_filter = None
    if HAS_SEASON:
        seasons = sorted(master["season"].dropna().unique())
        season_filter = st.multiselect("Season", seasons, default=seasons)

    station_filter = None
    if HAS_STATION:
        stations = sorted(master["station"].dropna().unique())
        station_filter = st.multiselect("Station", stations, default=stations)

# Apply global filters
filtered = master.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["Date"].dt.date >= start) & (filtered["Date"].dt.date <= end)]
if season_filter:
    filtered = filtered[filtered["season"].isin(season_filter)]
if station_filter:
    filtered = filtered[filtered["station"].isin(station_filter)]

# ────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ────────────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("Bangalore Air Quality — Overview")
    st.caption("Predictive modeling of AQI and PM2.5 using historical weather and pollution data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg AQI (filtered)", f"{filtered['AQI'].mean():.0f}")
    c2.metric("Avg PM2.5 (filtered)", f"{filtered['PM2.5'].mean():.1f}")
    c3.metric("Peak AQI (filtered)", f"{filtered['AQI'].max():.0f}")
    c4.metric("Days of data", f"{len(filtered):,}")

    st.markdown("### Key insights from EDA")
    st.markdown(
        """
- **AQI shows strong seasonality** — pollution levels rise in the dry winter months and
  drop sharply during the monsoon, driven largely by rainfall and wind speed.
- **Yesterday's AQI is the single strongest predictor** of today's AQI (`AQI_lag1`),
  reflecting the persistence of pollution episodes.
- **Wind speed has a consistent dampening effect** on AQI — higher wind correlates with
  cleaner air by dispersing particulates.
- **Random Forest and OLS Regression perform comparably** on held-out test data, with
  Random Forest slightly ahead on error metrics — see the Model Comparison page for details.
        """
    )

    fig = px.line(filtered, x="Date", y="AQI", title="AQI over time (filtered range)")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# PAGE: TRENDS
# ────────────────────────────────────────────────────────────────────────────
elif page == "Trends":
    st.title("Interactive Trends")

    metric = st.radio("Metric", ["AQI", "PM2.5"], horizontal=True)

    fig1 = px.line(
        filtered, x="Date", y=metric,
        color="season" if HAS_SEASON else None,
        title=f"{metric} over time",
    )
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if HAS_SEASON:
            fig2 = px.box(filtered, x="season", y=metric, title=f"{metric} distribution by season")
            st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if "day_of_week" in filtered.columns:
            dow_avg = filtered.groupby("day_of_week")[metric].mean().reset_index()
            fig3 = px.bar(dow_avg, x="day_of_week", y=metric, title=f"Avg {metric} by day of week")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Correlation heatmap")
    num_cols = [c for c in ["AQI", "PM2.5", "temp", "humidity", "wind_speed", "rainfall"] if c in filtered.columns]
    if len(num_cols) > 1:
        corr = filtered[num_cols].corr()
        fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig4, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTIONS
# ────────────────────────────────────────────────────────────────────────────
elif page == "Predictions":
    st.title("Forecasting & What-If Analysis")

    tab1, tab2 = st.tabs(["📈 Prophet Forecast", "🎛️ What-If Slider"])

    with tab1:
        st.subheader("Prophet forecast — future AQI trend")
        horizon = st.slider("Days to forecast", 7, 90, 30)

        @st.cache_resource
        def fit_prophet(df: pd.DataFrame):
            from prophet import Prophet

            prophet_df = df[["Date", "AQI"]].rename(columns={"Date": "ds", "AQI": "y"}).dropna()
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(prophet_df)
            return m

        with st.spinner("Fitting Prophet model..."):
            m = fit_prophet(master)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title=f"AQI forecast — next {horizon} days")
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(
            x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(99,110,250,0.2)", name="Confidence interval",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Adjust weather inputs to see the predicted AQI change")
        st.caption(
            "Powered by a lightweight linear model trained on the full dataset "
            "(swap in your saved Random Forest / OLS model for production use)."
        )

        last_row = master.dropna(subset=whatif_features).iloc[-1]
        inputs = {}
        cols = st.columns(len(whatif_features))
        for col, feat in zip(cols, whatif_features):
            lo, hi = float(master[feat].min()), float(master[feat].max())
            inputs[feat] = col.slider(feat, lo, hi, float(last_row[feat]))

        X_input = pd.DataFrame([inputs])[whatif_features]
        pred = whatif_model.predict(X_input)[0]

        st.metric("Predicted AQI", f"{pred:.0f}")

# ────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")

    # NOTE: fill these in with the exact numbers from your notebook's final
    # evaluation cell if they differ — these are placeholders based on the
    # results referenced in 03_modeling.ipynb.
    results = pd.DataFrame(
        {
            "Model": ["OLS Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
            "MAE": [9.35, 11.13, 9.11],
            "RMSE": [None, None, None],
            "R²": [0.6239, 0.4683, 0.6181],
        }
    )
    st.dataframe(results, use_container_width=True)

    st.markdown("### SHAP feature importance (Random Forest)")
    col1, col2 = st.columns(2)

    def find_image(name: str):
        for candidate in [f"assets/{name}", name]:
            if os.path.exists(candidate):
                return candidate
        return None

    bar_path = find_image("plot_shap_bar.png")
    beeswarm_path = find_image("plot_shap_beeswarm.png")

    with col1:
        if bar_path:
            st.image(bar_path, caption="Mean |SHAP value| — overall importance")
        else:
            st.info("plot_shap_bar.png not found in assets/ or repo root.")
    with col2:
        if beeswarm_path:
            st.image(beeswarm_path, caption="SHAP beeswarm — direction & magnitude")
        else:
            st.info("plot_shap_beeswarm.png not found in assets/ or repo root.")
