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

    # Lag / rolling features (used by the what-if predictor and by the
    # PM2.5 target model below)
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
# SIMPLE LIVE MODELS (for the what-if sliders on the Predictions page)
# Trained once per session on the available features — this is intentionally
# lightweight so the app doesn't need a pre-pickled model file committed to
# the repo. Swap this out for joblib.load("model.pkl") if you export one
# from your notebook. One model per target (AQI, PM2.5) so the what-if tool
# can predict either pollutant.
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_whatif_model(df: pd.DataFrame, target: str):
    from sklearn.linear_model import LinearRegression

    # Use weather features plus the *other* pollutant's lag as a predictor —
    # never a pollutant's own lag, since that would make the slider trivially
    # predict "yesterday's value" instead of responding to weather changes.
    if target == "PM2.5":
        feature_cols = [c for c in ["temp", "wind_speed", "humidity", "rainfall", "AQI_lag1"] if c in df.columns]
    else:
        feature_cols = [c for c in ["temp", "wind_speed", "humidity", "rainfall", "PM25_lag1"] if c in df.columns]

    model_df = df[feature_cols + [target]].dropna()
    X = model_df[feature_cols]
    y = model_df[target]
    model = LinearRegression().fit(X, y)
    return model, feature_cols


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
    st.title("Bangalore Air Quality: Overview")
    st.caption("Predictive modeling of AQI and PM2.5 using historical weather and pollution data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg AQI (filtered)", f"{filtered['AQI'].mean():.0f}")
    c2.metric("Avg PM2.5 (filtered)", f"{filtered['PM2.5'].mean():.1f}")
    c3.metric("Peak AQI (filtered)", f"{filtered['AQI'].max():.0f}")
    c4.metric("Days of data", f"{len(filtered):,}")

    st.markdown("### Key insights from EDA")
    st.markdown(
        """
- **AQI shows strong seasonality** - pollution levels rise in the dry winter months and
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
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title="AQI")
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
    fig1.update_layout(xaxis_title="Date", yaxis_title=metric)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if HAS_SEASON:
            fig2 = px.box(filtered, x="season", y=metric, title=f"{metric} distribution by season")
            fig2.update_layout(xaxis_title="Season", yaxis_title=metric)
            st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if "day_of_week" in filtered.columns:
            dow_avg = filtered.groupby("day_of_week")[metric].mean().reset_index()
            fig3 = px.bar(dow_avg, x="day_of_week", y=metric, title=f"Avg {metric} by day of week")
            fig3.update_layout(xaxis_title="Day of week", yaxis_title=f"Avg {metric}")
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

    target = st.radio("Pollutant to predict", ["AQI", "PM2.5"], horizontal=True)
    target_col = "AQI" if target == "AQI" else "PM2.5"

    tab1, tab2 = st.tabs(["📈 Prophet Forecast", "🎛️ What-If Slider"])

    with tab1:
        st.subheader(f"Prophet forecast — future {target} trend")
        horizon = st.slider("Days to forecast", 7, 90, 30)

        @st.cache_resource
        def fit_prophet(df: pd.DataFrame, col: str):
            from prophet import Prophet

            prophet_df = df[["Date", col]].rename(columns={"Date": "ds", col: "y"}).dropna()
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(prophet_df)
            return m

        with st.spinner("Fitting Prophet model..."):
            m = fit_prophet(master, target_col)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

        # Only plot the last 60 days of *history* plus the forecasted horizon.
        # Plotting the full 5.5-year history made the chart look identical
        # regardless of the slider, since a few extra/fewer future days were
        # a tiny sliver of ~2000 total days. Zooming in makes the slider's
        # effect visible, and clearly separates "known" from "forecast".
        last_actual_date = master["Date"].max()
        context_start = last_actual_date - pd.Timedelta(days=60)

        history_view = forecast[forecast["ds"] <= last_actual_date]
        history_view = history_view[history_view["ds"] >= context_start]
        future_view = forecast[forecast["ds"] > last_actual_date]

        fig = px.line(title=f"{target} forecast — last 60 days + next {horizon} days")
        fig.add_scatter(
            x=history_view["ds"], y=history_view["yhat"], mode="lines",
            name="Fitted (recent history)", line=dict(color="#636EFA"),
        )
        fig.add_scatter(
            x=future_view["ds"], y=future_view["yhat"], mode="lines",
            name=f"Forecast (next {horizon} days)", line=dict(color="#EF553B", dash="dash"),
        )
        fig.add_scatter(
            x=future_view["ds"], y=future_view["yhat_upper"], mode="lines",
            line=dict(width=0), showlegend=False,
        )
        fig.add_scatter(
            x=future_view["ds"], y=future_view["yhat_lower"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(239,85,59,0.15)", name="Forecast confidence interval",
        )
        fig.add_vline(x=last_actual_date, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Date", yaxis_title=f"{target} (predicted)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Dotted vertical line marks today ({last_actual_date.date()}) — the model has real data "
            "to the left, and is extrapolating to the right. Drag the slider above to extend or "
            "shorten that right-hand forecast window."
        )

    with tab2:
        st.subheader(f"Adjust weather inputs to see the predicted {target} change")
        st.caption(
            "Powered by a lightweight linear model trained on the full dataset "
            "(swap in your saved Random Forest / OLS model for production use)."
        )

        whatif_model, whatif_features = train_whatif_model(master, target_col)
        last_row = master.dropna(subset=whatif_features).iloc[-1]
        inputs = {}
        cols = st.columns(len(whatif_features))
        for col, feat in zip(cols, whatif_features):
            lo, hi = float(master[feat].min()), float(master[feat].max())
            inputs[feat] = col.slider(feat, lo, hi, float(last_row[feat]))

        X_input = pd.DataFrame([inputs])[whatif_features]
        pred = whatif_model.predict(X_input)[0]

        st.metric(f"Predicted {target}", f"{pred:.0f}" if target == "AQI" else f"{pred:.1f}")

# ────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption(
        "All models were evaluated on the same chronological test set "
        "(June 2019 – July 2020) to keep the comparison fair."
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Random Forest (tuned)", "OLS Linear Regression", "Decision Tree (tuned)",
                "SARIMA(1,1,1)(1,1,1,365)", "ARIMA(1,1,1)", "Prophet (+ weather regressors)",
            ],
            "MAE": [9.11, 9.35, 11.13, 28.08, 29.13, 45.91],
            "RMSE": [12.09, 12.00, 14.27, 31.86, 32.83, 50.21],
            "R²": [0.618, 0.624, 0.468, -1.651, -1.816, -5.590],
        }
    )
    st.dataframe(results, use_container_width=True)

    with st.expander("🤔 Why is Prophet included if it performed worst?", expanded=False):
        st.markdown(
            """
Prophet isn't here because it's competitive on this test set — it's here because it was part of
the project's original scope, and because *why it underperforms* is itself a useful finding.

- **It was scoped from the start.** The project's methodology names ARIMA, SARIMA, and Prophet as
  the primary statistical forecasting track (trend + seasonality decomposition), separate from the
  OLS/Decision Tree/Random Forest ML track. Dropping the weakest result would mean not reporting the
  full comparison that was actually planned and run.
- **A fair comparison has to include the losers, not just the winners.** Showing only the two best
  models would make the analysis look cherry-picked. Including all six shows every approach was
  genuinely tested against the same test set, rather than the outcome being assumed in advance.
- **Its failure has a specific, identifiable cause — it isn't just "a bad model."** Prophet works by
  fitting a smooth trend-plus-seasonality curve and extrapolating it forward, which assumes the
  future looks statistically like the past. The test window (June 2019 – July 2020) includes the
  COVID-19 lockdowns, when Bangalore's AQI dropped sharply due to a one-off collapse in traffic and
  industrial activity — not a seasonal pattern. Prophet had no way to anticipate that from pre-2020
  data, so it kept forecasting a "normal" seasonal curve while actual AQI fell far below it. That's
  exactly why its R² is so deeply negative (-5.59): confidently wrong during a structural break,
  not just imprecise.
- **The contrast is a real methodological conclusion.** Trend-decomposition models like Prophet suit
  stable, cyclical forecasting — genuinely their strength — but are fragile to regime shocks. The
  lag-feature-based ML models (Random Forest, OLS) lean on *yesterday's actual reading* rather than a
  fitted long-term curve, so they adapt one day at a time instead of extrapolating blindly through
  a shock.

**In short:** Prophet's presence here demonstrates the full scoped model set was evaluated honestly,
and its underperformance is explained by a specific cause (COVID-19) rather than hidden from the
comparison.
            """
        )

    with st.expander("ℹ️ What do MAE, RMSE, and R² mean?", expanded=True):
        st.markdown(
            """
- **MAE (Mean Absolute Error)** — the average size of the model's prediction error, in AQI units.
  An MAE of 9.11 means predictions are, on average, about 9 AQI points off from the true value.
  **Lower is better.**
- **RMSE (Root Mean Squared Error)** — similar to MAE, but squares errors before averaging, so it
  punishes large mistakes more heavily than small ones. If RMSE is much bigger than MAE, the model
  is making a few very large errors on top of many small ones. **Lower is better.**
- **R² (R-squared)** — the share of variation in actual AQI that the model successfully explains,
  from 1.0 (perfect) down to negative values (worse than just guessing the average every time).
  A **negative R²**, like SARIMA/ARIMA/Prophet show here, means those models perform *worse* than a
  naive "predict the historical average" baseline on this particular test window — driven by the
  COVID-19 lockdown period sitting inside the test set, which none of those models had seen a
  precedent for. **Higher is better.**

**Bottom line:** Random Forest and OLS Linear Regression are the two usable models here — they're
close enough in accuracy that OLS's simplicity and interpretability make it a reasonable first
choice, with Random Forest as a slightly more accurate but less transparent alternative.
            """
        )

    st.markdown("### SHAP feature importance (Random Forest)")
    st.markdown(
        "SHAP values explain *why* the Random Forest model makes the predictions it does, by "
        "attributing each prediction to the features that pushed it up or down."
    )
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
            st.caption(
                "**How to read this:** longer bars = features the model relies on more heavily "
                "across all its predictions, regardless of direction. AQI_lag1 (yesterday's AQI) "
                "and PM25_lag1 (yesterday's PM2.5) dominate — together they explain roughly "
                "two-thirds of the model's predictive power, confirming that pollution is highly "
                "persistent day-to-day."
            )
        else:
            st.info("plot_shap_bar.png not found in assets/ or repo root.")
    with col2:
        if beeswarm_path:
            st.image(beeswarm_path, caption="SHAP beeswarm — direction & magnitude")
            st.caption(
                "**How to read this:** each dot is one test-set day. Red = high feature value, "
                "blue = low. Dots to the right push the prediction *up*; dots to the left push it "
                "*down*. E.g. red dots (high wind_speed) sitting left of center confirm wind "
                "lowers predicted AQI, while red dots (high AQI_lag1) sitting right confirm "
                "yesterday's pollution carries forward into today's prediction."
            )
        else:
            st.info("plot_shap_beeswarm.png not found in assets/ or repo root.")
