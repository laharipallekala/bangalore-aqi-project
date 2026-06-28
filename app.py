import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Bengaluru AQI Dashboard',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('processed/bangalore_master_aqi.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    df['AQI_lag1']  = df['AQI'].shift(1)
    df['PM25_lag1'] = df['PM2.5'].shift(1)
    df = df.dropna(subset=['AQI_lag1', 'PM25_lag1'])
    season_dummies = pd.get_dummies(df['season'],      prefix='season', drop_first=True)
    dow_dummies    = pd.get_dummies(df['day_of_week'], prefix='dow',    drop_first=True)
    df = pd.concat([df, season_dummies, dow_dummies], axis=1)
    return df

master = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_Karnataka.svg/320px-Flag_of_Karnataka.svg.png',
        width=80
    )
    st.title('🌿 Bengaluru AQI')
    st.markdown('---')
    page = st.selectbox(
        'Navigate to',
        ['🏠 Home', '📈 Trends', '🔮 Predictions', '📊 Model Comparison']
    )
    st.markdown('---')
    st.subheader('Global Filters')
    selected_seasons = st.multiselect(
        'Filter by Season',
        options=['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'],
        default=['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    )
    year_range = st.slider(
        'Year Range',
        min_value=int(master['Date'].dt.year.min()),
        max_value=int(master['Date'].dt.year.max()),
        value=(int(master['Date'].dt.year.min()), int(master['Date'].dt.year.max()))
    )
    filtered = master[
        (master['season'].isin(selected_seasons)) &
        (master['Date'].dt.year >= year_range[0]) &
        (master['Date'].dt.year <= year_range[1])
    ].copy()
    st.markdown('---')
    st.caption(f'Showing {len(filtered):,} of {len(master):,} days')

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == '🏠 Home':
    st.title('🌿 Bengaluru Air Quality Dashboard')
    st.markdown('**Predictive Modeling of AQI and PM2.5 Levels | 2015–2020**')
    st.markdown('---')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('📅 Data Period',
                  f"{master['Date'].dt.year.min()}–{master['Date'].dt.year.max()}",
                  f"{len(master):,} days")
    with col2:
        avg_aqi = round(master['AQI'].mean(), 1)
        st.metric('💨 Mean AQI', avg_aqi, 'Moderate category')
    with col3:
        avg_pm25 = round(master['PM2.5'].mean(), 1)
        st.metric('🏭 Mean PM2.5', f'{avg_pm25} µg/m³', '2.4× WHO guideline')
    with col4:
        worst_season = master.groupby('season')['AQI'].mean().idxmax()
        st.metric('📆 Worst Season', worst_season, 'by average AQI')

    st.markdown('---')
    left, right = st.columns([3, 2])

    with left:
        st.subheader('Monthly Average AQI')
        monthly = (master
                   .groupby(master['Date'].dt.to_period('M'))['AQI']
                   .mean().reset_index())
        monthly['Date'] = monthly['Date'].astype(str)
        fig_m = px.bar(monthly, x='Date', y='AQI',
                       color='AQI', color_continuous_scale='RdYlGn_r',
                       labels={'AQI': 'Avg AQI', 'Date': 'Month'})
        fig_m.update_layout(height=350, showlegend=False,
                            xaxis_tickangle=45, xaxis_tickfont_size=9)
        st.plotly_chart(fig_m, use_container_width=True)

    with right:
        st.subheader('Key EDA Findings')
        st.info('🌧️ **Monsoon** is the cleanest season (lowest median AQI)')
        st.warning('❄️ **Winter** is the most polluted (temperature inversions trap pollutants)')
        st.success('💨 **Wind speed** is the strongest weather predictor (Spearman r = −0.307)')
        st.error('📊 **PM2.5** averages 35.8 µg/m³ — 2.4× the WHO daily guideline')
        st.subheader('AQI by Season')
        season_avg = (master.groupby('season')['AQI']
                      .mean().reset_index()
                      .sort_values('AQI', ascending=False))
        fig_s = px.bar(season_avg, x='season', y='AQI',
                       color='AQI', color_continuous_scale='RdYlGn_r')
        fig_s.update_layout(height=260, showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('---')
    st.subheader('Model Performance Summary')
    summary = pd.DataFrame({
        'Model':  ['OLS Linear Regression', 'ARIMA(p,d,q)', 'SARIMA',
                   'Decision Tree (tuned)', 'Random Forest (tuned)', 'Prophet (basic)'],
        'MAE':    [9.35, 29.13, 28.08, 11.13, 9.11, 53.03],
        'RMSE':   [12.00, 32.83, 31.86, None, None, 56.67],
        'R²':     [0.6239, -1.8158, -1.6514, 0.4683, 0.6181, -7.389],
        'Best?':  ['✅ Highest R²', '—', '—', '—', '✅ Lowest MAE', '⚠️ COVID break'],
    })
    st.dataframe(
        summary.style
            .highlight_min(subset=['MAE'], color='#c6efce')
            .highlight_max(subset=['R²'],  color='#c6efce')
            .format({'MAE': '{:.2f}', 'R²': '{:.3f}'}, na_rep='—'),
        use_container_width=True, hide_index=True
    )

# ── TRENDS ────────────────────────────────────────────────────────────────────
elif page == '📈 Trends':
    st.title('📈 Air Quality Trends')
    st.markdown(f'Showing **{len(filtered):,} days** based on your sidebar filters.')
    st.markdown('---')

    tab1, tab2, tab3 = st.tabs(['Time Series', 'Seasonal Patterns', 'Correlations'])

    with tab1:
        st.subheader('Daily AQI Over Time')
        metric = st.selectbox('Select metric', ['AQI', 'PM2.5'])
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=filtered['Date'], y=filtered[metric],
            mode='lines', name=metric,
            line=dict(color='steelblue', width=1), opacity=0.6
        ))
        fig_ts.add_trace(go.Scatter(
            x=filtered['Date'],
            y=filtered[metric].rolling(30).mean(),
            mode='lines', name='30-day avg',
            line=dict(color='red', width=2)
        ))
        fig_ts.update_layout(
            title=f'{metric} over time with 30-day rolling average',
            xaxis_title='Date', yaxis_title=metric,
            height=450, hovermode='x unified'
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        st.subheader('Seasonal Patterns')
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(filtered, x='season', y='AQI', color='season',
                category_orders={'season': ['Winter','Summer','Monsoon','Post-Monsoon']},
                title='AQI Distribution by Season',
                color_discrete_map={
                    'Winter':'#4878CF','Summer':'#E8601C',
                    'Monsoon':'#1DAA6D','Post-Monsoon':'#BC8F2F'
                })
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        with col_b:
            filtered = filtered.copy()
            filtered['year']  = filtered['Date'].dt.year
            filtered['month'] = filtered['Date'].dt.month
            monthly_heat = filtered.groupby(['year','month'])['AQI'].mean().reset_index()
            fig_heat = px.density_heatmap(monthly_heat, x='month', y='year',
                z='AQI', color_continuous_scale='RdYlGn_r',
                title='AQI Heatmap — Month × Year',
                labels={'month':'Month','year':'Year','AQI':'Avg AQI'})
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.subheader('Correlation with Weather Variables')
        weather_vars = ['wind_speed', 'rainfall', 'humidity']
        available_vars = [v for v in weather_vars if v in filtered.columns]
        x_var = st.selectbox('Weather variable (X axis)', available_vars)
        fig_sc = px.scatter(filtered, x=x_var, y='AQI', color='season',
            trendline='ols', opacity=0.4,
            color_discrete_map={
                'Winter':'#4878CF','Summer':'#E8601C',
                'Monsoon':'#1DAA6D','Post-Monsoon':'#BC8F2F'
            },
            title=f'AQI vs {x_var} coloured by season')
        fig_sc.update_layout(height=450)
        st.plotly_chart(fig_sc, use_container_width=True)
        num_cols = ['AQI', 'PM2.5'] + available_vars
        corr = filtered[num_cols].corr()
        st.write('**Pearson Correlation Matrix:**')
        st.dataframe(
            corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1)
                      .format('{:.3f}'),
            use_container_width=True
        )

# ── PREDICTIONS ───────────────────────────────────────────────────────────────
elif page == '🔮 Predictions':
    st.title('🔮 AQI Predictions')
    st.markdown('---')

    pred_tab1, pred_tab2 = st.tabs(['🎛️ What-If Simulator', '📅 Prophet Forecast'])

    with pred_tab1:
        st.subheader('What-If AQI Simulator')
        st.markdown(
            'Adjust the inputs below to see how weather conditions and '
            'recent pollution levels affect the **next-day AQI prediction**. '
            'Uses your trained OLS model coefficients.'
        )
        col1, col2 = st.columns(2)
        with col1:
            wind_input     = st.slider('💨 Wind Speed (km/h)', 0.0, 40.0, 17.0, 0.5)
            rainfall_input = st.slider('🌧️ Rainfall (mm)',     0.0, 50.0,  2.5, 0.5)
            lag1_input     = st.number_input('📅 Yesterday AQI (AQI_lag1)', 20, 350, 90)
        with col2:
            pm25_lag_input = st.number_input('🏭 Yesterday PM2.5 (PM25_lag1)', 1, 200, 35)
            season_input   = st.selectbox('🗓️ Season',
                ['Monsoon', 'Post-Monsoon', 'Summer', 'Winter'])
            dow_input      = st.selectbox('📆 Day of Week',
                ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

        # Actual OLS coefficients from your trained model
        intercept         =  40.0180
        coef_lag1         =   0.5457
        coef_wind         =  -0.3489
        coef_rain         =  -0.1420
        coef_pm25         =   0.3780
        coef_post_monsoon =  -3.7437
        coef_summer       =  -2.2950
        coef_winter       =  -0.5987
        coef_monday       =  -2.0460
        coef_saturday     =   0.7455
        coef_sunday       =  -2.2247
        coef_thursday     =  -4.5349
        coef_tuesday      =  -0.0281
        coef_wednesday    =  -2.4414

        predicted_aqi = (
            intercept
            + coef_lag1         * lag1_input
            + coef_wind         * wind_input
            + coef_rain         * rainfall_input
            + coef_pm25         * pm25_lag_input
            + coef_post_monsoon * (1 if season_input == 'Post-Monsoon' else 0)
            + coef_summer       * (1 if season_input == 'Summer'        else 0)
            + coef_winter       * (1 if season_input == 'Winter'        else 0)
            + coef_monday       * (1 if dow_input == 'Monday'    else 0)
            + coef_saturday     * (1 if dow_input == 'Saturday'  else 0)
            + coef_sunday       * (1 if dow_input == 'Sunday'    else 0)
            + coef_thursday     * (1 if dow_input == 'Thursday'  else 0)
            + coef_tuesday      * (1 if dow_input == 'Tuesday'   else 0)
            + coef_wednesday    * (1 if dow_input == 'Wednesday' else 0)
        )
        predicted_aqi = max(0, min(500, round(predicted_aqi, 1)))

        st.markdown('---')
        res1, res2 = st.columns(2)
        with res1:
            if predicted_aqi <= 50:
                st.success(f'### ✅ Predicted AQI: {predicted_aqi}\n**Good** — Air quality is satisfactory.')
            elif predicted_aqi <= 100:
                st.info(f'### 💙 Predicted AQI: {predicted_aqi}\n**Moderate** — Acceptable for most people.')
            elif predicted_aqi <= 150:
                st.warning(f'### ⚠️ Predicted AQI: {predicted_aqi}\n**Unhealthy for sensitive groups.**')
            elif predicted_aqi <= 200:
                st.warning(f'### 🟠 Predicted AQI: {predicted_aqi}\n**Unhealthy** — Everyone may be affected.')
            else:
                st.error(f'### 🔴 Predicted AQI: {predicted_aqi}\n**Very Unhealthy / Hazardous.**')
        with res2:
            st.markdown('**Inputs used:**')
            st.write(f'- Wind speed: **{wind_input} km/h**')
            st.write(f'- Rainfall: **{rainfall_input} mm**')
            st.write(f'- Yesterday AQI: **{lag1_input}**')
            st.write(f'- Yesterday PM2.5: **{pm25_lag_input} µg/m³**')
            st.write(f'- Season: **{season_input}**  |  Day: **{dow_input}**')
            st.markdown('---')
            st.caption('Model: OLS Linear Regression  |  Test MAE = 9.35 AQI units  |  R² = 0.624')

    with pred_tab2:
        st.subheader('Prophet 90-Day AQI Forecast')
        st.markdown('Trained on 80% of data. Forecasts 90 days beyond the training cutoff.')

        @st.cache_data
        def run_prophet(df):
            prophet_df = (df[['Date', 'AQI']]
                          .rename(columns={'Date': 'ds', 'AQI': 'y'})
                          .dropna())
            split   = int(len(prophet_df) * 0.80)
            train_p = prophet_df.iloc[:split]
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05
            )
            m.fit(train_p)
            future   = m.make_future_dataframe(periods=90, freq='D')
            forecast = m.predict(future)
            return forecast, train_p, prophet_df

        with st.spinner('Training Prophet model… first run takes ~30 seconds'):
            forecast, train_p, full_p = run_prophet(master)

        fig_proph = go.Figure()
        fig_proph.add_trace(go.Scatter(
            x=full_p['ds'], y=full_p['y'],
            mode='lines', name='Actual AQI',
            line=dict(color='steelblue', width=1.2)
        ))
        fig_proph.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Prophet Forecast',
            line=dict(color='red', width=2)
        ))
        fig_proph.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% Confidence Interval'
        ))
        fig_proph.add_vline(
            x=str(train_p['ds'].max()),
            line_dash='dash', line_color='gray',
            annotation_text='Train / Test split'
        )
        fig_proph.update_layout(
            title='Prophet AQI Forecast with 90-Day Future Projection',
            xaxis_title='Date', yaxis_title='AQI',
            height=500, hovermode='x unified'
        )
        st.plotly_chart(fig_proph, use_container_width=True)
        st.info(
            '⚠️ Prophet overestimates in the test period (Jun 2019–Jul 2020) due to '
            'COVID-19 lockdowns causing an unpredicted AQI drop. '
            'This is a documented project limitation, not a model failure.'
        )

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
elif page == '📊 Model Comparison':
    st.title('📊 Model Performance Comparison')
    st.markdown('Test set: **Jun 2019 – Jul 2020** (chronological 80/20 split)')
    st.markdown('---')

    results = pd.DataFrame({
        'Model':  ['OLS Linear Regression', 'ARIMA(p,d,q)', 'SARIMA',
                   'Decision Tree (tuned)', 'Random Forest (tuned)', 'Prophet (basic)'],
        'MAE':    [9.35,  29.13, 28.08, 11.13, 9.11,  53.03],
        'RMSE':   [12.00, 32.83, 31.86, None,  None,  56.67],
        'R²':     [0.6239, -1.8158, -1.6514, 0.4683, 0.6181, -7.389],
        'Notes':  [
            'DW=2.08; lag variables are key driver; highest R²',
            'Pure time-series; no weather features',
            'Captures annual seasonality explicitly',
            'Best depth=3 (GridSearchCV tuned)',
            'n_estimators=100 (GridSearchCV tuned); lowest MAE',
            'COVID-19 structural break inflates error on test period',
        ]
    })

    st.subheader('All Models — Test Set Metrics')
    st.dataframe(
        results.style
            .highlight_min(subset=['MAE'],  color='#c6efce')
            .highlight_max(subset=['R²'],   color='#c6efce')
            .highlight_min(subset=['RMSE'], color='#c6efce')
            .format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R²': '{:.4f}'}, na_rep='—'),
        use_container_width=True, hide_index=True
    )

    st.markdown('---')
    st.subheader('Visual Comparison (excluding Prophet)')
    results_viz = results[results['Model'] != 'Prophet (basic)'].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_mae = px.bar(results_viz, x='Model', y='MAE',
            color='MAE', color_continuous_scale='RdYlGn_r',
            title='MAE — lower is better')
        fig_mae.update_layout(height=380, showlegend=False, xaxis_tickangle=30)
        st.plotly_chart(fig_mae, use_container_width=True)
    with col2:
        results_rmse = results_viz.dropna(subset=['RMSE'])
        fig_rmse = px.bar(results_rmse, x='Model', y='RMSE',
            color='RMSE', color_continuous_scale='RdYlGn_r',
            title='RMSE — lower is better')
        fig_rmse.update_layout(height=380, showlegend=False, xaxis_tickangle=30)
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col3:
        fig_r2 = px.bar(results_viz, x='Model', y='R²',
            color='R²', color_continuous_scale='RdYlGn',
            title='R² — higher is better')
        fig_r2.update_layout(height=380, showlegend=False, xaxis_tickangle=30)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown('---')
    st.subheader('Feature Importance (Random Forest — Built-in)')
    feat_imp = {
        'AQI_lag1':            0.4380,
        'PM25_lag1':           0.2401,
        'wind_speed':          0.1215,
        'rainfall':            0.1055,
        'season_Winter':       0.0168,
        'dow_Saturday':        0.0115,
        'dow_Tuesday':         0.0105,
        'dow_Monday':          0.0104,
        'season_Summer':       0.0102,
        'season_Post-Monsoon': 0.0094,
        'dow_Wednesday':       0.0091,
        'dow_Sunday':          0.0090,
        'dow_Thursday':        0.0079,
    }
    feat_df = (pd.DataFrame({'Feature': list(feat_imp.keys()),
                             'Importance': list(feat_imp.values())})
               .sort_values('Importance', ascending=True))
    fig_fi = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='teal',
        title='Random Forest — Built-in Feature Importance')
    fig_fi.update_layout(height=460, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info(
        '**Key SHAP findings:** AQI_lag1 (yesterday\'s AQI) is the single strongest '
        'predictor — pollution persists day-to-day due to atmospheric residence time. '
        'Wind speed has a consistently negative SHAP value — it disperses pollutants. '
        'Winter season is a positive predictor due to temperature inversions trapping '
        'pollutants near ground level.'
    )

    st.markdown('---')
    st.subheader('SHAP Explainability Plots')
    shap_c1, shap_c2 = st.columns(2)
    with shap_c1:
        try:
            st.image('assets/plot_shap_beeswarm.png',
                     caption='SHAP Beeswarm — direction and magnitude per feature',
                     use_container_width=True)
        except Exception:
            st.info('Upload assets/plot_shap_beeswarm.png to GitHub to display this plot.')
    with shap_c2:
        try:
            st.image('assets/plot_shap_bar.png',
                     caption='SHAP Bar — mean absolute impact per feature',
                     use_container_width=True)
        except Exception:
            st.info('Upload assets/plot_shap_bar.png to GitHub to display this plot.')
