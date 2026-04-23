# Bangalore AQI Project
Predictive Modeling of AQI and PM2.5 Levels in Bangalore
A Statistical + Machine Learning Approach

## Project Structure
notebooks/Bengaluru AQI project.ipynb (google colab file)

## Datasets Used
- Rohanrao: India Air Quality Data (Kaggle) — AQI, PM2.5, PM10
- Open-Meteo API — Temperature, humidity, wind speed, rainfall

## Setup
pip install -r requirements.txt

## Data Download
1. Set up Kaggle API credentials (see notebook Cell 3)
2. Run notebooks in order starting from 01_data_collection.ipynb
3. Weather data fetched automatically via Open-Meteo API

## Output
processed/bangalore_master_aqi.csv — master dataset ready for EDA
