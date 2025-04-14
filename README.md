# 📈 Predicting Net Claims Incurred in the European Insurance Market

This project leverages data from the **European Insurance and Occupational Pension Authority (EIOPA)**, derived from the annual European insurance overview. The focus is on the **European life insurance sector**, utilizing **Solvency II annual reports**, which ensure consistent and reliable standards across all countries in the **European Economic Area (EEA)**.

The goal is to build predictive models that accurately forecast **net claims incurred**, enabling insurers to make data-driven decisions and proactively manage risk.

---

## 🎯 Business Objective

The primary objective is to use historical insurance data to develop machine learning and time series models capable of predicting net claims incurred. These predictions support strategic decision-making in pricing, capital allocation, and risk mitigation in the European insurance industry.

Understanding trends and external influences—like economic changes or catastrophic events—provides vital input into **insurance pricing strategies** and **solvency forecasting**.

---

## 🤖 Models Developed

A hybrid modeling approach was implemented to capture both short-term variations and long-term structural patterns in the data:

- **Machine Learning Models:**  These models were trained on a **combined dataset from 10 countries**, then **evaluated separately for each country** to test performance and generalization.
  - `XGBoost` — for powerful, high-performance gradient boosting
  - `LightGBM` — optimized for speed and efficiency on large-scale datasets

- **Time Series Models:** These were trained **individually for each of the 10 countries**, enabling them to capture localized trends and seasonality.
  - `ARIMA` — for analyzing and forecasting long-term time-based patterns
  - `Moving Average` — for capturing and smoothing short-term fluctuations

These models were trained on a rich set of features derived from the Solvency II reports, including economic indicators and operational metrics.
All models were tested in both **pre-tuned** and **Bayesian-optimized** versions for enhanced performance.

---

## ✅ Model Evaluation Strategy

To validate predictive performance:

- A **backtesting** approach was applied using **four quarters of unseen data**.
- Models were evaluated using:
  - **Accuracy**
  - **Bias** (forecast error deviation)
- For machine learning models, feature importance was also visualized to interpret drivers behind predictions.

---

## 🖥️ Final Product: Interactive Dashboard

The project culminated in a **user-friendly Dash/Plotly dashboard** featuring:

- **Six organized tabs**:
  1. Business Objective
  2. Data Preprocessing
  3. Exploratory Data Analysis
  4. Model Performance
  5. Model Robustness
  6. Business Solution

- **Key Features**:
  - Upload **multiple CSV files**
  - View **dynamic plots**, **tables**, and **KPIs**
  - Download predictions for **any selected country**
  - Compare model results interactively

