# **Walmart Sales Forecasting Pipeline**
### End-to-End Machine Learning System using Machine Learning and MLOps

---

## **Project Overview**
### From raw retail data to a deployed forecasting system

This repository contains an end-to-end machine learning project focused on forecasting Walmart’s weekly sales at the store–department level.

The project goes beyond model training and focuses on building a **reliable, reproducible, and deployable machine learning system**.

---

## **Project Goal**
### Building a usable ML system, not just a model

The primary goal of this project was to move from:

**“I trained a model” → “I built a machine learning system”**

This includes:
- Understanding retail demand behaviour
- Systematic experimentation instead of random trials
- Tracking experiments using MLflow
- Packaging and deploying the solution

---

## **Data Gathering**
### Retail sales data sourced from Kaggle

The dataset was collected from a publicly available **Walmart sales dataset on Kaggle**.

It includes:
- Weekly sales data
- Store and department identifiers
- Holiday indicators
- Macroeconomic variables (CPI, unemployment, fuel price, temperature)

---

## **Data Preprocessing**
### Cleaning while preserving business meaning

Data preprocessing focused on validation and consistency rather than aggressive cleaning.

Steps performed:
- Merging multiple datasets into a single table
- Handling missing values carefully
- Removing duplicate records
- Fixing data types (especially dates)
- Preserving holiday and promotion-driven spikes

Missing markdown values were treated as **no discount applied**, not missing data.

---

## **Exploratory Data Analysis (EDA)**
### Understanding sales behaviour before modelling

Key insights from EDA:
- Weekly sales show heavy right skew
- Strong seasonality across the year
- Holiday weeks consistently produce sales spikes
- Macroeconomic variables provide contextual signals
- Some departments dominate total sales contribution

These insights guided feature engineering and model selection.

---

## **Feature Engineering**
### Features driven by data understanding and logic

Engineered features include:
- Time-based features (week, month, year, season)
- Lag features for previous weeks’ sales
- Rolling averages to capture trends
- Categorical encodings for stores and departments
- Promotion and markdown-related features
- Economic change indicators (CPI, unemployment, fuel)

Each feature was added with clear justification.

---

## **MLflow Experimentation**
### Structured and reproducible model experimentation

MLflow was used for:
- Tracking preprocessing strategies (scaling and encoding)
- Comparing regression models
- Logging parameters, metrics, and artifacts
- Ensuring reproducibility across runs

Models were evaluated using:
- RMSE
- MAE
- R²

Time-aware validation was used to avoid data leakage.

---

## **Docker Containerisation**
### Making the system portable and reproducible

The entire application was containerised using **Docker**.

The Docker image includes:
- Preprocessing pipeline
- Trained ML model
- Streamlit application
- All required dependencies

This ensures consistent behaviour across environments.

---

## **Deployment**
### Streamlit application deployed on Hugging Face

The final model was deployed as an interactive **Streamlit application** on **Hugging Face Spaces**.

The app allows:
- User input of features
- Real-time weekly sales prediction
- Consistent preprocessing during inference

---

## **Repository Structure**
### Project file organisation

