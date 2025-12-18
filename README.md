# **Walmart Retail Demand Forecasting System**
### Building a Production-Ready Walmart Sales Forecasting Pipeline using Machine Learning & MLOps

---

## **Introduction**
### Why this project exists

Machine learning projects often look impressive on the surface — a trained model, a good score, maybe even a deployed demo. But what usually remains hidden is **how decisions were made**, how experiments were tracked, and whether the system can actually be used outside a notebook.

This project focuses on building a **real-world retail demand forecasting system** using Walmart sales data. The goal was not just to predict weekly sales, but to design an **end-to-end machine learning pipeline** that is reproducible, explainable, and deployable.

---

## **Project Objective**
### From model training to system building

The main objective of this project was to move from:

**“I trained a machine learning model”**  
to  
**“I built a machine learning system.”**

This includes:
- Understanding how retail sales data behaves
- Making modelling decisions based on data insights
- Tracking experiments systematically
- Ensuring reproducibility using MLOps tools
- Deploying the model as a usable application

---

## **Data Source**
### Walmart sales dataset from Kaggle

The dataset used in this project was sourced from **Kaggle** and contains historical Walmart sales data.

The data includes:
- Weekly sales values (target variable)
- Store and department identifiers
- Holiday indicators
- Macroeconomic features such as:
  - CPI
  - Unemployment rate
  - Fuel price
  - Temperature
- Promotional markdown information

This dataset closely resembles real retail demand data with seasonality and external economic effects.

---

## **Data Preprocessing**
### Cleaning with business context in mind

Data preprocessing was performed carefully to **preserve business meaning** rather than blindly cleaning the data.

Key preprocessing steps:
- Merging multiple datasets (`train`, `stores`, `features`) into a single table
- Handling missing values based on domain understanding
- Removing duplicate records
- Correcting data types, especially date fields
- Preserving extreme sales values caused by holidays and promotions

### **Important Insight**
Missing values in markdown columns do not indicate missing data — they indicate **no discount applied**. These values were filled with zero instead of being dropped or averaged.

---

## **Exploratory Data Analysis (EDA)**
### Understanding sales behaviour before modelling

EDA was used to understand the underlying patterns in the data.

Key findings:
- Weekly sales distribution is heavily right-skewed
- Strong seasonality exists across the year
- Holiday weeks consistently produce large sales spikes
- Macroeconomic variables act as contextual signals rather than direct predictors
- A small number of departments contribute disproportionately to total sales

These insights directly influenced feature engineering and model selection.

---

## **Feature Engineering**
### Translating business behaviour into features

Feature engineering was driven by **data understanding and retail logic**, not feature quantity.

Engineered features include:
- **Temporal features**: week, month, year, season
- **Lag features**: previous week and two-week lag sales
- **Rolling features**: moving averages to capture trends
- **Categorical features**: store, department, holiday
- **Promotion features**: total markdown, markdown intensity
- **Economic indicators**: CPI change, unemployment change, fuel price change

Each feature was added with a clear justification related to retail demand behaviour.

---

## **Modeling & MLflow Experimentation**
### Structured experimentation using MLOps

Instead of running untracked experiments, **MLflow** was used to manage the modelling process.

MLflow was used for:
- Tracking preprocessing strategies (scaling, encoding)
- Comparing different regression algorithms
- Logging parameters, metrics, and model artifacts
- Maintaining reproducibility across experiments

### **Evaluation Strategy**
- Time-based train–validation split to prevent data leakage
- Metrics used:
  - RMSE
  - MAE
  - R² score

This approach ensured fair and reliable model comparisons.

---

## **Docker Containerisation**
### Ensuring reproducibility across environments

The complete application was containerised using **Docker**.

The Docker image includes:
- Data preprocessing pipeline
- Trained machine learning model
- Streamlit application
- All required dependencies

This ensures that the project runs consistently across different machines and platforms.

---

## **Deployment**
### Streamlit application on Hugging Face Spaces

The final model was deployed as an interactive **Streamlit web application** hosted on **Hugging Face Spaces**.

The deployed app allows users to:
- Input relevant features
- Get real-time weekly sales predictions
- Interact with the model without running code locally

Deployment helped uncover practical issues that are not visible during notebook experimentation.

---

## **Project Structure**
### Repository layout

