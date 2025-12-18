# **Walmart Retail Demand Forecasting System**
![12660749993_3167dcef09_k](https://github.com/user-attachments/assets/68a3f328-2027-47b1-a223-fb45906aebec)

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
<img width="1948" height="848" alt="download (1)" src="https://github.com/user-attachments/assets/746a1cf1-f84b-463b-ae3c-b4094879ba18" />

- Strong seasonality exists across the year
<img width="1948" height="848" alt="download 2" src="https://github.com/user-attachments/assets/a004384c-87c3-4ee9-97a0-5a8305993668" />

- Holiday weeks consistently produce large sales spikes
<img width="1697" height="824" alt="download 3" src="https://github.com/user-attachments/assets/eb3559ab-89af-4794-a331-229c46b59414" />

- Macroeconomic variables act as contextual signals rather than direct predictors
<img width="1948" height="847" alt="download 7" src="https://github.com/user-attachments/assets/7311dc2c-369f-4afe-8a04-7f0c52948ff8" />
<img width="1711" height="824" alt="download 8" src="https://github.com/user-attachments/assets/f3fb7eee-5998-462f-901d-1f167b302d70" />
<img width="1711" height="824" alt="download 9" src="https://github.com/user-attachments/assets/3b39b48f-476b-4669-91fa-faf7b761a8e3" />


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
<img width="1550" height="543" alt="Screenshot 2025-12-17 230543" src="https://github.com/user-attachments/assets/44d22c27-03e2-48c9-93ce-004f903f5d35" />

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
- Input csv file 
- Get real-time weekly sales predictions, along with insights and can perform future forcasting
- Interact with the model without running code locally
![App demo (online-video-cutter com) (online-video-cutter com) (2)](https://github.com/user-attachments/assets/c88a0381-c552-4089-bb6f-a088417e9042)

├── data/
│   ├── scripts/
          
│   └── dataset/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── mlflow/
├── Dockerfile
├── requirements.txt
└── README.md

Deployment helped uncover practical issues that are not visible during notebook experimentation.

---

## **Project Structure**
### Repository layout

