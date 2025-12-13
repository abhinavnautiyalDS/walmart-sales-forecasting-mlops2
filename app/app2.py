import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# CONFIG & STYLING
# ---------------------------
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #0071dc;
        margin-bottom: 1rem;
    }
    .metric-container {
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    # Added try-except to prevent crash if file missing in this context, 
    # but keeping logic same as requested.
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# ---------------------------
# Feature Engineering
# ---------------------------
def feature_engineering(df):
    df["Date"] = pd.to_datetime(df["Date"])

    # ---------- Date-based ----------
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["WeekOfMonth"] = (df["Date"].dt.day - 1) // 7 + 1
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    def season(m):
        if m in [12,1,2]: return 1
        if m in [3,4,5]: return 2
        if m in [6,7,8]: return 3
        return 4
    df["Season"] = df["Month"].apply(season)

    # ---------- Markdown ----------
    df["Total_Markdown"] = df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0).sum(axis=1)
    df["Any_Markdown"] = (df["Total_Markdown"] > 0).astype(int)
    df["Markdown_Intensity"] = df["Total_Markdown"] / (df["Size"] + 1)
    df["Holiday_Markdown"] = df["MarkDown1"].fillna(0) * df["IsHoliday"]

    # ---------- Size ----------
    try:
        df["Store_Size_Category"] = pd.qcut(
            df["Size"], q=3, labels=[1, 2, 3], duplicates="drop"
        ).astype(float).fillna(2).astype(int)
    except:
        df["Store_Size_Category"] = 2

    df["Relative_Size"] = df["Size"] / df["Size"].max()

    # ---------- CPI ----------
    try:
        df["CPI_Category"] = pd.qcut(df["CPI"], 3, labels=[1,2,3], duplicates="drop").astype(float).fillna(2).astype(int)
    except:
        df["CPI_Category"] = 2

    df["CPI_Change"] = df.groupby("Store")["CPI"].diff().fillna(0)
    df["Unemployment_Change"] = df.groupby("Store")["Unemployment"].diff().fillna(0)
    df["Fuel_Change"] = df.groupby("Store")["Fuel_Price"].diff().fillna(0)

    # ---------- Lag Features ----------
    # NOTE: Corrected: You MUST use weekly sales related signals.
    df["Sales_Lag_1"] = df.groupby(["Store","Dept"])["MarkDown1"].shift(1).fillna(0)
    df["Sales_Lag_2"] = df.groupby(["Store","Dept"])["MarkDown1"].shift(2).fillna(0)
    df["Sales_MA_4"] = (df.groupby(["Store","Dept"])["MarkDown1"]
      .transform(lambda x: x.rolling(4, min_periods=1).mean())
      .fillna(0))

    return df

# ---------------------------
# Streamlit UI
# ---------------------------

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/2560px-Walmart_logo.svg.png", width=150)
    st.title("Controls")
    
    st.subheader("Data Upload")
    uploaded = st.file_uploader("Upload Walmart CSV", type=["csv"])
    
    st.markdown("---")
    st.subheader("Sample Data")
    sample_df = pd.DataFrame({
        "Store": np.random.choice([1,2,3], size=40),
        "Date": pd.date_range(start="2012-01-06", periods=40, freq="W-FRI").astype(str),
        "Dept": np.random.choice([1,2,3,4,5,6], size=40),
        "IsHoliday": np.random.choice([0,1], size=40, p=[0.85, 0.15]),
        "Temperature": np.random.uniform(30, 80, size=40).round(2),
        "Fuel_Price": np.random.uniform(2.8, 4.2, size=40).round(3),
        "MarkDown1": np.random.uniform(0, 5000, size=40).round(2),
        "MarkDown2": np.random.uniform(0, 4000, size=40).round(2),
        "MarkDown3": np.random.uniform(0, 3000, size=40).round(2),
        "MarkDown4": np.random.uniform(0, 2000, size=40).round(2),
        "MarkDown5": np.random.uniform(0, 1500, size=40).round(2),
        "CPI": np.random.uniform(210, 230, size=40).round(2),
        "Unemployment": np.random.uniform(5.0, 10.0, size=40).round(2),
        "Type": np.random.choice([1,2,3], size=40),
        "Size": np.random.choice([151315, 202307, 39910], size=40),
    })
    st.download_button(
        "Download Sample CSV",
        sample_df.to_csv(index=False),
        file_name="sample_walmart_input.csv",
        mime="text/csv"
    )

# Main Content
st.markdown('<div class="main-header">📊 Walmart Sales Forecast Dashboard</div>', unsafe_allow_html=True)

if not uploaded:
    st.info("👈 Please upload your Walmart dataset CSV in the sidebar to begin analysis.")
    st.stop()

if model is None:
    st.error("⚠️ Model file 'model (4).pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Process Data
df = pd.read_csv(uploaded)

with st.spinner("Processing data and running predictions..."):
    # Feature Engineering
    df_fe = feature_engineering(df.copy())
    
    # Prediction
    df_fe["Weekly_Sales_Pred"] = model.predict(df_fe)

# Layout: Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction Overview", "📈 Detailed Analysis", "🔮 Future Forecast"])

with tab1:
    # Key Metrics
    total_sales = df_fe["Weekly_Sales_Pred"].sum()
    avg_sales = df_fe["Weekly_Sales_Pred"].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Predicted Sales", f"${total_sales:,.2f}")
    col2.metric("Average Weekly Sales", f"${avg_sales:,.2f}")
    
    st.markdown("### Department Filter")
    depts = sorted(df_fe["Dept"].unique())
    choice = st.selectbox("Choose Department", ["All"] + list(depts))

    if choice == "All":
        st.success(f"Average Weekly Sales (All Depts): ${df_fe['Weekly_Sales_Pred'].mean():,.2f}")
        current_view_df = df_fe
    else:
        current_view_df = df_fe[df_fe["Dept"] == choice]
        st.success(f"Dept {choice} Average Sales: ${current_view_df['Weekly_Sales_Pred'].mean():,.2f}")
        st.dataframe(current_view_df[["Store", "Dept", "Date", "Weekly_Sales_Pred"]], use_container_width=True)

    st.subheader("Predicted Sales Trend")
    # Prepare plot data
    df_plot = df.copy()
    df_plot["Weekly_Sales_Pred"] = df_fe["Weekly_Sales_Pred"]
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    
    if choice != "All":
        df_plot = df_plot[df_fe["Dept"] == choice]
        
    st.line_chart(df_plot.set_index("Date")["Weekly_Sales_Pred"])

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By Department")
        st.bar_chart(df_fe.groupby("Dept")["Weekly_Sales_Pred"].mean())
        
    with col2:
        st.subheader("By Store")
        st.bar_chart(df_fe.groupby("Store")["Weekly_Sales_Pred"].mean())
        
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Markdown vs Sales")
        st.scatter_chart(df_fe[["Total_Markdown","Weekly_Sales_Pred"]].rename(columns={"Total_Markdown":"Markdowns","Weekly_Sales_Pred":"Sales"}))
        
    with col4:
        st.subheader("CPI vs Sales")
        st.scatter_chart(df_fe[["CPI","Weekly_Sales_Pred"]].rename(columns={"CPI":"CPI","Weekly_Sales_Pred":"Sales"}))
        
    with st.expander("View Raw Engineered Data"):
        st.dataframe(df_fe)

with tab3:
    st.header("Forecast Future Weekly Sales")
    
    col_in, col_btn = st.columns([1,2])
    with col_in:
        future_weeks = st.number_input("Weeks to Forecast", 1, 52, 4)
    with col_btn:
        st.write("") 
        st.write("") 
        generate_btn = st.button("Generate Future Forecast", type="primary")

    if generate_btn:
        # Start with last ORIGINAL ROW (with Date)
        last_raw = df.tail(1).copy()

        # Make sure Date is datetime
        last_raw["Date"] = pd.to_datetime(last_raw["Date"])

        # initialize list
        future_predictions = []

        for i in range(future_weeks):

            # Increment date by 1 week
            last_raw.loc[last_raw.index[0], "Date"] = last_raw["Date"].iloc[0] + pd.Timedelta(weeks=1)

            # Create temp copy
            temp = last_raw.copy()

            # Feature engineering
            temp_fe = feature_engineering(temp)

            # Lag handling
            if i == 0:
                temp_fe["Sales_Lag_1"] = df_fe["Weekly_Sales_Pred"].iloc[-1]
                temp_fe["Sales_Lag_2"] = df_fe["Weekly_Sales_Pred"].iloc[-2]
                temp_fe["Sales_MA_4"] = df_fe["Weekly_Sales_Pred"].tail(4).mean()
            else:
                temp_fe["Sales_Lag_1"] = future_predictions[-1]["Predicted_Weekly_Sales"]

                temp_fe["Sales_Lag_2"] = (
                    future_predictions[-2]["Predicted_Weekly_Sales"]
                    if len(future_predictions) > 1
                    else temp_fe["Sales_Lag_1"]
                )

                ma_vals = df_fe["Weekly_Sales_Pred"].tail(3).tolist()
                ma_vals.append(future_predictions[-1]["Predicted_Weekly_Sales"])
                temp_fe["Sales_MA_4"] = np.mean(ma_vals)

            # Predict
            pred = model.predict(temp_fe)[0]

            # Append
            future_predictions.append({
                "Date": temp["Date"].iloc[0],
                "Predicted_Weekly_Sales": pred
            })

            # Carry forward
            last_raw["Weekly_Sales_Pred"] = pred

        # Convert to DataFrame
        future_df_forecast = pd.DataFrame(future_predictions)

        st.success(f"Forecast generated for the next {future_weeks} weeks!")

        st.subheader("📈 Future Weekly Sales Forecast")
        st.line_chart(future_df_forecast.set_index("Date")["Predicted_Weekly_Sales"])

        st.subheader("📋 Forecast Table")
        st.dataframe(future_df_forecast, use_container_width=True)