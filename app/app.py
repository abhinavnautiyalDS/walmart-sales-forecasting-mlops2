import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------------------
# CONFIG & STYLING
# ---------------------------
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
/* ============================
   Unified Streamlit Theme CSS
   Paste this entire block (replaces older CSS)
   ============================ */

/* ----------------------------
   Theme variables (single source of truth)
   ---------------------------- */
:root{
  --bg-100: #f5f7fa;        /* app background */
  --bg-200: #eef2f6;        /* panels / tabs background */
  --card-bg: #ffffff;       /* cards / metric background */
  --text-primary: #0f172a;  /* primary body text / headings */
  --text-secondary: #475569;/* muted / secondary text */
  --muted: #64748b;
  --primary: #0071dc;       /* brand primary */
  --primary-600: #005bb5;   /* darker primary for hover */
  --accent: #0ea5a4;        /* accent (optional) */
  --sidebar-bg: #0f172a;    /* navy sidebar */
  --border: #e6eef8;        /* subtle borders */
  --radius-lg: 12px;
  --radius-md: 8px;
  --elevation-1: 0 1px 6px rgba(16,24,40,0.06);
  --elevation-2: 0 6px 20px rgba(16,24,40,0.08);
  --metric-text: #0f172a;   /* strong dark for numbers */
  --metric-label: #475569;  /* muted label under number */
  --sidebar-text: #e6eef8;  /* text on dark sidebar */
}

/* ----------------------------
   Font
   ---------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: var(--text-primary);
  background-color: var(--bg-100);
}

/* Ensure Streamlit root uses bg */
.stApp {
  background-color: var(--bg-100) !important;
  padding: 1.25rem;
}

/* ----------------------------
   Header / Hero
   ---------------------------- */
.main-header {
  text-align: center;
  padding: 2rem 1.5rem;
  margin-bottom: 1.75rem;
  background: linear-gradient(135deg, var(--primary) 0%, #0055a4 100%);
  color: #ffffff;
  border-radius: var(--radius-lg);
  box-shadow: var(--elevation-2);
}

.main-header h1 {
  font-size: 2.6rem;
  font-weight: 800;
  margin: 0;
  line-height: 1;
  letter-spacing: -0.6px;
  color: #ffffff;
}

.main-header p {
  margin: .5rem 0 0;
  font-size: 1.05rem;
  opacity: 0.95;
}

/* ----------------------------
   Cards & Panels
   ---------------------------- */
.card, .panel, .stCard, .stContainer, [data-testid="stMetric"], [data-testid="stTable"] {
  background: var(--card-bg) !important;
  border-radius: var(--radius-md);
  border: 1px solid var(--border) !important;
  box-shadow: var(--elevation-1);
}

/* Generic spacing inside cards */
.card, .panel, .stCard, .stContainer {
  padding: 1rem;
}

/* ----------------------------
   Metric Cards — Strong, readable
   ---------------------------- */
/* Ensure metric wrapper is clean */
[data-testid="stMetric"] {
  padding: 1.25rem !important;
  border-radius: var(--radius-md) !important;
  transition: transform .18s ease, box-shadow .18s ease;
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
}

/* Hover lift */
[data-testid="stMetric"]:hover {
  transform: translateY(-4px);
  box-shadow: var(--elevation-2);
}

/* Force high-contrast numbers and labels inside metrics */
/* Broad coverage for different Streamlit versions */
[data-testid="stMetric"], [data-testid="stMetric"] * ,
.stMetric, .stMetric * , .stMetricValue, .stMetricLabel,
[data-testid="stMetric"] .css-1w6e6r2, /* fallback internal class names */
[data-testid="stMetric"] .css-1h2m3xm,
[data-testid="stMetric"] [data-testid] {
  color: var(--metric-text) !important;
  -webkit-text-fill-color: var(--metric-text) !important;
  opacity: 1 !important;
  background: transparent !important;
  font-weight: 700 !important;
}

/* Metric label styling (smaller, muted) */
[data-testid="stMetric"] label, [data-testid="stMetric"] .stMetricLabel,
.stMetricLabel {
  color: var(--metric-label) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  opacity: 1 !important;
}

/* Prevent inline-styles from forcing pale colours */
[data-testid="stMetric"] [style] {
  color: var(--metric-text) !important;
  -webkit-text-fill-color: var(--metric-text) !important;
  opacity: 1 !important;
}

/* Badge inside metric (forecast, trend) */
[data-testid="stMetric"] .stBadge, [data-testid="stMetric"] .badge, [data-testid="stMetric"] .css-1q8dd3e {
  color: #065f46 !important;
  background: #ecfdf5 !important;
  font-weight: 700 !important;
  border-radius: 999px !important;
  padding: 0.15rem 0.4rem !important;
  display: inline-block;
}

/* ----------------------------
   Tabs
   ---------------------------- */
.stTabs [data-baseweb="tab-list"] {
  display:flex;
  gap: 18px;
  background-color: var(--bg-200);
  padding: 0.85rem 1rem 0.6rem 1rem;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
  height: 44px;
  white-space: nowrap;
  background: transparent;
  border-radius: 8px;
  color: var(--muted);
  font-weight: 600;
  font-size: 14px;
  padding: 8px 14px;
  display:inline-flex;
  align-items:center;
}

.stTabs [aria-selected="true"] {
  color: var(--primary);
  border-bottom: 3px solid var(--primary);
  background: var(--card-bg);
}

/* ----------------------------
   Buttons & Inputs
   ---------------------------- */
.stButton > button {
  background-color: var(--primary) !important;
  color: white !important;
  border: none !important;
  padding: 0.6rem 1.1rem !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  transition: background-color .14s ease, box-shadow .14s ease;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

.stButton > button:hover {
  background-color: var(--primary-600) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
}

/* Inputs style (broad selectors) */
input[type="text"], input[type="number"], textarea, .stTextInput, .stTextArea, .stSelectbox, .stMultiSelect, select, .css-1y0tads {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: var(--card-bg) !important;
  padding: 0.5rem 0.65rem !important;
  color: var(--text-primary) !important;
  box-shadow: none !important;
}

input[type="text"]:focus, input[type="number"]:focus, textarea:focus, .stTextInput:focus {
  outline: none !important;
  box-shadow: 0 8px 30px rgba(0,113,220,0.08) !important;
  border-color: var(--primary) !important;
}

/* Form labels */
label, .stMarkdown legend, .css-1v4p2jh {
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
}

/* ----------------------------
   Sidebar
   ---------------------------- */
[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
  color: var(--sidebar-text) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
  padding: 1rem !important;
  min-width: 240px;
}

/* Ensure children readable */
[data-testid="stSidebar"] * {
  color: var(--sidebar-text) !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #ffffff !important;
  margin: 0 0 0.35rem 0;
}

/* ----------------------------
   Tables
   ---------------------------- */
[data-testid="stTable"] {
  border-radius: 10px;
  border: 1px solid var(--border);
  overflow: hidden;
  background: var(--card-bg);
}

table {
  color: var(--text-primary);
}

table thead th {
  background: var(--bg-200);
  color: var(--text-secondary);
  font-weight: 700;
}

/* ----------------------------
   Links & small text
   ---------------------------- */
a, .stMarkdown a {
  color: var(--primary) !important;
  text-decoration: none !important;
  font-weight: 600 !important;
}

a:hover, .stMarkdown a:hover {
  color: var(--primary-600) !important;
  text-decoration: underline !important;
}

.small-text, .muted, .stCaption, .markdown-text-container p small {
  color: var(--text-secondary) !important;
  font-size: 0.92rem;
}

/* Headings */
h1, h2, h3, h4 {
  color: var(--text-primary) !important;
  font-weight: 800;
  margin: 0.2rem 0 0.6rem 0;
}

/* Accessibility: focus outlines for keyboard users */
:focus {
  outline: 3px solid rgba(0,113,220,0.12);
  outline-offset: 2px;
}

/* Selection highlight */
::selection {
  background: rgba(0,113,220,0.9);
  color: #ffffff;
}

/* ----------------------------
   Optional Dark Theme
   Use by adding class "theme-dark" to .stApp or root element
   ---------------------------- */
.theme-dark, .theme-dark .stApp {
  --bg-100: #071029;
  --bg-200: #0b1726;
  --card-bg: rgba(255,255,255,0.03);
  --text-primary: #e6eef8;
  --text-secondary: #b7c6d9;
  --primary: #3aa0ff;
  --primary-600: #0d74d0;
  --accent: #3cd6c9;
  --sidebar-bg: #021124;
  --border: rgba(255,255,255,0.06);
  --metric-text: #e6eef8;
  --metric-label: #b7c6d9;
}

/* Ensure dark mode sidebar text inherits */
.theme-dark [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.theme-dark a, .theme-dark .stMarkdown a { color: var(--primary) !important; }

/* ----------------------------
   Responsive tweaks
   ---------------------------- */
@media (max-width: 720px) {
  .main-header h1 { font-size: 1.6rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 10px; padding: .6rem; }
  .stApp { padding: 0.6rem; }
}
            
/* ===== FIX: Make all markdown text visible ===== */

/* General markdown text */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, 
.markdown-text-container p, .markdown-text-container li {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* Bold text */
.stMarkdown strong, .markdown-text-container strong {
    color: var(--text-primary) !important;
}

/* Inline code, links */
.stMarkdown code, .markdown-text-container code {
    color: var(--text-primary) !important;
}

/* Headings inside markdown */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
.markdown-text-container h1, .markdown-text-container h2, .markdown-text-container h3 {
    color: var(--text-primary) !important;
}

/* Fix unordered & ordered lists */
.stMarkdown ul li, .stMarkdown ol li,
.markdown-text-container ul li, .markdown-text-container ol li {
    color: var(--text-primary) !important;
}

/* Important: override Streamlit's inline styles */
.stMarkdown [style], .markdown-text-container [style] {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* End of CSS */
</style>
""", unsafe_allow_html=True)




# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open("app/model1.pkl", "rb") as f:
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
    # Use a more stable image URL or handle error
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/2560px-Walmart_logo.svg.png", width=180)
    except:
        st.markdown("## Walmart Sales Predictor")
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("⚙️ Configuration")
    
    st.subheader("1. Data Input")
    uploaded = st.file_uploader("Upload Store CSV", type=["csv"], help="Upload weekly store data for analysis")
    
    st.markdown("---")
    
    st.subheader("2. Resources")
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
        "📥 Download Template",
        sample_df.to_csv(index=False),
        file_name="walmart_data_template.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>Walmart Sales Predictor</h1>
    <p> Machine Learning Forecasting & Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("👈 **Start Here**\n\nUpload your store data CSV in the sidebar to generate predictions.")
    with col2:
        st.markdown("""
        ### How it works
        1. **Upload Data**: Provide your weekly store metrics
        2. **Auto-Processing**: System cleans and engineers features
        3. **AI Prediction**: Advanced model forecasts weekly sales
        4. **Analytics**: Explore trends and future projections
        """)
    st.stop()

if model is None:
    st.error("⚠️ Model file 'model (4).pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Process Data
df = pd.read_csv(uploaded)

with st.spinner("🔄 Analyzing store data..."):
    # Feature Engineering
    df_fe = feature_engineering(df.copy())
    
    # Prediction
    df_fe["Weekly_Sales_Pred"] = model.predict(df_fe)

# Layout: Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🔍 Deep Dive Analysis", "🔮 Future Forecast"])

with tab1:
    st.markdown("### Performance Summary")
    
    # Key Metrics
    total_sales = df_fe["Weekly_Sales_Pred"].sum()
    avg_sales = df_fe["Weekly_Sales_Pred"].mean()
    peak_week = df_fe.loc[df_fe["Weekly_Sales_Pred"].idxmax()]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Projected Revenue", f"${total_sales:,.0f}", delta="Forecast")
    m2.metric("Avg Weekly Sales", f"${avg_sales:,.2f}")
    m3.metric("Peak Sales Volume", f"${peak_week['Weekly_Sales_Pred']:,.0f}", f"Week {peak_week['Week']}")
    
    st.markdown("---")
    
    # Main Trend Chart
    st.subheader("Sales Trajectory")
    
    # Filter Controls
    c1, c2 = st.columns([1, 3])
    with c1:
        depts = sorted(df_fe["Dept"].unique())
        choice = st.selectbox("Filter by Department", ["All Departments"] + list(depts))

    # Data Filtering
    if choice == "All Departments":
        plot_data = df_fe.groupby("Date")["Weekly_Sales_Pred"].sum().reset_index()
    else:
        plot_data = df_fe[df_fe["Dept"] == choice]
        
    # Interactive Plotly Chart
    fig = px.area(plot_data, x="Date", y="Weekly_Sales_Pred",
                 title="Weekly Sales Forecast",
                 labels={"Weekly_Sales_Pred": "Predicted Revenue ($)"},
                 color_discrete_sequence=["#0071dc"])
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, l=10, r=10, b=10),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Detailed Breakdown")
    
    row1_1, row1_2 = st.columns(2)
    
    with row1_1:
        st.markdown("#### Top Performing Departments")
        dept_perf = df_fe.groupby("Dept")["Weekly_Sales_Pred"].mean().reset_index().sort_values("Weekly_Sales_Pred", ascending=False).head(10)
        fig_dept = px.bar(dept_perf, x="Weekly_Sales_Pred", y="Dept", orientation='h',
                         color="Weekly_Sales_Pred", color_continuous_scale="Blues")
        fig_dept.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with row1_2:
        st.markdown("#### Store Performance Distribution")
        fig_store = px.box(df_fe, x="Store", y="Weekly_Sales_Pred", color="Store")
        st.plotly_chart(fig_store, use_container_width=True)
        
    st.markdown("---")
    
    row2_1, row2_2 = st.columns(2)
    with row2_1:
        st.markdown("#### Markdown Impact Analysis")
        fig_md = px.scatter(df_fe, x="Total_Markdown", y="Weekly_Sales_Pred", 
                           color="Type", opacity=0.6,
                           labels={"Total_Markdown": "Promotional Markdowns ($)"})
        st.plotly_chart(fig_md, use_container_width=True)
        
    with row2_2:
        st.markdown("#### Economic Sensitivity (CPI)")
        # Removed trendline="ols" to fix ModuleNotFoundError: No module named 'statsmodels'
        fig_cpi = px.scatter(df_fe, x="CPI", y="Weekly_Sales_Pred", 
                            labels={"CPI": "Consumer Price Index"})
        st.plotly_chart(fig_cpi, use_container_width=True)

with tab3:
    st.markdown("### 🔮 Future Scenario Planning")
    
    with st.container():
        col_in, col_viz = st.columns([1, 2])
        
        with col_in:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">
                <h4>Forecast Settings</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Generate forward-looking predictions based on current trends.</p>
            </div>
            """, unsafe_allow_html=True)
            
            future_weeks = st.slider("Forecast Horizon (Weeks)", 4, 52, 12)
            generate_btn = st.button("🚀 Generate Forecast", use_container_width=True)

        if generate_btn:
            # Start with last ORIGINAL ROW (with Date)
            last_raw = df.tail(1).copy()
            last_raw["Date"] = pd.to_datetime(last_raw["Date"])
            future_predictions = []

            for i in range(future_weeks):
                last_raw.loc[last_raw.index[0], "Date"] = last_raw["Date"].iloc[0] + pd.Timedelta(weeks=1)
                temp = last_raw.copy()
                temp_fe = feature_engineering(temp)

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

                pred = model.predict(temp_fe)[0]
                future_predictions.append({
                    "Date": temp["Date"].iloc[0],
                    "Predicted_Weekly_Sales": pred
                })
                last_raw["Weekly_Sales_Pred"] = pred

            future_df_forecast = pd.DataFrame(future_predictions)
            
            with col_viz:
                st.success(f"✅ Generated forecast for next {future_weeks} weeks")
                fig_forecast = px.line(future_df_forecast, x="Date", y="Predicted_Weekly_Sales",
                                     markers=True, line_shape="spline")
                fig_forecast.update_traces(line_color="#00C853", line_width=3)
                fig_forecast.add_annotation(x=future_df_forecast['Date'].iloc[-1], 
                                          y=future_df_forecast['Predicted_Weekly_Sales'].iloc[-1],
                                          text="Projected", showarrow=True, arrowhead=1)
                st.plotly_chart(fig_forecast, use_container_width=True)

            st.markdown("#### Forecast Data Table")
            st.dataframe(future_df_forecast, use_container_width=True)
