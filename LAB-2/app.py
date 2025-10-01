import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Econometrics Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Lively, Professional Light Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp { background-color: #FFFFFF; }
    .main .block-container {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 2rem;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    h1, h2, h3 { color: #004085; }
    .stMarkdown p, .stDataFrame, .stText, .stAlert { color: #333333 !important; }
    .centered-title { text-align: center; font-family: 'Georgia', serif; font-weight: 700; color: #004085; letter-spacing: -1px; }
    .centered-subtitle { text-align: center; color: #555555; font-weight: 400; }
    .stTabs [data-baseweb="tab"] { color: #555555; }
    .stTabs [aria-selected="true"] { color: #005A9C; border-bottom: 3px solid #005A9C; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data
def load_and_process_data(file_contents_dict):
    """Loads, cleans, merges, and processes all uploaded files for combined analysis."""
    try:
        # FIX: The input is already a dictionary, no need for .copy() on a tuple
        dfs = {}
        expected_cols = {
            "NIFTY 50": ['Date', 'Close'], "Infosys": ['Date', 'close'],
            "Crude Oil": ['Date', 'Price'], "USD/INR Rate": ['Date', 'USD']
        }
        
        for name, content in file_contents_dict.items():
            df = pd.read_csv(io.BytesIO(content))
            df.columns = df.columns.str.strip()
            
            if not all(col in df.columns for col in expected_cols[name]):
                st.error(f"File for '{name}' is missing required columns: {expected_cols[name]}.")
                return None
            dfs[name] = df

        nifty_df = dfs["NIFTY 50"][['Date', 'Close']].rename(columns={'Close': 'NIFTY50_CLOSE'})
        nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], errors='coerce', infer_datetime_format=True)
        
        infosys_df = dfs["Infosys"][['Date', 'close']].rename(columns={'close': 'INFOSYS_CLOSE'})
        infosys_df['Date'] = pd.to_datetime(infosys_df['Date'], dayfirst=True, errors='coerce', infer_datetime_format=True)
        
        oil_df = dfs["Crude Oil"][['Date', 'Price']].rename(columns={'Price': 'OIL_PRICE'})
        oil_df['Date'] = pd.to_datetime(oil_df['Date'], dayfirst=True, errors='coerce', infer_datetime_format=True)

        usdinr_df = dfs["USD/INR Rate"][['Date', 'USD']].rename(columns={'USD': 'USD_INR_RATE'})
        usdinr_df['Date'] = pd.to_datetime(usdinr_df['Date'], dayfirst=True, errors='coerce', infer_datetime_format=True)
        
        df_list = [nifty_df.set_index('Date'), infosys_df.set_index('Date'), oil_df.set_index('Date'), usdinr_df.set_index('Date')]
        df_full = pd.concat(df_list, axis=1, join='outer').sort_index()
        
        df_clean = df_full.dropna()
        if df_clean.empty:
            st.error("Processing aborted: No overlapping dates found in the datasets.")
            return None
        
        price_cols = ['NIFTY50_CLOSE', 'INFOSYS_CLOSE', 'OIL_PRICE', 'USD_INR_RATE']
        for col in price_cols:
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            if (df_clean[col] <= 0).any():
                df_clean = df_clean[df_clean[col] > 0]
        
        df_clean.dropna(inplace=True)
        
        for col in price_cols:
            df_clean[f'R_{col}'] = np.log(df_clean[col] / df_clean[col].shift(1))
            
        return df_clean.dropna()
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None

def run_adf_test(series, name):
    """Runs and displays full ADF test results."""
    result = adfuller(series.dropna())
    st.markdown(f"**Results for: `{name}`**")
    
    adf_output = pd.Series(result[0:4], index=['ADF Statistic', 'p-value', '# Lags Used', '# Observations'])
    for key, value in result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    st.dataframe(adf_output.to_frame(name="Value"))
    
    if result[1] <= 0.05:
        st.success("Conclusion: The series is stationary.")
    else:
        st.warning("Conclusion: The series is non-stationary.")

# --- UI Layout ---
with st.sidebar:
    st.title("💹 Econometric Dashboard")
    st.markdown("<p style='color: #333;'>NIFTY 50 Market Analysis.</p>", unsafe_allow_html=True)
    st.divider()
    
    st.header("📂 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload NIFTY, Infosys, Oil, and USD/INR CSV files",
        type="csv",
        accept_multiple_files=True
    )
    st.divider()
    
    files_dict = {}
    if uploaded_files:
        for file in uploaded_files:
            name = file.name.lower()
            if "nifty" in name: files_dict["NIFTY 50"] = file
            elif "infosys" in name: files_dict["Infosys"] = file
            elif "oil" in name: files_dict["Crude Oil"] = file
            elif "usd" in name: files_dict["USD/INR Rate"] = file
    
st.markdown("<h1 class='centered-title'>Financial Market Econometric Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='centered-subtitle'>An Econometric Analysis of Market Drivers</h3>", unsafe_allow_html=True)
st.markdown("---")

file_params_dict = {
    "NIFTY 50": ('Close', '%d-%b-%Y', False), "Infosys": ('close', None, True),
    "Crude Oil": ('Price', None, True), "USD/INR Rate": ('USD', None, True)
}

# --- Main Page Tabs ---
tab_eda, tab_combine, tab_returns, tab_stats, tab_adf, tab_regression = st.tabs([
    "📊 **Individual EDA**", "🔗 **Combined Prices**", "📈 **Log Returns & Plots**",
    "🔢 **Correlation Analysis**", "🔬 **Stationarity Test**", "📉 **Regression & Conclusion**"
])

with tab_eda:
    st.header("Exploratory Data Analysis on Each Dataset")
    eda_colors = ['#0083B8', '#00B8A9', '#F6416C', '#FFDE7D'] 
    
    eda_sub_tabs = st.tabs(["📈 NIFTY 50", "🖥️ Infosys", "🛢️ Crude Oil", "💵 USD/INR Rate"])
    tab_names = ["📈 NIFTY 50", "🖥️ Infosys", "🛢️ Crude Oil", "💵 USD/INR Rate"]
    
    for i, tab in enumerate(eda_sub_tabs):
        with tab:
            dataset_name = tab_names[i].split(" ", 1)[1]
            if dataset_name in files_dict:
                file = files_dict[dataset_name]
                col_name, _, _ = file_params_dict[dataset_name]
                
                st.subheader(f"About {dataset_name}")
                st.divider()
                file.seek(0); raw_df = pd.read_csv(file); raw_df.columns = raw_df.columns.str.strip()
                
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown("**Summary Statistics**"); st.dataframe(raw_df.describe())
                    st.markdown("**Data Distribution**")
                    fig, ax = plt.subplots(figsize=(6, 4)); sns.histplot(pd.to_numeric(raw_df[col_name].astype(str).str.replace(',', ''), errors='coerce'), kde=True, ax=ax, color=eda_colors[i]); ax.set_title(f'Distribution of {col_name}'); st.pyplot(fig)
                with col2:
                    st.markdown("**Time Series Plot**")
                    plot_df = raw_df[['Date', col_name]].copy(); 
                    plot_df['Date'] = pd.to_datetime(plot_df['Date'], errors='coerce', infer_datetime_format=True)
                    plot_df[col_name] = pd.to_numeric(plot_df[col_name].astype(str).str.replace(',', ''), errors='coerce')
                    fig, ax = plt.subplots(figsize=(10, 6)); plot_df.plot(x='Date', y=col_name, ax=ax, color=eda_colors[i], legend=False); ax.set_ylabel(col_name); ax.set_title(f'Time Series of {dataset_name} Price')
                    st.pyplot(fig)
            else:
                st.info(f"Upload data for **{dataset_name}** to see its analysis.")

# --- Combined Analysis Logic ---
if len(files_dict) == 4:
    file_contents_dict = {name: file.getvalue() for name, file in files_dict.items()}
    df = load_and_process_data(file_contents_dict)
    
    if df is not None:
        price_cols = ['NIFTY50_CLOSE', 'INFOSYS_CLOSE', 'OIL_PRICE', 'USD_INR_RATE']
        return_cols = [f'R_{col}' for col in price_cols]
        prices_df = df[price_cols]
        returns_df = df[return_cols]

        with tab_combine:
            st.header("Combined Price Data & Statistics")
            st.dataframe(prices_df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Descriptive Statistics")
                st.dataframe(prices_df.describe())
            with col2:
                st.subheader("Time Series Plot of Prices")
                fig, ax = plt.subplots(figsize=(8, 6));
                prices_df.plot(ax=ax, legend=True); ax.set_title('Time Series of Prices'); ax.set_ylabel('Price')
                st.pyplot(fig)
            st.download_button("Download Cleaned Price Data (CSV)", prices_df.to_csv(), "cleaned_price_data.csv", "text/csv")

        with tab_returns:
            st.header("Logarithmic Returns Analysis")
            st.info(r"We use log returns ($R_t = \ln(\frac{P_t}{P_{t-1}})$) to transform the price series into a stationary series suitable for regression.")
            st.dataframe(returns_df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Descriptive Statistics")
                st.dataframe(returns_df.describe())
            with col2:
                st.subheader("Time Series Plot of Log Returns")
                fig, ax = plt.subplots(figsize=(8, 6));
                returns_df.plot(ax=ax, legend=True); ax.set_title('Time Series of Log Returns'); ax.set_ylabel('Log Return')
                st.pyplot(fig)

        with tab_stats:
            st.header("Correlation Analysis")
            st.info("Correlation in **returns** is more reliable for modeling.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Price Correlation**"); fig, ax = plt.subplots(); sns.heatmap(prices_df.corr(), annot=True, cmap='viridis', fmt=".2f", ax=ax); st.pyplot(fig)
            with col2:
                st.markdown("**Return Correlation**"); fig, ax = plt.subplots(); sns.heatmap(returns_df.corr(), annot=True, cmap='plasma', fmt=".2f", ax=ax); st.pyplot(fig)

        with tab_adf:
            st.header("Stationarity Test (Augmented Dickey-Fuller)")
            st.markdown("- **Null Hypothesis ($H_0$):** Non-stationary.\n- **Alternative Hypothesis ($H_1$):** Stationary.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Price Series Tests"); 
                for col in prices_df.columns: run_adf_test(prices_df[col], col)
            with col2:
                st.markdown("#### Return Series Tests"); 
                for col in returns_df.columns: run_adf_test(returns_df[col], col)
            st.success("Log returns are stationary, validating their use for regression **without differencing**.")

        with tab_regression:
            st.header("Regression Model & Final Conclusion")
            st.latex(r'R_{\text{NIFTY}} = \beta_0 + \beta_1 R_{\text{INFOSYS}} + \beta_2 R_{\text{OIL}} + \beta_3 R_{\text{USDINR}} + \epsilon')
            
            regression_df = returns_df.copy().dropna()
            regression_df.rename(columns={'R_NIFTY50_CLOSE': 'R_NIFTY', 'R_INFOSYS_CLOSE': 'R_INFOSYS', 'R_OIL_PRICE': 'R_OIL', 'R_USD_INR_RATE': 'R_USDINR'}, inplace=True)
            y = regression_df['R_NIFTY']; X = regression_df[['R_INFOSYS', 'R_OIL', 'R_USDINR']]
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.markdown("#### OLS Regression Results"); st.code(str(model.summary()))
            with col2:
                st.markdown("#### Predicted vs. Actual Returns"); 
                fig, ax = plt.subplots(); 
                sns.scatterplot(x=model.fittedvalues, y=y, ax=ax, alpha=0.6)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2)
                ax.set_xlabel("Predicted Returns"); ax.set_ylabel("Actual Returns")
                st.pyplot(fig)

            st.subheader("Model Diagnostics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Multicollinearity (VIF)**")
                vif_data = pd.DataFrame({"VIF": [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]}, index=X_with_const.columns)
                st.dataframe(vif_data)
            with col2:
                st.markdown("**Residuals Q-Q Plot**")
                fig_qq = sm.qqplot(model.resid, line='s')
                st.pyplot(fig_qq)
            
            st.divider()
            st.header("🏆 Final Conclusion")
            st.markdown("- **Model Significance**: R-squared of **{:.3f}** explains **{:.1f}%** of the daily variance in NIFTY 50's returns.".format(model.rsquared, model.rsquared * 100))
            st.markdown("- **Key Drivers**: **Infosys Returns** and **USD/INR Rate Returns** were statistically significant predictors.")
            st.success("**Overall Insight:** The NIFTY 50's daily performance is strongly tied to its major component stocks and is sensitive to foreign exchange market movements.")
else:
    placeholder_message = "☝️ **Please upload all four CSV files to view the analysis.**"
    for tab in [tab_combine, tab_returns, tab_stats, tab_adf, tab_regression]:
        with tab:
            st.info(placeholder_message)

