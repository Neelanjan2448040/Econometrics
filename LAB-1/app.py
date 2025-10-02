import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Regression Analyzer",
    page_icon="📊",
    layout="wide",
)

# --- Custom CSS for a clean and professional look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #F0F2F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        font-weight: 600;
    }
    .main .block-container {
        padding: 2rem 3rem;
        background-color: #FFFFFF;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #e6e6e6;
    }
    h1, h2, h3 {
        color: #1E3A5F;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# --- Helper Functions ---
def perform_analysis(df_original, processed_df, y_var, x_vars, analysis_type):
    """
    Performs the full analysis and displays results in tabs.
    """
    st.divider()
    # --- Create Tabs for Results ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Dataset Info", 
        "🔗 Covariance Analysis", 
        "🔗 Correlation Analysis", 
        "📈 Regression Analysis"
    ])

    analysis_df = processed_df[[y_var] + x_vars].copy()

    # --- Tab 1: Dataset Info ---
    with tab1:
        st.subheader("Exploratory Data Analysis")
        
        if analysis_type == 'multiple' and any(c.endswith('_R') for c in analysis_df.columns):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Original Uploaded Dataset**")
                st.dataframe(df_original)
            with col2:
                st.markdown("**Generated Returns Dataset (Selected Variables)**")
                st.dataframe(analysis_df.dropna())
            
            st.divider()
            st.markdown("**Descriptive Statistics (Selected Variables)**")
            st.dataframe(analysis_df.describe())
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Complete Uploaded Dataset**")
                st.dataframe(df_original)
            with col2:
                st.markdown("**Descriptive Statistics (Selected Variables)**")
                st.dataframe(analysis_df.describe())
        
        st.subheader("Variable Distributions (Histograms)")
        num_plots = len(analysis_df.columns)
        if num_plots > 0:
            color_palette = sns.color_palette("Set2", num_plots)
            num_cols = min(num_plots, 3) 
            cols = st.columns(num_cols)
            for i, column in enumerate(analysis_df.columns):
                with cols[i % num_cols]:
                    fig, ax = plt.subplots()
                    sns.histplot(analysis_df[column].dropna(), kde=True, ax=ax, color=color_palette[i])
                    ax.set_title(f'Distribution of {column}')
                    st.pyplot(fig)

    # --- Tab 2: Covariance ---
    with tab2:
        st.subheader("Covariance Analysis")
        st.markdown("Covariance measures the joint variability of two random variables. A positive covariance indicates that the variables tend to move in the same direction, while a negative covariance indicates they move in opposite directions.")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("**Covariance Matrix**")
            st.dataframe(analysis_df.cov())
        with col2:
            st.markdown("**Covariance Heatmap**")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(analysis_df.cov(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
            st.pyplot(fig)
            
    # --- Tab 3: Correlation ---
    with tab3:
        st.subheader("Correlation Analysis")
        st.markdown("Correlation is a standardized version of covariance that measures the strength and direction of a *linear* relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive).")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("**Correlation Matrix**")
            st.dataframe(analysis_df.corr())
        with col2:
            st.markdown("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(analysis_df.corr(), annot=True, cmap='plasma', fmt=".2f", ax=ax)
            st.pyplot(fig)

    # --- Tab 4: Regression ---
    with tab4:
        st.subheader(f"{analysis_type.replace('_', ' ').title()} Analysis")
        
        st.markdown("**General Regression Equation:**")
        if analysis_type == "simple":
            st.latex(fr'{y_var} = \beta_0 + \beta_1 ({x_vars[0]}) + \epsilon')
        else:
            betas = " + ".join([fr"\beta_{i+1} ({var})" for i, var in enumerate(x_vars)])
            st.latex(fr'{y_var} = \beta_0 + {betas} + \epsilon')
        
        st.markdown("---")
        
        y = processed_df[y_var]
        X = processed_df[x_vars]
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const, missing='drop').fit()
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("**OLS Regression Summary**")
            st.text(model.summary())
        with col2:
            st.markdown("**Regression Plot**")
            if analysis_type == "simple":
                fig, ax = plt.subplots()
                sns.regplot(x=x_vars[0], y=y_var, data=processed_df, ax=ax, line_kws={"color":"#E41A1C"}, scatter_kws={"color":"#377EB8"})
                ax.set_title(f"Scatter Plot of {y_var} vs. {x_vars[0]}")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                sns.scatterplot(x=model.fittedvalues, y=model.model.endog, ax=ax, color="#4DAF4A")
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Actual Values")
                ax.set_title("Actual vs. Predicted Values")
                ax.plot([model.model.endog.min(), model.model.endog.max()], [model.model.endog.min(), model.model.endog.max()], '#E41A1C', lw=2, linestyle='--')
                st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("**Fitted Regression Equation:**")
        intercept = model.params.get('const', 0)
        equation_parts = [f"{intercept:.4f}"]
        for var in x_vars:
            coef = model.params.get(var, 0)
            sign = "+" if coef >= 0 else "-"
            equation_parts.append(f"{sign} {abs(coef):.4f} ({var.replace('_', ' ')})")
        
        final_equation = f"{y_var.replace('_', ' ')} = {' '.join(equation_parts)}"
        st.success(final_equation)
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Multicollinearity Check (VIF)**")
            X_vif = X_with_const.dropna()
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_vif.columns
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            st.dataframe(vif_data)
            st.info("For multiple regression, a VIF > 5 is a common red flag for high multicollinearity.")
        with col4:
            st.markdown("**Residuals Q-Q Plot**")
            fig_qq = sm.qqplot(model.resid, line='s')
            st.pyplot(fig_qq)
            st.info("If the residuals follow the red line, it suggests they are normally distributed, a key assumption of OLS.")

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Analysis Configuration")
    analysis_choice = st.radio("Select Analysis Type", ["Simple Linear Regression", "Multiple Linear Regression"], key="analysis_type")
    
    uploaded_file = st.file_uploader(
        "Upload your Data File",
        type=['csv', 'xlsx']
    )

# --- MAIN PAGE ---
st.title("📊 Advanced Regression Analysis Dashboard")

if uploaded_file is None:
    st.markdown("""
    ### Welcome to the Interactive Regression Analyzer!
    This tool allows you to perform sophisticated statistical analysis on your own data with just a few clicks.
    
    **Get started in 3 simple steps:**
    1.  **Select Analysis Type:** Choose between 'Simple' or 'Multiple' Linear Regression in the sidebar.
    2.  **Upload Your Data:** Upload a CSV or Excel file containing your dataset.
    3.  **Configure & Run:** Select your variables and click 'Run Analysis' to see the results.
    
    Your results will be presented across organized tabs, providing deep insights into your data's characteristics and relationships.
    """)
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_original = pd.read_csv(uploaded_file)
        else:
            df_original = pd.read_excel(uploaded_file)
        
        # Clean column names by stripping whitespace
        df_original.columns = df_original.columns.str.strip()

        # --- Handle Different Analysis Flows ---
        if st.session_state.analysis_type == "Simple Linear Regression":
            st.header("Setup for: Simple Linear Regression")
            st.markdown("Select your variables below and click 'Run Analysis' to generate the report.")
            
            df_cleaned = df_original.copy()
            for col in df_cleaned.columns[1:]: # Assume first col is date/identifier
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

            y_var = st.selectbox("Select Dependent Variable (Y)", df_cleaned.columns[1:])
            x_var = st.selectbox("Select Independent Variable (X)", [col for col in df_cleaned.columns[1:] if col != y_var])
            x_vars = [x_var] if x_var else []
            
            df_for_analysis = df_cleaned
        
        else: # Multiple Linear Regression
            st.header("Setup for: Multiple Linear Regression")
            
            analysis_level = st.sidebar.radio("Choose Data to Analyze", ["Prices", "Returns"], key="data_type")

            if analysis_level == "Returns":
                with st.spinner("Calculating log returns..."):
                    df_with_returns = df_original.copy()
                    
                    first_col = df_with_returns.columns[0]
                    cols_to_process = [col for col in df_with_returns.columns if col != first_col and not col.endswith('_R')]

                    for col in cols_to_process:
                        # FIX: Robustly convert column to numeric BEFORE calculating return
                        df_with_returns[col] = pd.to_numeric(df_with_returns[col], errors='coerce')
                        # FIX: Always calculate/overwrite the return column
                        if df_with_returns[col].notna().any():
                            df_with_returns[f"{col}_R"] = np.log(df_with_returns[col] / df_with_returns[col].shift(1))
                
                st.sidebar.success("Log Return columns (_R) calculated.")
                
                df_for_analysis = df_with_returns
                valid_cols = [col for col in df_for_analysis.columns if col.endswith('_R') and df_for_analysis[col].notna().any()]
                
                if not valid_cols:
                    st.error("Could not calculate any valid return columns. Check dataset for missing values.")
                    st.stop()
                
                default_y_index = valid_cols.index('Nifty 50_R') if 'Nifty 50_R' in valid_cols else 0
                y_var = st.selectbox("Select Dependent Variable (Y)", valid_cols, index=default_y_index, key="y_returns")
                
                available_x_cols = [col for col in valid_cols if col != y_var]
                x_vars = st.multiselect("Select Independent Variables (X)", available_x_cols, default=available_x_cols, key="x_returns")

            else: # Prices Analysis
                df_cleaned = df_original.copy()
                for col in df_cleaned.columns[1:]:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                
                df_for_analysis = df_cleaned
                valid_cols = [col for col in df_for_analysis.columns[1:] if df_for_analysis[col].notna().any()]

                default_y_index = valid_cols.index('Nifty 50') if 'Nifty 50' in valid_cols else 0
                y_var = st.selectbox("Select Dependent Variable (Y)", valid_cols, index=default_y_index, key="y_prices")
                
                available_x_cols = [col for col in valid_cols if col != y_var]
                x_vars = st.multiselect("Select Independent Variables (X)", available_x_cols, default=available_x_cols, key="x_prices")

        # --- Run Button and Analysis Execution ---
        if st.button("Run Analysis", key="run_button"):
            st.session_state.run_analysis = True

        if st.session_state.run_analysis:
            if y_var and x_vars:
                analysis_type_key = "simple" if st.session_state.analysis_type == "Simple Linear Regression" else "multiple"
                perform_analysis(df_original, df_for_analysis, y_var, x_vars, analysis_type_key)
            else:
                st.warning("Warning: Please select all required variables before running the analysis.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

