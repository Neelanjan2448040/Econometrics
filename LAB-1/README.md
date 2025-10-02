# Advanced Regression Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/)

An interactive web application built with Streamlit for powerful financial analysis providing a full suite of diagnostics for Simple and Multiple OLS Regression, including VIF scores for multicollinearity and Q-Q plots for residual analysis

## 🚀 Live Demo

You can access the live, deployed application here:
**[https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/](https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/)**

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [How to Use the App](#-how-to-use-the-app)
- [Local Setup](#-local-setup)
- [Repository Structure](#-repository-structure)

## 📖 Overview

This project is a powerful and user-friendly tool for conducting comprehensive econometric analysis. Built entirely in Python with the Streamlit framework, the dashboard allows users to upload their own datasets (CSV or Excel) and interactively explore relationships between variables.

Its standout feature is the dual-analysis capability for multiple regression, allowing users to seamlessly switch between analyzing raw price levels and automatically calculated logarithmic returns. The application streamlines the workflow by presenting a full suite of analyses—from exploratory data visualizations to detailed regression diagnostics—in a clean, tabbed interface, empowering users to derive deep insights without writing any code.

## ✨ Features

- **✌️ Dual Regression Modes:** Choose between **Simple** and **Multiple** Linear Regression.
- **🔀 Prices vs. Returns Analysis:** Seamlessly switch between analyzing raw price levels or auto-calculated log returns for multiple regression.
- **⬆️ Flexible Data Upload:** Supports both CSV and Excel file formats.
- **🔢 Automatic Equation Generation::** Displays both the general mathematical formula and the final fitted regression equation with calculated coefficients.
- **📉 OLS Regression:** Instantly run an Ordinary Least Squares (OLS) regression model on the selected variables.
- **📄 Detailed Summaries:** View a complete OLS regression results table, including coefficients, R-squared, p-values, and other key statistics.
- **🔍 Data Preview:** Inspect the first few rows of your uploaded data to ensure it's loaded correctly.

## 🛠️ Technology Stack

- **Python:** Core programming language.
- **Streamlit:** For building and deploying the interactive web application.
- **Pandas:** For data manipulation and analysis.
- **Numpy:** For high-performance numerical operations, including log return calculations.
- **Statsmodels:** For executing statistical models like OLS and ADF tests.
- **Matplotlib & Seaborn:** For generating plots and visualizations.

## ⚙️ How to Use the App

1.  **Navigate** to the [live application link](https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/).
2.  **Upload a CSV file** using the "Upload your CSV file" button in the sidebar.
3.  **Select one Dependent Variable** from the dropdown menu.
4.  **Select one or more Independent Variables** from the multi-select box.
5.  **Select prices or returns in case of MLR** from the radio button.
5.  **View the results** which appear automatically in the main panel.

## 🚀 Local Setup

To run this application on your local machine, please follow the steps below.

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Git for cloning the repository

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Neelanjan2448040/Econometrics.git](https://github.com/Neelanjan2448040/Econometrics.git)
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Econometrics/LAB-1/
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```
    Your web browser should automatically open to the application's local address.
