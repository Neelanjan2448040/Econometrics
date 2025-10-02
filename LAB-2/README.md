# Financial Econometrics Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/)

An interactive web application built with Streamlit for performing and visualizing key financial econometrics analyses, including OLS regression and time-series stationarity tests.

## 🚀 Live Demo

You can access the live, deployed application here:
**[https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/](https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/)**

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [How to Use the App](#-how-to-use-the-app)
- [Local Setup](#-local-setup)

## 📖 Overview

This project is a powerful yet user-friendly tool for conducting essential econometric analyses on financial time-series data. Built entirely in Python with the Streamlit framework, the dashboard allows users to upload their own datasets and interactively explore relationships between variables.

The application streamlines the analytical workflow: a user uploads a CSV file, selects dependent and independent variables through a simple sidebar menu, and instantly receives a comprehensive statistical summary. This includes a test for data stationarity (Augmented Dickey-Fuller test) and a complete Ordinary Least Squares (OLS) regression analysis, empowering users to derive insights without writing a single line of code.

## ✨ Features

- **⬆️ Dynamic CSV Data Upload:** Upload your own time-series dataset directly in the browser.
- **📊 Interactive Variable Selection:** Use intuitive dropdown menus in the sidebar to choose your dependent and independent variables.
- **📈 Stationarity Testing:** Automatically perform the Augmented Dickey-Fuller (ADF) test to check if your time series is stationary.
- **📉 OLS Regression:** Instantly run an Ordinary Least Squares (OLS) regression model on the selected variables.
- **📄 Detailed Summaries:** View a complete OLS regression results table, including coefficients, R-squared, p-values, and other key statistics.
- **🔍 Data Preview:** Inspect the first few rows of your uploaded data to ensure it's loaded correctly.

## 🛠️ Technology Stack

- **Python:** Core programming language.
- **Streamlit:** For building and deploying the interactive web application.
- **Pandas:** For data manipulation and analysis.
- **Statsmodels:** For executing statistical models like OLS and ADF tests.
- **Matplotlib:** For generating plots and visualizations.

## ⚙️ How to Use the App

1.  **Navigate** to the [live application link](https://econometrics-hxq3uxg78rjejooxjshbnu.streamlit.app/).
2.  **Upload a CSV file** using the "Upload your CSV file" button in the sidebar.
3.  **Select one Dependent Variable** from the dropdown menu.
4.  **Select one or more Independent Variables** from the multi-select box.
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
    cd Econometrics/LAB-2/
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
