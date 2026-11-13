# 📈 AI-Powered Stock Market Forecasting & Risk Intelligence System

An AI and Machine Learning based stock market forecasting and risk analysis project focused on three major Indian companies: TCS, Infosys, and Reliance.

## 🎯 Project Overview

This project uses historical stock market data and Machine Learning, Deep Learning, and statistical models to study stock price movements and estimate market risk.

## 🏢 Stocks Analyzed

- TCS (TCS.NS)
- Infosys (INFY.NS)
- Reliance Industries (RELIANCE.NS)

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Yahoo Finance (yfinance)
- Scikit-learn
- Random Forest
- XGBoost
- TensorFlow / Keras (LSTM)
- ARCH / GARCH
- Streamlit
- Plotly
- Matplotlib

## 🤖 Models Implemented

1. Random Forest – Stock price direction classification.
2. XGBoost – Stock price forecasting.
3. LSTM – Time-series stock price forecasting.
4. GARCH – Volatility and risk estimation.

## 📊 Model Performance

### Random Forest Classification Accuracy

| Company | Accuracy |
|---|---:|
| TCS | 44.30% |
| Infosys | 51.90% |
| Reliance | 48.52% |

### XGBoost and LSTM – Test Results

| Company | XGBoost MAPE | LSTM MAPE |
|---|---:|---:|
| TCS | 15.61% | 16.66% |
| Infosys | 6.33% | 4.80% |
| Reliance | 3.22% | 3.25% |

### GARCH – Estimated 5-Day Volatility

| Company | Volatility |
|---|---:|
| TCS | 4.074% |
| Infosys | 3.668% |
| Reliance | 4.924% |

*These are historical test results and statistical estimates, not guaranteed future performance.*

## 📊 Interactive Dashboard

The Streamlit dashboard provides:

- Stock selection
- Model performance metrics
- XGBoost vs LSTM error comparison
- GARCH risk analysis
- Model comparison table

## 📁 Project Structure

```text
Stock_Market_Intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── outputs/
│   └── model_comparison.csv
│
├── src/
│   └── app.py
│
├── 01_data_collection.ipynb
├── requirements.txt
└── .gitignore
```

## 🚀 Run the Dashboard

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run src/app.py
```

## 📌 Data Source

Historical daily stock market data is collected using Yahoo Finance (`yfinance`).

The project uses historical daily data, not live tick-by-tick market data.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not financial or investment advice. Stock market predictions involve uncertainty and risk.

## 👩‍💻 Author

Developed as a Machine Learning and Data Science portfolio project.
