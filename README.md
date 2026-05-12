# QuantInsight Lab

A Flask-based financial quantitative analysis platform for user-uploaded market datasets. QuantInsight Lab v2.4 supports robust CSV/Excel ingestion, Data Agent v2 column mapping, data quality scoring, strategy backtesting, strategy comparison, ML single-model analysis, ML compare-all-models, and a consolidated Report v2.

---

## 1) Project Overview

### What this project does
QuantInsight Lab provides an end-to-end financial analysis workflow in a browser-based interface. It transforms raw uploaded historical market data into standardized analysis-ready data, then supports strategy research, model evaluation, and consolidated reporting.

### Why it was built
This project was built as a practical, modular, and portfolio-ready quantitative research platform. It combines robust data ingestion, technical strategy evaluation, and ML benchmarking in one lightweight Flask app.

### Main workflow
**Upload → Dashboard → Strategy Lab → Strategy Comparison → ML Analysis → ML Compare All Models → Report v2**

---

## 2) Key Features

- Robust upload support:
  - comma delimiter
  - tab delimiter
  - semicolon delimiter
  - pipe delimiter
  - whitespace fallback
- Column Mapping System (Data Agent v2):
  - alias matching
  - fuzzy matching
  - mapping confidence report
- Numeric cleaning:
  - dollar signs
  - commas
  - spaces
- Data Quality Score and quality checklist
- Market Dashboard (interactive Plotly charts and quality/mapping diagnostics)
- Strategy Lab backtesting:
  - Moving Average Crossover
  - RSI Strategy
  - MACD Strategy
- Strategy Comparison:
  - side-by-side MA/RSI/MACD metrics
  - best by Sharpe Ratio
  - best by Total Return
  - lowest Max Drawdown
- Backtest metrics:
  - Total Return
  - Annualized Return
  - Annualized Volatility
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Number of Trades
- ML Analysis:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - KNN
- ML Compare All Models:
  - cross-model comparison table
  - best by F1-score
  - best by Accuracy
- ML metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - Feature Importance
- Report v2:
  - dataset overview
  - data quality
  - column mapping
  - strategy summary
  - strategy comparison
  - ML summary
  - ML comparison
  - key notes

---

## 3) Tech Stack

- Python
- Flask
- pandas
- numpy
- scikit-learn
- Plotly
- HTML
- CSS
- JavaScript
- openpyxl

---

## 4) Project Structure

```text
quantinsight_lab/
├── app.py
├── config.py
├── requirements.txt
├── uploads/
├── modules/
├── templates/
├── static/
└── README.md
```

---

## 5) Data Format

### Expected standard columns

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

### Supported alternative names (auto-mapped by Data Agent)

- `Date`: `date`, `time`, `datetime`, `timestamp`, `trading date`, `日期`
- `Close`: `close`, `close/last`, `last`, `last price`, `last sale`, `adj close`, `adjusted close`, `closing price`, `price`, `收盘价`
- `Open`: `open`, `open price`, `opening price`, `开盘价`
- `High`: `high`, `highest`, `high price`, `最高价`
- `Low`: `low`, `lowest`, `low price`, `最低价`
- `Volume`: `volume`, `vol`, `trading volume`, `turnover volume`, `成交量`

The loader also supports non-standard delimited files and attempts robust parsing before mapping.

---

## 6) How to Run

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Then open:

[`http://127.0.0.1:5000`](http://127.0.0.1:5000)

---

## 7) How to Use

1. Upload dataset  
2. View dashboard  
3. Run strategy backtest in Strategy Lab  
4. Run strategy comparison  
5. Train a single ML model  
6. Compare all ML models  
7. Open Report v2

---

## 8) Notes and Limitations

- Educational and research use only.
- This app does not fetch real-time market data; users must upload historical data.
- Backtest results do not guarantee future returns.
- ML prediction is based on historical engineered features only.
- ML results on small or artificial datasets may be misleading.
- No user login or database-backed persistence yet.
- Deep Learning Lab is planned future work.
- This is not financial advice.

---

## 9) Future Improvements

- PDF export
- User accounts
- Database-backed history
- More strategies
- More ML/deep learning models
- Real market data API integration
- Deployment with Docker

---

## 10) Screenshots

### Upload Page

<!-- Add screenshot here -->

### Dashboard

<!-- Add screenshot here -->

### Strategy Lab

<!-- Add screenshot here -->

### Strategy Comparison

<!-- Add screenshot here -->

### ML Analysis

<!-- Add screenshot here -->

### ML Comparison

<!-- Add screenshot here -->

### Report v2

<!-- Add screenshot here -->
