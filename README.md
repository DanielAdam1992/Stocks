# 📈 Stock Prediction & Investment Strategy System

## Project Overview

This project is an end-to-end AI system for stock market analysis, combining **time-series forecasting**, **computer vision**, and **LLM** to predict future prices and generate actionable investment recommendations.

The system follows a hybrid, multi-modal approach:

1. **Machine Learning** for Feature Selection – XGBoost identifies top informative features from a rich time-series dataset  
2. **Deep Learning** for Time Series Forecasting – LSTM, GRU, CNN, and hybrid architectures predict future prices  
3. **Computer Vision** for Strategy Classification – A `fastai` classifies price-chart images into BUY / KEEP / SELL  
4. **LLM-Driven User Interface** – A Gemini agent interprets natural-language user queries and triggers the prediction pipeline  

---

## Key Capabilities

### 1️⃣ Robust Data Pipeline

The system aggregates historical stock prices together with macroeconomic indicators to capture broader market dynamics.

**Inputs include:**
- Target assets: Individual stock tickers  
- Macroeconomic indicators:
  - Oil (WTI)
  - Gold
  - S&P 500
  - NASDAQ
  - Real Estate indices
  - Inflation expectations  

### 2️⃣ Advanced Feature Engineering & Selection

- **Feature generation:** approximately 300 auto-generated time-series features per asset, including
  - Lagged values
  - Rolling min / max / mean / std / diff / pct change /

- **Feature selection:** an XGBoost model ranks feature importance and selects top 20 most impactful features

### 3️⃣ Dual-Model Architecture

#### A. Time-Series Regression (Price Forecasting)

- **Models:** LSTM, GRU, CNN, and hybrid/ensemble combinations  
- **Objective:** Predict future stock prices over a configurable horizon  
- **Performance:** best-performing models achieved ~97% predictive performance (R² / accuracy depending on configuration)


#### B. Visual Strategy Classification (Investment Recommendation)

- **Framework:** `fastai` (CNN-based computer vision)  
- **Methodology:**
  1. Convert historical price data into rolling 1-year chart images
  2. Label each image according to future price behavior
  3. Train a classifier to output: **BUY**, **KEEP**, **SELL**

- **Performance:** best vision model achieved ~73% accuracy

### 4️⃣ LLM-Powered Natural Language Interface

- **LLM:** Google Gemini  
- **Functionality:**
  - Parses natural-language user prompts  
  - Extracts: Target ticker (TKL), Prediction horizon (days)
  - Automatically triggers the inference pipeline  

**Example prompt:**
> *“What is the outlook for the GPU company over the next 7 days?”*

---

## 📊 Results Summary

| Model Type        | Architecture        | Task                     | Performance      |
|------------------|---------------------|--------------------------|-------------------|
| Time Series      | LSTM / GRU / CNN    | Price Prediction         | ~97%              |
| Computer Vision  | fastai CNN          | Buy / Keep / Sell        | ~73% accuracy     |

---

## 📂 Repository Structure
```

{REPOSITORY_PATH}/
│
├── data/                                 
│ ├── Project Slides.pptx
│ ├── <stock1>.df.csv                     # Time-series datasets (per stock)
│ └── <stockN>.df.csv
│
├── notebooks/
│ ├── dataprep_for_train.ipynb             # Feature engineering & selection
│ ├── imagesprep_for_train.ipynb           # Chart image generation
│ ├── train_models.ipynb                   # Model selection & training
│ ├── train.ipynb                          # End-to-end training pipeline
│ ├── llm_api.ipynb                        # Gemini-based prompt parsing
│ ├── predict.ipynb                        # Full inference pipeline
│ ├── dataprep_for_inference.ipynb         # Refresh data with latest prices
│ ├── predict_future.ipynb                 # Time-series forecasting
│ └── recommend_investment_strategy.ipynb  # Vision-based recommendation
│
├── src/
│ ├── config.json                          # Global configuration
│ ├── my_project_utils.py                  # Shared helper functions
│ └── init.py
│
├── images/                                # Generated chart images
├── output/                                # Logs and results
│
├── pickles/                               # Serialized models & datasets
│ ├── <stock>.best_model.X_features.keras
│ └── <stock>.df.pkl
│
├── README.md
└── .gitignore
```
---

### `train.ipynb` (Model Training)

<img width="1750" height="1057" alt="image" src="https://github.com/user-attachments/assets/b409c9dc-ac74-466f-8cb6-a910af3f202e" />

---

### `llm_api.ipynb` (Inference & Application)

<img width="1404" height="1055" alt="image" src="https://github.com/user-attachments/assets/9d4c6589-95d8-4d90-9ac5-1bcc2aaf08ae" />

---

## ▶️ How to Run

1. **Clone** the repository
2. Create a `.env` file in the runtime root (e.g. `/content/.env`)
3. In `.env` define the project path: e.g. `PROJECT_PATH=/content/drive/MyDrive/Projects/GitHub/Stocks/`
4. For prediction, add your Gemini API key in `.env`: i.e `GEMINI_API_KEY=apikey`
5. For train, define your targt `TKL` parameter in `src/config.json`
7. Choose one:
8. To **train models:** run `notebooks/train.ipynb`
9. For **inference a stock via LLM API:** run `notebooks/llm_api.ipynb`

---

## 📦 Project Deliverables

- 
- `data/` – Time-series datasets with selected XGB features  
- `pickles/` – Best trained models per stock  
- `images/` – Chart images for vision-based strategy classification

---

⚠️ **Disclaimer:**  this project is for **research and educational purposes only**. It does **not** constitute financial advice.




---


## Investment firm research / paper-trading pilot (2026)

An independent, unmerged development branch adds a **research-only** investment-firm prototype under [algotrading/](algotrading/). Scope: **US equities/ETFs, $50,000 simulated portfolio, no leverage**.

Implemented so far:
- Research-only backtest contract accepting **out-of-sample** daily predicted returns (not yet connected to the original neural models).
- SEC filings analyst for filed-as-of 10-K/10-Q structured facts; time-aware values from the precise filed accession.
- SEC filing HTML reader that extracts **unverified** candidate MD&A and risk-factor excerpts; subsequent 8-K report monitor.
- Research runner with CIO / independent risk-veto decision record, role statuses, and broker execution disabled.
- Synthetic tests and GitHub Actions workflow; read [algotrading/PHASE2.md](algotrading/PHASE2.md) for setup and limitations.

**Current research expansion:** [Quant model tuning and 11-sector/23-industry comparison](algotrading/MODEL_TUNING.md) is now implemented on this development branch. It runs walk-forward baseline/Ridge/HistGradientBoosting/XGBoost forecasts with next-open trade returns; a separate optional LSTM/GRU/CNN architecture re-training comparison is also available. Historical benchmark results live as downloadable GitHub Actions run artifacts, not in `main`. The current experiments do not establish a durable trading edge.

**Not implemented:** verified qualitative filings/news intelligence, externally approved model promotion, multi-asset rebalancing or broker paper trading. The original notebooks and saved model weights have **not** been integrated into a production engine. The goal of 1% net profit/day remains experimental, not a forecast or guarantee.

### Verified quantitative-research milestone (October 8, 2026)

- **All 11 sector ETFs plus SPY tested** in historical, time-aware 252-session out-of-sample simulations; none of the sector models passed the research acceptance screen.
- **23 individual stocks from different industries tested** under the same rules. Visa alone passed the initial one-year screen; longer-history and greater-cost stress tests did not preserve its pass. No strategy has institutional approval.
- **Neural architecture families (LSTM, GRU, CNN) re-trained from scratch** for AAPL, XLF, XLE; 126-session matched baseline showed inconsistent improvements compared with CPU methods and all three lagged allocated buy-and-hold.
- **As-of Quant Research Agent integrated**: reads current historical OHLCV, retrains a forecast using only resolved targets, reports model/forecast and passes it to the CIO research committee. The independent Risk Officer holds all four example research-only signals with zero orders.
- **No actual broker connection, paper-order execution, real-world profit, or stable 1%-per-day strategy has been established.**

Research workflow links: [sectors](https://github.com/DanielAdam1992/Stocks/actions/runs/37778185709) · [industries](https://github.com/DanielAdam1992/Stocks/actions/runs/37778236537) · [robustness](https://github.com/DanielAdam1992/Stocks/actions/runs/37778570584) · [neural models](https://github.com/DanielAdam1992/Stocks/actions/runs/37778801345) · [latest model-only committee](https://github.com/DanielAdam1992/Stocks/actions/runs/37779466747).

See [algotrading/MODEL_TUNING.md](algotrading/MODEL_TUNING.md) for code, reproducibility instructions and limitations.