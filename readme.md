## STOCK MARKET TREND PREDICTOR

Hello Everyone,
This is a professional, production-grade machine learning dashboard for predicting stock market trends. This application uses advanced technical indicators and multiple machine learning algorithms to forecast whether a stock's price will move **UP** or **DOWN** on the next trading day.

## Live Demo:
The project is ready for deployment on **Streamlit Community Cloud**. 
Connect your GitHub repository to [Streamlit Share](https://share.streamlit.io/) and select `app.py` as the entry point.

## Key Features:

##Intelligent Model Selection-
The app features a sophisticated **Auto-Selection Algorithm** that analyzes your dataset size and chooses the most appropriate model:
- **Logistic Regression**: Stability for very small datasets (< 300 samples).
- **SVM (Support Vector Machine)**: High-dimensional boundary mapping for small-medium data (300-800 samples).
- **Random Forest**: Robust ensemble learning for medium datasets (800-1500 samples).
- **Neural Networks (MLP)**: Complex pattern recognition for large datasets (1500-3000 samples).
- **XGBoost**: State-of-the-art performance for ultra-large datasets (> 3000 samples).

###Selection Modes:
- **Automated**: Let the AI decide the best algorithm for your data.
- **Manual**: Take full control and select your preferred algorithm from the sidebar.

###Advanced Technical Analysis:
The predictor engineers features from raw market data, including:
- **Moving Averages (MA5, MA20)**
- **Relative Strength Index (RSI)**
- **Bollinger Bands** (Upper, Lower, Width, and Position)
- **Volatility Metrics**

###Flexible Time Horizons:
Analyze stocks over various periods:
- **Daily / 1 Month / 3 Months / 6 Months**
- **1 Year to 10 Years**
- **Full Historical Data (Max Available)**

##Tech Stack:
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance API)
- **Machine Learning**: Scikit-Learn, XGBoost
- **Visualization**: Plotly Interactive Charts

##Installation:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dharanish-007/Stock_predictor.git
   cd Stock_predictor
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run:

Launch the application using Streamlit:
```bash
streamlit run app.py
```
The app will be available at `http://localhost:8501`.

## Project Structure:
```text
.
├── app.py              # Main Streamlit Dashboard
├── config/
│   └── settings.py     # Configuration & Hyperparameters
├── src/
│   ├── data/           # Data fetching logic
│   ├── features/       # Technical indicators engineering
│   └── models/         # ML model training & selection logic
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Disclaimer:
*This tool is for educational and informational purposes only. Stock market investments carry risks. Always perform your own research or consult with a financial advisor before making investment decisions.*
