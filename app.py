"""
Professional Streamlit Dashboard
Production-grade web interface for stock prediction
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data.market_data import MarketDataFetcher
from src.features.technical_indicators import TechnicalIndicators
from src.models.trainer import ModelTrainer
from src.models.model_selector import ModelSelector
from config.settings import Config

# Page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-up {
        background-color: #00ff00;
        color: #000;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-down {
        background-color: #ff0000;
        color: #fff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header"><h1 style="color: white;">📈 Stock Market Trend Predictor</h1></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        stock_symbol = st.text_input("Stock Symbol", value="RELIANCE.NS").upper()
        period_options = {
            "1 Day": "1d",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "10 Years": "10y",
            "Max Available": "max"
        }
        selected_period_label = st.selectbox("Data Period", list(period_options.keys()), index=6) # Default to 5 Years
        period = period_options[selected_period_label]
        st.markdown("---")
        selection_mode = st.radio("Model Selection Mode", ["Automated", "Manual"], index=0)
        
        model_type = None
        if selection_mode == "Manual":
            model_type = st.selectbox(
                "Select Algorithm",
                ["random_forest", "xgboost", "logistic", "svm", "neural_network"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        st.markdown("---")
        st.markdown("### 📊 Features Used")
        st.markdown("- Moving Averages (MA5, MA20)")
        st.markdown("- RSI (Relative Strength Index)")
        st.markdown("- Bollinger Bands")
        st.markdown("- Volatility")
        
        predict_button = st.button("🚀 Predict Tomorrow's Trend", type="primary", use_container_width=True)
    
    # Main content
    if predict_button:
        with st.spinner("Fetching market data and training model..."):
            try:
                # Fetch data
                fetcher = MarketDataFetcher(stock_symbol)
                raw_data = fetcher.fetch(period=period)
                
                if len(raw_data) < 20:
                    st.error(f"Insufficient data for {stock_symbol} in the selected period ({selected_period_label}). At least 20 days of history are required for technical indicators.")
                    st.stop()
                
                # Feature engineering
                df = TechnicalIndicators.engineer_features(raw_data)
                
                # Prepare features
                feature_cols = ['MA5', 'MA20', 'RSI', 'Volatility', 'BB_Width', 'BB_Position']
                available_features = [col for col in feature_cols if col in df.columns]
                
                # Model selection
                if selection_mode == "Automated":
                    model_type, selection_reason = ModelSelector.select_model(df)
                    st.info(f"🤖 **Automated Model Selection:** {selection_reason}")
                else:
                    st.success(f"✅ **Manual Selection:** Using {model_type.replace('_', ' ').title()}")
                
                # Train model
                trainer = ModelTrainer(model_type=model_type)
                X, y = trainer.prepare_data(df, available_features)
                X_train, X_test, y_train, y_test = trainer.time_series_split(X, y)
                trainer.train(X_train, y_train)
                
                # Evaluate
                metrics = trainer.evaluate(X_test, y_test)
                
                # Predict tomorrow
                latest_features = X.iloc[-1:].values
                prediction = trainer.model.predict(latest_features)[0]
                prediction_proba = trainer.model.predict_proba(latest_features)[0]
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"₹{raw_data['Close'].iloc[-1]:.2f}",
                        delta=f"{raw_data['Close'].pct_change().iloc[-1]*100:.2f}%"
                    )
                
                with col2:
                    if prediction == 1:
                        st.markdown('<div class="prediction-up">📈 UP Tomorrow</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-down">📉 DOWN Tomorrow</div>', unsafe_allow_html=True)
                
                with col3:
                    st.metric(
                        label="Confidence",
                        value=f"{max(prediction_proba)*100:.1f}%",
                        delta=f"Algo: {model_type.replace('_', ' ').title()}"
                    )
                
                # Performance metrics
                st.subheader("📊 Model Performance")
                perf_cols = st.columns(4)
                metrics_display = [
                    ("Accuracy", f"{metrics['accuracy']:.1%}"),
                    ("Precision", f"{metrics['precision']:.1%}"),
                    ("Recall", f"{metrics['recall']:.1%}"),
                    ("F1 Score", f"{metrics['f1_score']:.1%}")
                ]
                for col, (label, value) in zip(perf_cols, metrics_display):
                    col.metric(label, value)
                
                # Price chart with Plotly
                st.subheader("📈 Price Chart with Technical Indicators")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=("Price & Bollinger Bands", "RSI", "Volume")
                )
                
                # Candlestick chart (simplified with line)
                fig.add_trace(go.Scatter(
                    x=raw_data.index[-60:],
                    y=raw_data['Close'][-60:],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ), row=1, col=1)
                
                # Bollinger Bands
                if 'BB_Upper' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index[-60:],
                        y=df['BB_Upper'][-60:],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index[-60:],
                        y=df['BB_Lower'][-60:],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash')
                    ), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=df.index[-60:],
                    y=df['RSI'][-60:],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Volume
                fig.add_trace(go.Bar(
                    x=raw_data.index[-60:],
                    y=raw_data['Volume'][-60:],
                    name='Volume',
                    marker_color='lightblue'
                ), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True, title_text=f"{stock_symbol} Analysis")
                fig.update_xaxes(title_text="Date", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check the stock symbol and try again.")
    
    else:
        # Welcome message
        st.info("👈 Enter a stock symbol and click 'Predict Tomorrow's Trend' to get started")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🎯 How It Works
            1. Enter a stock symbol (e.g., RELIANCE.NS, TCS.NS, AAPL)
            2. Select data period
            3. Click predict; our **Auto-Selection Algorithm** will choose the best model based on dataset size
            4. View performance metrics and charts
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 Intelligent Selection
            The system automatically selects between:
            - **Logistic Regression**: Optimized for very small datasets (< 300)
            - **SVM**: Optimized for small-medium datasets (300-800)
            - **Random Forest**: Optimized for medium datasets (800-1500)
            - **Neural Network**: Optimized for large datasets (1500-3000)
            - **XGBoost**: Optimized for ultra-large datasets (> 3000)
            """)

if __name__ == "__main__":
    main()