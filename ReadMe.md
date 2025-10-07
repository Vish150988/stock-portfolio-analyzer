📊 Advanced Stock & Portfolio Analyzer
An interactive web application built with Streamlit for comprehensive stock and portfolio analysis. This tool provides technical and fundamental insights, risk analysis, AI-powered summaries, and advanced visualizations.

✨ Features
This analyzer is packed with features for both single-stock and multi-asset portfolio analysis.

Single Stock Analysis
📈 Interactive Technical Chart: Candlestick chart with toggleable overlays:
Simple Moving Averages (SMA 20, 50, 200)
Bollinger Bands
Fibonacci Retracement Levels
📊 Volume Analysis: Color-coded volume bars and On-Balance Volume (OBV) indicator.
🤖 Technical Summary: At-a-glance signals for SMAs, MACD crossovers, and RSI levels.
🧾 In-Depth Fundamentals: Key metrics (P/E, P/S, P/B, Beta), company summary, historical ratio analysis (ROE, Debt-to-Equity), and full financial statements.
🗓️ Earnings & Ratings: Visualizations for historical EPS surprises and the distribution of analyst ratings.
🔮 AI-Powered Forecasts:
Time-series forecasting with Prophet.
Probabilistic price projections using Monte Carlo simulation.
📰 News & Sentiment: Recent news with sentiment scores and optional AI-generated news briefs (requires API key).
Portfolio Analysis
🧮 Portfolio Optimization: Monte Carlo simulation to find the Efficient Frontier, identifying portfolios with maximum Sharpe ratio and minimum volatility.
📈 Performance & Risk: Analysis of cumulative returns, rolling Sharpe ratio, and average drawdown.
🛡️ Advanced Risk Metrics: Calculation of Value at Risk (VaR) and Conditional Value at Risk (CVaR).
🔗 Correlation Analysis: Heatmap and network graph to visualize correlations between assets.
🧭 Sector Allocation: Pie chart showing the portfolio's allocation across different market sectors.
🚀 Getting Started
Follow these steps to get the application running on your local machine.

Prerequisites
Python 3.8 or higher
Pip package manager
Installation
Clone the repository:

git clone [https://github.com/your-username/stock-analyzer.git](https://github.com/your-username/stock-analyzer.git)
cd stock-analyzer
Create and activate a virtual environment (recommended):

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required packages:

pip install -r requirements.txt
Download NLTK data (first time only): The app will attempt to download the vader_lexicon for sentiment analysis automatically. If this fails, you can run this command in a Python shell:

import nltk
nltk.download('vader_lexicon')
Usage
Run the Streamlit application with the following command:

streamlit run app.py