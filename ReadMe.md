# 📊 Advanced Stock & Portfolio Analyzer

An interactive Streamlit application for in-depth equity and multi-asset portfolio research. The app combines technical analysis, fundamentals, portfolio risk tooling, and optional AI-powered narratives to provide a single-pane view for retail investors or analysts.

## ✨ Feature Highlights

### Single Ticker Deep Dive
- **Interactive candlestick chart** with overlays for SMA (20/50/200), Bollinger Bands, Fibonacci retracements, and intraday volume.
- **Momentum indicators** including RSI, MACD (line, signal, histogram), and On-Balance Volume.
- **Fundamental snapshots** pulling valuation ratios, financial statements, calendar events, and analyst recommendations from Yahoo Finance.
- **Earnings and news** timelines with sentiment scoring via NLTK's VADER lexicon.
- **Scenario exploration** with Prophet-based time-series forecasting and Monte Carlo price simulations (optional dependency).

### Portfolio Intelligence
- **Efficient frontier search** using Monte Carlo resampling to surface maximum Sharpe and minimum volatility mixes.
- **Performance diagnostics** such as cumulative returns, rolling Sharpe ratio, and drawdown analytics.
- **Advanced risk metrics** including Value at Risk (VaR) and Conditional VaR.
- **Correlation explorer** featuring heatmaps and an optional NetworkX-powered graph layout.
- **Sector allocation** visualization to highlight diversification gaps.

### AI & Productivity Extras
- Optional OpenAI-compatible summaries for news or technical context (DeepSeek or OpenAI models).
- Export buttons for most tables to simplify downstream reporting.
- Graceful degradation: features that rely on optional libraries are hidden when the dependency is unavailable.

## 🗂️ Project Structure

```
App.py          # Streamlit interface and business logic
Assets/         # Image assets used within the UI
requirements.txt
ReadMe.md
```

## ✅ Requirements

- Python 3.8+
- pip (the Python package manager)
- Internet access for Yahoo Finance API calls and optional AI/news features

### Optional Python Dependencies

Some capabilities are activated only when the corresponding library is installed:

| Capability | Library | Notes |
|------------|---------|-------|
| Time-series forecasting | `prophet` | Forecast card appears only when installed |
| News sentiment | `nltk` | Downloads the `vader_lexicon` on first use |
| Correlation network graph | `networkx` | Enables force-directed network view |
| AI-generated summaries | `openai` | Works with OpenAI or DeepSeek-compatible endpoints |

Install them individually or simply rely on `requirements.txt`, which already includes the core dependencies.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/stock-portfolio-analyzer.git
   cd stock-portfolio-analyzer
   ```
2. **(Recommended) Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\\venv\\Scripts\\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK data (if prompted)**
   The app attempts to download the VADER lexicon automatically. If it fails, run the following inside a Python shell:
   ```python
   import nltk
   nltk.download("vader_lexicon")
   ```

## 🔐 Configuration

Set environment variables if you want to enable optional AI features:

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` | Authentication for LLM summaries |
| `OPENAI_BASE_URL` | Override the default DeepSeek endpoint when using another provider |

If no key is provided, the application silently skips AI summarization and continues to function normally.

## 🚀 Usage

Launch the Streamlit server from the project root:

```bash
streamlit run App.py
```

Then open the printed local URL (usually http://localhost:8501) in your browser.

### Typical Workflow

1. Enter a stock ticker in the sidebar to populate the single-stock analysis dashboard.
2. (Optional) Add multiple tickers separated by commas to unlock portfolio analytics.
3. Explore the tabs to review technicals, fundamentals, risk metrics, earnings, and news.
4. Export any tables you need via the download buttons.

## 📡 Data Sources & Caching

- Market and fundamentals data are pulled live from [Yahoo Finance](https://finance.yahoo.com) via `yfinance`.
- Streamlit caching is used for price history and company blobs to reduce repeated API calls during a session.
- Monte Carlo simulations and forecasts are recomputed when the underlying inputs change.

## 🧰 Troubleshooting

- **Missing optional features** – Verify the corresponding library is installed (see table above).
- **Sentiment analyzer errors** – Ensure the VADER lexicon is present; re-run the NLTK download snippet if necessary.
- **API rate limits or connectivity issues** – Yahoo Finance occasionally throttles requests; retry after a short wait.

## 📄 License

This project is released under the MIT License. See `LICENSE` (if provided) or adapt the license to your distribution needs.

