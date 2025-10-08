# app.py
# =============================================================================
# Advanced Stock & Portfolio Analyzer — v7.2 (Chart Fix)
# =============================================================================
# What’s fixed:
# - BUG FIX: Chart controls now work correctly. The plot order was changed to
#   ensure SMA lines are drawn on top of the Bollinger Bands fill area.
# - Keeps all features from prior versions.
# =============================================================================

import os
import math
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ---------- Optional libs (graceful fallback) ----------
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

# OpenAI-compatible client (DeepSeek or OpenAI). Only used if a key is provided.
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Advanced Stock & Portfolio Analyzer — v7.2", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# Helpers (bulletproof numeric extraction, formatting, etc.)
# =============================================================================
def get_scalar(val) -> float:
    """Safely extract a float scalar from Series/array/DataFrame/anything."""
    try:
        if isinstance(val, pd.Series):
            return float(val.iloc[-1])
        if isinstance(val, pd.DataFrame):
            return float(val.iloc[-1].squeeze())
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[-1])
        return float(val)
    except Exception:
        return np.nan

def last_val(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    try:
        return get_scalar(df[col].iloc[-1])
    except Exception:
        return np.nan

def safe_metric_fmt(val, kind="num"):
    """Format scalars for metric display while tolerating pandas objects."""
    val = get_scalar(val)
    if not isinstance(val, (int, float, np.floating)) or np.isnan(val):
        return "N/A"
    if kind == "money_b":
        return f"${val/1e9:,.2f}B"
    if kind == "pct":
        return f"{val*100:,.2f}%"
    return f"{val:,.2f}"

def export_csv_button(df, label, filename):
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.download_button(label, df.to_csv().encode("utf-8"), filename, "text/csv")

def try_init_vader():
    if not _HAS_NLTK:
        return None
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        try:
            nltk.download("vader_lexicon")
        except Exception:
            return None
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

# ---------- Optional LLM enrichment ----------
def get_llm_client(api_key: str | None, base_url: str | None = None):
    """Return an OpenAI-compatible client or None if unavailable."""
    if not _HAS_OPENAI:
        return None
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    # DeepSeek is OpenAI-compatible; base_url is required for DeepSeek
    base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
    try:
        return OpenAI(api_key=key, base_url=base)
    except Exception:
        return None

def llm_summarize(client, prompt: str, model: str = "deepseek-chat", max_tokens: int = 300) -> str | None:
    """LLM summary with safe fallbacks; returns None if anything fails."""
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a precise financial assistant. Be concise and factual."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
        return content
    except Exception:
        return None

# =============================================================================
# Data fetch & indicators
# =============================================================================
@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df.dropna()
    except Exception:
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace({0: np.nan})
    d["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Histogram"] = d["MACD"] - d["MACD_Signal"]

    # Bollinger
    d["SMA20"] = c.rolling(20, min_periods=20).mean()
    d["StdDev"] = c.rolling(20, min_periods=20).std()
    d["UpperBB"] = d["SMA20"] + 2*d["StdDev"]
    d["LowerBB"] = d["SMA20"] - 2*d["StdDev"]

    # SMAs
    d["SMA50"] = c.rolling(50, min_periods=50).mean()
    d["SMA200"] = c.rolling(200, min_periods=200).mean()
    return d

@st.cache_data(show_spinner=False)
def robust_company_blob(ticker_str: str):
    """
    Return dict with info, news, financials, balance_sheet, etc.
    Now includes calendar, earnings_dates, and recommendations.
    Always returns non-None structures (possibly empty).
    """
    blob = {
        "info": {}, "news": [], "financials": pd.DataFrame(), "balance_sheet": pd.DataFrame(),
        "insider": pd.DataFrame(), "holders": pd.DataFrame(), "calendar": pd.DataFrame(),
        "earnings_dates": pd.DataFrame(), "recommendations": pd.DataFrame()
    }
    try:
        t = yf.Ticker(ticker_str)
        # --- info fallbacks
        for _getter in ("get_info", "info"):
            try:
                val = getattr(t, _getter) if _getter == "info" else t.get_info()
                if callable(val): val = val()
                if isinstance(val, dict) and val: blob["info"].update(val)
            except Exception: pass
        try:
            fi = t.fast_info
            if fi:
                blob["info"].setdefault("marketCap", getattr(fi, "market_cap", None) or fi.get("market_cap") if isinstance(fi, dict) else None)
                blob["info"].setdefault("lastPrice", getattr(fi, "last_price", None) or fi.get("last_price") if isinstance(fi, dict) else None)
        except Exception: pass

        # --- news fallbacks
        try:
            if hasattr(t, "get_news"):
                n = t.get_news() or []
                if n: blob["news"] = n
        except Exception: pass
        if not blob["news"]:
            try:
                n = t.news or []
                if n: blob["news"] = n
            except Exception: pass

        # --- Financials, Balance Sheet, Insiders, Holders ---
        data_attrs = {
            "financials": ("financials", "income_stmt", "get_income_stmt"),
            "balance_sheet": ("balance_sheet", "get_balance_sheet"),
            "insider": ("insider_transactions",),
            "holders": ("institutional_holders",),
            "calendar": ("calendar",),
            "earnings_dates": ("earnings_dates",),
            "recommendations": ("recommendations",)
        }
        for key, attrs in data_attrs.items():
            for attr in attrs:
                try:
                    v = getattr(t, attr)
                    v = v() if callable(v) else v
                    if isinstance(v, pd.DataFrame) and not v.empty:
                        blob[key] = v
                        break
                except Exception: pass
    except Exception: pass
    return blob

def summarize_signals(df: pd.DataFrame):
    if df is None or df.empty:
        return ["No data available."]
    notes = []
    close = last_val(df, "Close")

    for w in (20, 50, 200):
        sma = last_val(df, f"SMA{w}")
        if not np.isnan(close) and not np.isnan(sma):
            notes.append(("📈" if close > sma else "📉") + f" Price {'above' if close > sma else 'below'} SMA{w}")

    macd, macd_sig = last_val(df, "MACD"), last_val(df, "MACD_Signal")
    if not np.isnan(macd) and not np.isnan(macd_sig):
        if macd > macd_sig:
            notes.append("🚀 MACD bullish crossover")
        elif macd < macd_sig:
            notes.append("🧯 MACD bearish crossover")

    rsi = last_val(df, "RSI")
    if not np.isnan(rsi):
        if rsi >= 70:
            notes.append("⚠️ RSI overbought (≥70)")
        elif rsi <= 30:
            notes.append("💰 RSI oversold (≤30)")
        else:
            notes.append(f"ℹ️ RSI neutral ({rsi:.1f})")
    return notes

def forecast_with_prophet(df: pd.DataFrame, periods=30) -> pd.DataFrame | None:
    if not _HAS_PROPHET or df is None or df.empty or len(df) < 40:
        return None
    try:
        tmp = df.reset_index()
        date_col = "Date" if "Date" in tmp.columns else tmp.columns[0]
        ds = pd.to_datetime(tmp[date_col])
        y = tmp["Close"].astype(float)
        fdf = pd.DataFrame({"ds": ds, "y": y}).dropna()
        if fdf.empty or fdf["y"].nunique() < 5:
            return None
        model = Prophet(daily_seasonality=True)
        model.fit(fdf)
        future = model.make_future_dataframe(periods=periods)
        fc = model.predict(future)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception:
        return None

def compute_drawdown_curve(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    peak = series.cummax()
    return (series / peak) - 1.0

def _normalize_price_series(close_like) -> pd.Series:
    """Return a 1D float Series from many possible close-price inputs."""
    if close_like is None:
        return pd.Series(dtype=float)

    if isinstance(close_like, pd.DataFrame):
        if close_like.empty:
            return pd.Series(dtype=float)
        numeric = close_like.select_dtypes(include=[np.number])
        if numeric.empty:
            numeric = close_like.apply(pd.to_numeric, errors="coerce")
        series = numeric.squeeze()
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    elif isinstance(close_like, pd.Series):
        series = close_like
    else:
        try:
            series = pd.Series(close_like)
        except Exception:
            return pd.Series(dtype=float)

    series = pd.to_numeric(series, errors="coerce")
    return series.dropna()


def _to_float(val) -> float:
    """Best-effort conversion of statistics to plain floats."""
    if val is None:
        return np.nan

    if isinstance(val, pd.DataFrame):
        if val.empty:
            return np.nan
        val = val.select_dtypes(include=[np.number])
        if val.empty:
            return np.nan
        arr = val.to_numpy().ravel()
        return float(arr[0]) if arr.size else np.nan

    if isinstance(val, pd.Series):
        if val.empty:
            return np.nan
        return float(val.dropna().iloc[0]) if val.dropna().size else np.nan

    try:
        return float(val)
    except Exception:
        try:
            arr = np.asarray(val, dtype=float).ravel()
            return float(arr[0]) if arr.size else np.nan
        except Exception:
            return np.nan


def compute_risk_metrics(close: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """Return key risk metrics for a close-price series."""
    close_series = _normalize_price_series(close)
    if close_series.empty:
        return {}

    returns = close_series.pct_change().dropna()
    if returns.empty:
        return {}

    stats: dict[str, float] = {}
    daily_mean = _to_float(returns.mean())
    daily_vol = _to_float(returns.std())

    stats["daily_return"] = float(daily_mean)
    stats["daily_vol"] = float(daily_vol)

    # Geometric annualized return using log returns for numerical stability
    log_ret = np.log1p(returns)
    annual_return = float(np.exp(log_ret.mean() * 252) - 1)
    stats["annual_return"] = annual_return
    annual_vol = (
        float(daily_vol * np.sqrt(252))
        if math.isfinite(float(daily_vol))
        else np.nan
    )
    stats["annual_vol"] = annual_vol

    excess_return = annual_return - risk_free_rate
    stats["sharpe"] = (
        float(excess_return / annual_vol)
        if math.isfinite(float(annual_vol)) and float(annual_vol) != 0
        else np.nan
    )

    downside = _to_float(returns[returns < 0].std())
    annual_downside = (
        float(downside * np.sqrt(252))
        if math.isfinite(float(downside)) and float(downside) != 0
        else np.nan
    )
    stats["sortino"] = (
        float(excess_return / annual_downside)
        if math.isfinite(float(annual_downside)) and float(annual_downside) != 0
        else np.nan
    )

    var_level = 0.95
    var = _to_float(returns.quantile(1 - var_level))
    stats["var"] = var
    tail = returns[returns <= var]
    stats["cvar"] = _to_float(tail.mean()) if not tail.empty else np.nan

    equity_curve = (1 + returns).cumprod()
    drawdown = compute_drawdown_curve(equity_curve)
    stats["max_drawdown"] = _to_float(drawdown.min()) if not drawdown.empty else np.nan

    return stats

@st.cache_data(show_spinner=False)
def run_portfolio_simulation(returns_df, num_portfolios=10000, rf=0.02):
    cols = returns_df.columns
    n = len(cols)
    results = np.zeros((3, num_portfolios))
    weights = []
    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    for i in range(num_portfolios):
        w = np.random.random(n); w /= w.sum()
        weights.append(w)
        pret = float(np.dot(mu, w))
        pvol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        results[0, i] = pret
        results[1, i] = pvol
        results[2, i] = (pret - rf) / pvol if pvol else np.nan
    return results, weights

@st.cache_data(show_spinner=False)
def run_monte_carlo(price_df: pd.DataFrame, days=252, simulations=500) -> pd.DataFrame | None:
    if price_df is None or price_df.empty or len(price_df) < 2:
        return None
    try:
        log_returns = np.log(1 + price_df['Close'].pct_change())
        mu = log_returns.mean()
        var = log_returns.var()
        drift = mu - (0.5 * var)
        stdev = log_returns.std()

        daily_returns = np.exp(drift + stdev * np.random.standard_normal((days, simulations)))
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = price_df['Close'].iloc[-1]
        for t in range(1, days):
            price_paths[t] = price_paths[t - 1] * daily_returns[t]

        return pd.DataFrame(price_paths)
    except Exception:
        return None

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("📈 Stock Analyzer Settings")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Single Stock", "Portfolio Analysis"])

st.sidebar.header("Date Range")
c1, c2 = st.sidebar.columns(2)
# Set default end date to today and start date to 2 years ago from today
default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=730)
start_date = c1.date_input("Start Date", default_start_date)
end_date = c2.date_input("End Date", default_end_date)


if analysis_type == "Single Stock":
    ticker_symbol = st.sidebar.text_input("Ticker (e.g., AAPL)", "AAPL").upper().strip()
else:
    tickers_text = st.sidebar.text_area("Tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA")
    ticker_list = sorted({t.strip().upper() for t in tickers_text.split(",") if t.strip()})

with st.sidebar.expander("🤖 AI Enrichment (optional)"):
    use_llm = st.checkbox("Enable LLM summaries (DeepSeek/OpenAI-compatible)", value=False)
    key_in = st.text_input("API Key (won't be stored)", type="password")
    base_url = st.text_input("Base URL (keep default for DeepSeek)", value="https://api.deepseek.com")

run = st.sidebar.button("🚀 Analyze", type="primary")

st.title("📊 Advanced Stock & Portfolio Analyzer — v7.2")

# =============================================================================
# Single Stock
# =============================================================================
if run and analysis_type == "Single Stock":
    with st.spinner(f"Fetching {ticker_symbol} …"):
        price = load_stock_data(ticker_symbol, start_date, end_date)
        company_data = robust_company_blob(ticker_symbol)
        info = company_data["info"]

    if price is None or price.empty:
        st.error(f"No price data for {ticker_symbol} in this range.")
    else:
        price = calculate_indicators(price)
        name = info.get("longName") or info.get("shortName") or ticker_symbol
        st.header(f"🔍 Analysis for {name} (${ticker_symbol})")

        tabs = st.tabs([
            "📈 Technical Chart", "🤖 Technical Summary", "🧾 Fundamentals",
            "🗓️ Earnings & Ratings", "📰 News & Sentiment", "🔮 Forecast (Prophet)",
            "🎲 Monte Carlo Sim", "🏦 Insider & Institutions", "💾 Historical Data"
        ])

        # --- Technical Chart ---
        with tabs[0]:
            st.markdown("##### Chart Controls")
            controls = st.columns(4)
            show_sma20 = controls[0].checkbox("SMA 20", value=True)
            show_sma50 = controls[1].checkbox("SMA 50", value=True)
            show_sma200 = controls[2].checkbox("SMA 200", value=True)
            show_bb = controls[3].checkbox("Bollinger Bands", value=True)

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=price.index, open=price["Open"], high=price["High"], low=price["Low"], close=price["Close"], name="Price"), row=1, col=1)

            # === BUG FIX START: Reordered plotting to draw lines on top of filled areas ===
            # 1. Plot filled areas (Bollinger Bands) first
            if show_bb and "UpperBB" in price.columns:
                fig.add_trace(go.Scatter(x=price.index, y=price["UpperBB"], name="Upper BB", line=dict(color="gray", dash="dash"), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=price.index, y=price["LowerBB"], name="Lower BB", line=dict(color="gray", dash="dash"), fill='tonexty'), row=1, col=1)

            # 2. Plot lines (SMAs) on top
            if show_sma20: fig.add_trace(go.Scatter(x=price.index, y=price["SMA20"], name="SMA20", line=dict(color="yellow")), row=1, col=1)
            if show_sma50: fig.add_trace(go.Scatter(x=price.index, y=price["SMA50"], name="SMA50", line=dict(color="orange")), row=1, col=1)
            if show_sma200: fig.add_trace(go.Scatter(x=price.index, y=price["SMA200"], name="SMA200", line=dict(color="red")), row=1, col=1)
            # === BUG FIX END ===

            fig.add_trace(go.Scatter(x=price.index, y=price["MACD"], name="MACD", line=dict(color="blue")), row=2, col=1)
            fig.add_trace(go.Scatter(x=price.index, y=price["MACD_Signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
            fig.add_trace(go.Scatter(x=price.index, y=price["RSI"], name="RSI", line=dict(color="purple")), row=3, col=1)
            try:
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            except Exception: pass
            fig.update_layout(height=700, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            export_csv_button(price, "⬇️ Export Technicals CSV", f"{ticker_symbol}_technicals.csv")

        # --- Technical Summary ---
        with tabs[1]:
            st.subheader("Signal Snapshot")
            for line in summarize_signals(price):
                st.write(f"- {line}")
            last_close = last_val(price, "Close")
            rsi_v = last_val(price, "RSI")
            macd, macd_sig = last_val(price, "MACD"), last_val(price, "MACD_Signal")
            sma50, sma200 = last_val(price, "SMA50"), last_val(price, "SMA200")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Last Close", safe_metric_fmt(last_close))
            if not np.isnan(rsi_v): k2.metric("RSI", f"{rsi_v:.1f}")
            if not np.isnan(macd) and not np.isnan(macd_sig): k3.metric("MACD−Signal", f"{macd - macd_sig:.3f}")
            if not np.isnan(sma50) and not np.isnan(sma200): k4.metric("SMA50−SMA200", f"{sma50 - sma200:.2f}")

            risk_stats = compute_risk_metrics(price["Close"])
            if risk_stats:
                st.markdown("##### Historical Risk Snapshot")
                display_metrics = [
                    ("Annualized Return", risk_stats.get("annual_return"), "pct"),
                    ("Annualized Volatility", risk_stats.get("annual_vol"), "pct"),
                    ("Sharpe (Rf=2%)", risk_stats.get("sharpe"), "num"),
                    ("Sortino Ratio", risk_stats.get("sortino"), "num"),
                    ("Value at Risk (95%)", risk_stats.get("var"), "pct"),
                    ("Conditional VaR (95%)", risk_stats.get("cvar"), "pct"),
                    ("Max Drawdown", risk_stats.get("max_drawdown"), "pct"),
                ]

                for idx in range(0, len(display_metrics), 3):
                    cols = st.columns(3)
                    for col, (label, value, kind) in zip(cols, display_metrics[idx: idx + 3]):
                        col.metric(label, safe_metric_fmt(value, kind=kind))

                risk_df = pd.DataFrame(
                    {
                        "Metric": [m[0] for m in display_metrics],
                        "Value": [m[1] for m in display_metrics],
                    }
                )
                export_csv_button(risk_df, "⬇️ Export Risk Metrics CSV", f"{ticker_symbol}_risk_metrics.csv")
                st.caption("Computed from daily closes over the selected range; VaR metrics reflect losses (negative values).")

        # --- Fundamentals ---
        with tabs[2]:
            st.subheader("Key Metrics & Valuation")
            colA, colB, colC = st.columns(3)
            # Use info + fast_info fields
            mc = info.get("marketCap")
            pe_t = info.get("trailingPE")
            pe_f = info.get("forwardPE")
            beta = info.get("beta")
            ps_t = info.get("priceToSalesTrailing12Months")
            pb = info.get("priceToBook")

            colA.metric("Market Cap", safe_metric_fmt(mc, "money_b"))
            colA.metric("Beta", safe_metric_fmt(beta))
            colB.metric("Trailing P/E", safe_metric_fmt(pe_t))
            colB.metric("Forward P/E", safe_metric_fmt(pe_f))
            colC.metric("P/S (TTM)", safe_metric_fmt(ps_t))
            colC.metric("P/B", safe_metric_fmt(pb))

            st.divider()
            st.markdown(f"**Sector:** {info.get('sector','N/A')} | **Industry:** {info.get('industry','N/A')} | **Website:** {info.get('website','N/A')}")

            st.markdown("#### 🏢 About Company")
            about = (info.get("longBusinessSummary") or "").strip()
            if not about and use_llm:
                client = get_llm_client(api_key=key_in, base_url=base_url)
                prompt = f"Give a short, factual overview (80-120 words) of the public company with ticker {ticker_symbol}. Avoid hype."
                about = llm_summarize(client, prompt) or ""
            st.write(about if about else "No company description available.")

            st.markdown("#### 📈 Historical Ratio Analysis")
            ratios_df = pd.DataFrame()
            try:
                fin_T = company_data["financials"].T if not company_data["financials"].empty else pd.DataFrame()
                bs_T = company_data["balance_sheet"].T if not company_data["balance_sheet"].empty else pd.DataFrame()
                if not fin_T.empty and not bs_T.empty:
                    common_index = fin_T.index.intersection(bs_T.index)
                    net_income = pd.to_numeric(fin_T.loc[common_index].get('Net Income'), errors='coerce')
                    total_liab = pd.to_numeric(bs_T.loc[common_index].get('Total Liabilities Net Minority Interest'), errors='coerce')
                    stockholder_equity = pd.to_numeric(bs_T.loc[common_index].get('Stockholders Equity'), errors='coerce')
                    ratios_df['Return on Equity (ROE)'] = (net_income / stockholder_equity)
                    ratios_df['Debt to Equity'] = (total_liab / stockholder_equity)
                    ratios_df = ratios_df.replace([np.inf, -np.inf], np.nan).dropna()
                    ratios_df.index.name = "Year"
            except Exception: pass

            if not ratios_df.empty:
                r1, r2 = st.columns(2)
                r1.plotly_chart(px.bar(ratios_df, y='Return on Equity (ROE)', title='Return on Equity (ROE)').update_layout(yaxis_tickformat='.2%'), use_container_width=True)
                r2.plotly_chart(px.bar(ratios_df, y='Debt to Equity', title='Debt to Equity Ratio'), use_container_width=True)
            else:
                 st.info("Historical ratio data could not be calculated.")

            st.markdown("#### Income Statement (best-available)")
            st.dataframe(company_data["financials"])
            st.markdown("#### Balance Sheet (best-available)")
            st.dataframe(company_data["balance_sheet"])

            info_for_export = {k: str(v) for k, v in info.items()}
            info_df = pd.DataFrame(info_for_export, index=[0])
            export_csv_button(info_df, "⬇️ Export Info JSON → CSV", f"{ticker_symbol}_info.csv")

        # --- Earnings & Ratings ---
        with tabs[3]:
            st.subheader("Earnings & Analyst Ratings")
            calendar = company_data["calendar"]
            earnings_dates = company_data["earnings_dates"]
            recommendations = company_data["recommendations"]

            if not calendar.empty and 'Earnings Date' in calendar.columns:
                next_earnings_date = calendar['Earnings Date'].iloc[0]
                if isinstance(next_earnings_date, (pd.Timestamp, datetime)):
                    st.metric("Next Earnings Date", next_earnings_date.strftime('%Y-%m-%d'))
                else:
                    st.metric("Next Earnings Date", str(next_earnings_date))
            else:
                st.info("Next earnings date not available.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Historical EPS Surprises")
                if not earnings_dates.empty:
                    earnings_dates = earnings_dates.dropna(subset=['Reported EPS', 'EPS Estimate']).tail(8)
                    fig_eps = go.Figure()
                    fig_eps.add_trace(go.Bar(x=earnings_dates.index, y=earnings_dates['EPS Estimate'], name='Estimate', marker_color='grey'))
                    fig_eps.add_trace(go.Bar(x=earnings_dates.index, y=earnings_dates['Reported EPS'], name='Reported', marker_color='blue'))
                    fig_eps.update_layout(barmode='group', title="EPS: Reported vs. Estimate", height=400)
                    st.plotly_chart(fig_eps, use_container_width=True)
                else:
                    st.info("Historical earnings data not available.")

            with c2:
                st.markdown("##### Analyst Recommendations")
                if not recommendations.empty:
                    rating_column = None
                    possible_columns = ['To Grade', 'Action', 'Rating'] # List of potential column names
                    for col in possible_columns:
                        if col in recommendations.columns:
                            rating_column = col
                            break

                    if rating_column:
                        rec_counts = recommendations[rating_column].value_counts()
                        fig_rec = px.pie(values=rec_counts.values, names=rec_counts.index, title=f'Recommendation Distribution (by {rating_column})', hole=0.3)
                        st.plotly_chart(fig_rec, use_container_width=True)
                    else:
                        st.info("Could not find a valid recommendation column in the data.")
                        st.write("Available columns:", recommendations.columns.tolist())
                else:
                    st.info("Analyst recommendation data not available.")


        # --- News & Sentiment ---
        with tabs[4]:
            st.subheader("Recent News")
            news = company_data["news"]
            if not news:
                st.info("No recent news available from Yahoo for this ticker.")
            else:
                sia = try_init_vader()
                cleaned = []
                for item in news:
                    title = (item.get("title") or "").strip() or "Untitled Article"
                    link = item.get("link") or "#"
                    publisher = (item.get("publisher") or "").strip()
                    ts = item.get("providerPublishTime")
                    date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if isinstance(ts, (int, float)) and ts > 0 else ""
                    score = float(sia.polarity_scores(title)["compound"]) if sia and title else np.nan
                    cleaned.append({"Date": date_str, "Title": title, "Publisher": publisher, "Sentiment": score, "Link": link})

                news_df = pd.DataFrame(cleaned)
                if use_llm and not news_df.empty:
                    client = get_llm_client(api_key=key_in, base_url=base_url)
                    titles = "\n".join(f"- {t}" for t in news_df["Title"].tolist()[:10])
                    brief = llm_summarize(client, f"Summarize these headlines for {ticker_symbol} in 4 bullets:\n{titles}", max_tokens=220)
                    if brief:
                        st.markdown("**AI News Brief**"); st.write(brief); st.divider()

                if not news_df.empty:
                    for _, r in news_df.iterrows():
                        st.markdown(f"**[{r['Title']}]({r['Link']})**")
                        st.caption(f"{r['Publisher']} — {r['Date']}" + (f" | Sentiment: {r['Sentiment']:.2f}" if not pd.isna(r["Sentiment"]) else ""))
                        st.divider()
                    export_csv_button(news_df, "⬇️ Export News CSV", f"{ticker_symbol}_news.csv")

        # --- Forecast (Prophet) ---
        with tabs[5]:
            st.subheader("30-Day Price Forecast (via Prophet)")
            if not _HAS_PROPHET:
                st.info("Prophet not installed. Install with `pip install prophet` to enable forecasts.")
            else:
                fc = forecast_with_prophet(price, periods=30)
                if fc is None or fc.empty:
                    st.info("Not enough data to fit Prophet or model failed to converge.")
                else:
                    hist = price.reset_index()
                    date_col = "Date" if "Date" in hist.columns else hist.columns[0]
                    figf = go.Figure()
                    figf.add_trace(go.Scatter(x=hist[date_col], y=hist["Close"], name="Actual", line=dict(color='blue')))
                    figf.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast", line=dict(color='orange')))
                    figf.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], name="Upper Bound", line=dict(dash="dash", color='gray')))
                    figf.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], name="Lower Bound", fill='tonexty', line=dict(dash="dash", color='gray')))
                    figf.update_layout(height=450, xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(figf, use_container_width=True)
                    export_csv_button(fc, "⬇️ Export Forecast CSV", f"{ticker_symbol}_forecast.csv")

        # --- Monte Carlo Sim ---
        with tabs[6]:
            st.subheader("1-Year Price Projection (Monte Carlo)")
            mc_paths = run_monte_carlo(price, days=252, simulations=500)
            if mc_paths is None or mc_paths.empty:
                st.info("Could not run Monte Carlo simulation. More historical data might be needed.")
            else:
                fig_mc = go.Figure()
                fig_mc.add_traces([go.Scatter(y=mc_paths[col], mode='lines', line=dict(width=0.5, color='gray'), opacity=0.3, showlegend=False) for col in mc_paths.columns])
                mean_path = mc_paths.mean(axis=1)
                fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(width=3, color='orange'), name='Mean Simulation Path'))
                fig_mc.update_layout(height=500, xaxis_title="Trading Days from Today", yaxis_title="Simulated Price", title="500 Simulations of Future Price Paths")
                st.plotly_chart(fig_mc, use_container_width=True)

                last_day_sims = mc_paths.iloc[-1]
                st.subheader("Simulation Summary (at Day 252)")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Mean Price", f"${mean_path.iloc[-1]:.2f}")
                mc2.metric("5th Percentile", f"${last_day_sims.quantile(0.05):.2f}")
                mc3.metric("95th Percentile", f"${last_day_sims.quantile(0.95):.2f}")
                st.caption(f"Based on historical volatility and returns. Current Price: ${price['Close'].iloc[-1]:.2f}")

        # --- Insider & Institutions ---
        with tabs[7]:
            cI, cH = st.columns(2)
            cI.subheader("Insider Transactions")
            insider_df = company_data["insider"]
            if not insider_df.empty:
                cI.dataframe(insider_df)
                export_csv_button(insider_df, "⬇️ Export Insider CSV", f"{ticker_symbol}_insider.csv")
            else:
                cI.info("No insider transaction dataset available from Yahoo for this ticker.")

            cH.subheader("Institutional Holders")
            holders_df = company_data["holders"]
            if not holders_df.empty:
                cH.dataframe(holders_df)
                export_csv_button(holders_df, "⬇️ Export Holders CSV", f"{ticker_symbol}_holders.csv")
            else:
                cH.info("No institutional holders dataset available from Yahoo for this ticker.")

        # --- Historical Data ---
        with tabs[8]:
            st.dataframe(price.sort_index(ascending=False))
            export_csv_button(price, "⬇️ Export OHLCV CSV", f"{ticker_symbol}_ohlcv.csv")

# =============================================================================
# Portfolio Analysis
# =============================================================================
elif run and analysis_type == "Portfolio Analysis":
    if len(ticker_list) < 2:
        st.warning("Please enter at least two tickers.")
    else:
        st.header(f"📊 Portfolio Analysis ({', '.join(ticker_list)})")
        with st.spinner("Downloading portfolio prices…"):
            data = yf.download(" ".join(ticker_list), start=start_date, end=end_date, progress=False)
            price_data = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data[['Close']].copy()
            if not isinstance(data.columns, pd.MultiIndex): price_data.columns = [ticker_list[0]]
            price_data = price_data.dropna(axis="columns", how="all").dropna(how="any")

        if price_data.empty or len(price_data) <= 1:
            st.error("Could not build a valid aligned price matrix. Try a wider range.")
        else:
            returns_df = price_data.pct_change().dropna()
            tabs = st.tabs(["📈 Performance & Risk", "🧮 Portfolio Optimization", "🧭 Sector Allocation", "🔗 Correlation", "💾 Raw Data"])

            with tabs[0]:
                st.subheader("Cumulative Returns")
                eq = (1 + returns_df).cumprod()
                st.line_chart(eq)

                st.subheader("Portfolio Risk Metrics")
                portfolio_returns = returns_df.mean(axis=1) # Equal weighted portfolio
                confidence_level = 0.95
                var_95 = portfolio_returns.quantile(1 - confidence_level)
                cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

                risk_cols = st.columns(2)
                risk_cols[0].metric(
                    label=f"Value at Risk (VaR) at {confidence_level:.0%}",
                    value=f"{var_95:.2%}",
                    help="Indicates the maximum potential loss for an equal-weighted portfolio over a one-day period at the given confidence level."
                )
                risk_cols[1].metric(
                    label=f"Conditional VaR (CVaR) at {confidence_level:.0%}",
                    value=f"{cvar_95:.2%}",
                    help="Represents the average loss that would be incurred on the days when the loss is greater than the VaR threshold."
                )
                st.divider()

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Drawdown (Avg)")
                    dd = compute_drawdown_curve(eq.mean(axis=1))
                    fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", fill='tozeroy'))
                    fig_dd.update_layout(height=300, yaxis_tickformat=".0%")
                    st.plotly_chart(fig_dd, use_container_width=True)
                with c2:
                    st.subheader("Rolling 60-Day Sharpe (approx)")
                    roll_mean = portfolio_returns.rolling(60).mean()
                    roll_std = portfolio_returns.rolling(60).std() + 1e-9
                    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
                    st.line_chart(roll_sharpe, height=300)

                st.subheader("Correlation Matrix")
                fig_corr, ax_corr = plt.subplots()
                sns.heatmap(returns_df.corr(), annot=True, cmap="viridis", ax=ax_corr)
                st.pyplot(fig_corr)

                export_csv_button(returns_df, "⬇️ Export Returns CSV", "portfolio_returns.csv")
                export_csv_button(eq, "⬇️ Export Equity Curve CSV", "portfolio_equity_curve.csv")

            with tabs[1]:
                st.subheader("Efficient Frontier & Optimal Portfolios")
                with st.spinner("Running Monte Carlo…"):
                    results, weights = run_portfolio_simulation(returns_df)
                idx_ms = int(np.nanargmax(results[2]))
                idx_mv = int(np.nanargmin(results[1]))

                ms_ret, ms_vol = results[0, idx_ms], results[1, idx_ms]
                mv_ret, mv_vol = results[0, idx_mv], results[1, idx_mv]
                w_ms, w_mv = weights[idx_ms], weights[idx_mv]

                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=results[1, :], y=results[0, :], mode='markers', marker=dict(color=results[2, :], showscale=True, size=7, line=dict(width=1), colorscale="viridis", colorbar=dict(title="Sharpe")), name="Portfolios"))
                fig_ef.add_trace(go.Scatter(x=[ms_vol], y=[ms_ret], mode='markers', marker=dict(color='red', size=14, symbol='star'), name='Max Sharpe'))
                fig_ef.add_trace(go.Scatter(x=[mv_vol], y=[mv_ret], mode='markers', marker=dict(color='green', size=14, symbol='star'), name='Min Vol'))
                fig_ef.update_layout(title="Efficient Frontier", xaxis_title="Volatility (Std. Dev)", yaxis_title="Expected Return")
                st.plotly_chart(fig_ef, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Max Sharpe Weights**")
                    df_w1 = pd.DataFrame(w_ms, index=returns_df.columns, columns=["Weight"]).sort_values("Weight", ascending=False)
                    st.table(df_w1.style.format("{:.2%}"))
                    export_csv_button(df_w1, "⬇️ Export Max-Sharpe Weights CSV", "weights_max_sharpe.csv")
                with c2:
                    st.write("**Min Vol Weights**")
                    df_w2 = pd.DataFrame(w_mv, index=returns_df.columns, columns=["Weight"]).sort_values("Weight", ascending=False)
                    st.table(df_w2.style.format("{:.2%}"))
                    export_csv_button(df_w2, "⬇️ Export Min-Vol Weights CSV", "weights_min_vol.csv")

            with tabs[2]:
                st.subheader("Sector Allocation (Equal Weight)")
                sectors = []
                with st.spinner("Fetching sector data..."):
                    for tkr in returns_df.columns:
                        try: sectors.append(yf.Ticker(tkr).get_info().get("sector", "N/A"))
                        except Exception: sectors.append("N/A")
                sector_df = pd.DataFrame({"Ticker": returns_df.columns, "Sector": sectors})
                st.dataframe(sector_df)
                try:
                    fig_sector = px.pie(sector_df, names="Sector", title="Sector Allocation")
                    st.plotly_chart(fig_sector, use_container_width=True)
                except Exception: st.info("Pie chart unavailable.")
                export_csv_button(sector_df, "⬇️ Export Sectors CSV", "portfolio_sectors.csv")

            with tabs[3]:
                st.subheader("Correlation Network (|r| ≥ 0.6)")
                corr = returns_df.corr()
                if _HAS_NX:
                    G = nx.from_pandas_adjacency(corr[abs(corr) >= 0.6].fillna(0))
                    G.remove_edges_from(nx.selfloop_edges(G))
                    if G.number_of_edges() == 0:
                        st.info("No strong correlations above threshold.")
                    else:
                        pos = nx.spring_layout(G, seed=42)
                        edge_traces = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=1, color='gray'), hoverinfo='none'))

                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="top center", marker=dict(size=14, color='skyblue'))

                        fig_net = go.Figure(data=edge_traces + [node_trace])
                        fig_net.update_layout(height=500, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                        st.plotly_chart(fig_net, use_container_width=True)
                else:
                    st.info("Install `networkx` to enable the network view: `pip install networkx`")

            with tabs[4]:
                st.dataframe(price_data.sort_index(ascending=False))
                export_csv_button(price_data, "⬇️ Export Prices CSV", "portfolio_prices.csv")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data via Yahoo Finance | v7.2 — Chart Fix")