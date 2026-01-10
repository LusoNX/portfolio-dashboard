import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# =========================
# PAGE SETUP
# =========================
st.set_page_config(layout="wide")
st.title("Portfolio Performance Dashboard (USD)")

# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_excel(
        "portfolio_values_app.xlsx",
        index_col=0,
        parse_dates=True
    )
    df.index.name = "date"
    df = df[df.index.weekday < 5].dropna()

    df_w = pd.read_excel(
        "next_week_weights_estimate_app.xlsx",
        index_col=0
    )

    return df, df_w


df, df_w = load_data()

# =========================
# METRIC FUNCTIONS
# =========================
def perf_stats(returns):
    mean_ann = returns.mean() * 252
    vol_ann = returns.std() * np.sqrt(252)
    sharpe = mean_ann / vol_ann if vol_ann != 0 else np.nan

    cum = (1 + returns).cumprod()
    cum_ret = cum.iloc[-1] - 1
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    return cum_ret, vol_ann, sharpe, max_dd

def tracking_error(port_ret, bench_ret):
    diff = port_ret - bench_ret
    return diff.std() * np.sqrt(252)

# =========================
# TABS
# =========================
tab_perf, tab_week = st.tabs(
    ["📈 Performance (USD)", "📊 Weekly Performance & Weights"]
)

# ======================================================
# TAB 1 — PERFORMANCE
# ======================================================
with tab_perf:
    st.subheader("Portfolio vs S&P 500 (USD)")

    split_date = pd.to_datetime("2025-10-15")

    segments = {
        "October to Date": df[df.index >= split_date].copy(),
        "January to September": df[df.index < split_date].copy()
        
    }

    for name, df_seg in segments.items():

        if name == "January to September":
            df_seg["Port Perf EUR"] = df_seg["buying_power"] / df_seg["buying_power"].iloc[0]
            df_seg["Portfolio USD"] = df_seg["buying_power"] * df_seg["FX_EURUSD"]
            df_seg["Port Perf USD"] = df_seg["Portfolio USD"] / df_seg["Portfolio USD"].iloc[0]
        else:

            df_seg["Port Perf USD"] = df_seg["buying_power"] / df_seg["buying_power"].iloc[0]
            
        # df_seg["Portfolio USD"] = df_seg["portfolio_value"] * df_seg["FX_EURUSD"]
        # df_seg["Port Perf USD"] = df_seg["Portfolio USD"] / df_seg["Portfolio USD"].iloc[0]
        df_seg["SPX Perf USD"] = df_seg["SPX USD"] / df_seg["SPX USD"].iloc[0]

        df_seg["Port Ret"] = df_seg["Port Perf USD"].pct_change()
        df_seg["SPX Ret"] = df_seg["SPX Perf USD"].pct_change()

        df_ret = df_seg.dropna()

        fig = px.line(
            df_seg.reset_index(),
            x="date",
            y=["Port Perf USD", "SPX Perf USD"],
            title=f"{name} – Portfolio vs S&P 500 (USD, Indexed)",
            labels={"value": "Accumulated Return", "variable": ""}
        )
        st.plotly_chart(fig, use_container_width=True)

        pm, pv, ps, pdd = perf_stats(df_ret["Port Ret"])
        bm, bv, bs, bdd = perf_stats(df_ret["SPX Ret"])
        te = tracking_error(df_ret["Port Ret"], df_ret["SPX Ret"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚀 Portfolio")
            st.metric("Accumulated Period Return", f"{pm:.2%}")
            st.metric("Volatility", f"{pv:.2%}")
            st.metric("Sharpe", f"{ps:.2f}")
            st.metric("Max Drawdown", f"{pdd:.2%}")

        with col2:
            st.markdown("#### 📈 S&P 500")
            st.metric("Accumulated Period Return", f"{bm:.2%}")
            st.metric("Volatility", f"{bv:.2%}")
            st.metric("Sharpe", f"{bs:.2f}")
            st.metric("Max Drawdown", f"{bdd:.2%}")

        st.markdown(f"**📉 Tracking Error:** {te:.2%}")
        st.divider()

# ======================================================
# TAB 2 — WEEKLY PERFORMANCE & WEIGHTS
# ======================================================
with tab_week:
    st.subheader("Weekly Performance Overview")

    # =========================
    # WEEKLY PERFORMANCE
    # =========================
    #df_week = df.iloc[-4:].copy()

    df_week = df.iloc[-5:]
    df_week["Port EUR"] = df_week["buying_power"] / df_week["FX_EURUSD"]
    df_week["Port Perf EUR"] = df_week["Port EUR"] / df_week["Port EUR"].iloc[0]
    df_week["Port Perf USD"] = df_week["buying_power"] / df_week["buying_power"].iloc[0]
    df_week["SPX Perf USD"] = df_week["SPX USD"] / df_week["SPX USD"].iloc[0]

    week_start = df_week.index.min().date()
    week_end = df_week.index.max().date()

    port_ret = df_week["Port Perf USD"].iloc[-1] - 1
    spx_ret = df_week["SPX Perf USD"].iloc[-1] - 1
    outperf = port_ret - spx_ret

    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 Portfolio (USD)", f"{port_ret:+.2%}")
    c2.metric("📈 S&P 500 (USD)", f"{spx_ret:+.2%}")
    c3.metric("⚖️ Out / Underperformance", f"{outperf:+.2%}")

    st.caption(f"📅 {week_start} → {week_end}")

    st.divider()

    # =========================
    # TICKER PERFORMANCE
    # =========================
    st.subheader("Trading Tickers – Weekly Performance")

    trading_tickers = df_w.columns[:-1]  # exclude last column (date)

    ticker_perf = []

    for ticker in trading_tickers:
        try:
            data = yf.download(
                ticker,
                start=week_start,
                end=week_end,
                progress=False
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]

            ret = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
            
            ticker_perf.append((ticker, ret))
        except:
            pass
    


    
    if ticker_perf:
        perf_df = pd.DataFrame(ticker_perf, columns=["Ticker", "Return"])

        perf_df["Return"] = perf_df["Return"].round(4)

     
        fig = px.bar(
            perf_df,
            x="Return",
            y="Ticker",
            orientation="h",
            text="Return",
            title="Weekly Ticker Performance"
        )

        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Weekly Return"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # NEXT WEEK WEIGHTS
    # =========================
    st.subheader("Next Week Portfolio Weights")

    last_update = df_w["Last Update Date"].iloc[0]

    weights = (
        df_w
        .drop(columns=["Last Update Date"])
        .T
        .reset_index()
    )
    weights.columns = ["Asset", "Weight"]
    weights["Weight"] = weights["Weight"].round(2)
    weights = weights[weights["Weight"] > 0]

    fig = px.bar(
        weights,
        x="Weight",
        y="Asset",
        orientation="h",
        title=f"Next Week Weights — Last Update Date: {last_update}",
        text="Weight"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Weight"
    )

    st.plotly_chart(fig, use_container_width=True)
