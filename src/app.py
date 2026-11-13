import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from pathlib import Path
st.set_page_config(
    page_title="AI Stock Market Intelligence",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #17243A, #111827);
    border: 1px solid #263750;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    color: #38BDF8;
}
h1, h2, h3 {
    color: #F8FAFC;
}
</style>
""", unsafe_allow_html=True)

TICKERS = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS"
}

project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "outputs" / "model_comparison.csv"

@st.cache_data(ttl=50, show_spinner=False)
def get_market_data(ticker):
    stock = yf.Ticker(ticker)

    intraday = stock.history(
        period="1d",
        interval="1m",
        auto_adjust=False
    )

    daily = stock.history(
        period="5d",
        interval="1d",
        auto_adjust=False
    )

    return intraday, daily

st.title("📈 AI-Powered Stock Market Forecasting")
st.subheader("Risk Intelligence Dashboard")
st.caption("TCS  |  Infosys  |  Reliance")
st.divider()

if file_path.exists():

    df = pd.read_csv(file_path)

    st.sidebar.header("Dashboard Controls")

    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        df["Company"].tolist()
    )

    ticker = next(
        (
            value for name, value in TICKERS.items()
            if name.lower() in selected_stock.lower()
        ),
        None
    )

    # ==================================================
    # LIVE MARKET MONITOR
    # ==================================================

    st.header("📡 Live Market Monitor")

    @st.fragment(run_every="60s")
    def live_market_panel(stock_ticker, stock_name):

        if not stock_ticker:
            st.warning("Stock ticker mapping not found.")
            return

        try:
            intraday, daily = get_market_data(stock_ticker)

            if intraday.empty:
                st.warning(
                    "Intraday data unavailable. "
                    "Market may be closed or data unavailable."
                )
                return

            latest = intraday.iloc[-1]
            current_price = float(latest["Close"])

            previous_close = (
                float(daily["Close"].iloc[-2])
                if len(daily) >= 2 else None
            )

            c1, c2, c3 = st.columns(3)

            if previous_close:
                change_pct = (
                    (current_price - previous_close)
                    / previous_close
                ) * 100

                c1.metric(
                    "Latest Available Price",
                    f"₹{current_price:,.2f}",
                    f"{change_pct:+.2f}%"
                )
            else:
                c1.metric(
                    "Latest Available Price",
                    f"₹{current_price:,.2f}"
                )

            c2.metric(
                "Previous Close",
                f"₹{previous_close:,.2f}"
                if previous_close else "Unavailable"
            )

            c3.metric(
                "Latest Volume",
                f"{int(latest['Volume']):,}"
            )

            st.caption(
                f"Latest available data: {intraday.index[-1]} "
                "| Auto-refresh: 60 seconds"
            )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.7, 0.3]
            )

            fig.add_trace(
                go.Candlestick(
                    x=intraday.index,
                    open=intraday["Open"],
                    high=intraday["High"],
                    low=intraday["Low"],
                    close=intraday["Close"],
                    name="Price",
                    increasing_line_color="#22C55E",
                    decreasing_line_color="#F43F5E"
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Bar(
                    x=intraday.index,
                    y=intraday["Volume"],
                    marker_color="#38BDF8",
                    name="Volume"
                ),
                row=2,
                col=1
            )

            fig.update_layout(
                title=f"{stock_name} Intraday Price",
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#E2E8F0"),
                height=550,
                xaxis_rangeslider_visible=False,
                showlegend=False
            )

            fig.update_yaxes(
                title_text="Price (₹)",
                gridcolor="#263449",
                row=1,
                col=1
            )

            fig.update_yaxes(
                title_text="Volume",
                gridcolor="#263449",
                row=2,
                col=1
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Source: Yahoo Finance. Data may be delayed "
                "or unavailable."
            )

        except Exception as error:
            st.error(f"Market data error: {error}")

    live_market_panel(ticker, selected_stock)

    st.divider()
    selected = df[
        df["Company"] == selected_stock
    ].iloc[0]

    st.header(f"📊 {selected_stock} Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Random Forest Accuracy",
        f"{selected['RF_Accuracy_%']:.2f}%"
    )

    c2.metric(
        "XGBoost MAPE",
        f"{selected['XGB_MAPE_%']:.2f}%"
    )

    c3.metric(
        "LSTM MAPE",
        f"{selected['LSTM_MAPE_%']:.2f}%"
    )

    st.divider()
    st.header("🎯 Prediction Model Comparison")

    st.caption(
        "Historical test-set error comparison. "
        "Lower MAPE means lower prediction error."
    )

    xgb_mape = float(selected["XGB_MAPE_%"])
    lstm_mape = float(selected["LSTM_MAPE_%"])

    # Model comparison cards

    best_model = (
        "XGBoost" if xgb_mape < lstm_mape
        else "LSTM" if lstm_mape < xgb_mape
        else "Tie"
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric("XGBoost Error", f"{xgb_mape:.2f}%")

    with p2:
        st.metric("LSTM Error", f"{lstm_mape:.2f}%")

    with p3:
        st.metric("Lower Error Model", best_model)
    prediction_fig = go.Figure()

    prediction_fig.add_trace(
        go.Bar(
            x=["XGBoost", "LSTM"],
            y=[xgb_mape, lstm_mape],
            marker=dict(
                color=["#38BDF8", "#A78BFA"],
                line=dict(
                    color=["#7DD3FC", "#C4B5FD"],
                    width=2
                )
            ),
            width=0.48,
            text=[
                f"{xgb_mape:.2f}%",
                f"{lstm_mape:.2f}%"
            ],
            textposition="outside",
            textfont=dict(
                size=17,
                color="#FFFFFF"
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "MAPE: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    prediction_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#E2E8F0"),
        height=420,
        title=dict(
            text="Model Error Performance",
            x=0.03,
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Prediction Models",
            showgrid=False
        ),
        yaxis=dict(
            title="MAPE (%)",
            showgrid=True,
            gridcolor="#263449",
            rangemode="tozero"
        ),
        margin=dict(
            l=50,
            r=40,
            t=80,
            b=60
        ),
        showlegend=False
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )

    if best_model == "Tie":
        st.info("Both models have equal MAPE.")
    else:
        st.success(
            f"{best_model} has the lower historical test-set MAPE."
        )

    st.divider()

    st.header("⚠️ GARCH Risk Intelligence")

    st.caption(
        "Estimated 5-trading-day volatility. "
        "These are model-based historical risk estimates, "
        "not live risk predictions."
    )

    risk_df = df[
        ["Company", "GARCH_5D_Volatility_%"]
    ].copy()

    risk_df["GARCH_5D_Volatility_%"] = pd.to_numeric(
        risk_df["GARCH_5D_Volatility_%"],
        errors="coerce"
    )

    risk_df = risk_df.dropna()

    if not risk_df.empty:

        # Three individual gauge indicators

        gauge_columns = st.columns(len(risk_df))

        max_risk = max(
            float(risk_df["GARCH_5D_Volatility_%"].max()) * 1.25,
            1.0
        )

        for column, (_, row) in zip(
            gauge_columns,
            risk_df.iterrows()
        ):

            company = row["Company"]
            volatility = float(
                row["GARCH_5D_Volatility_%"]
            )

            with column:

                st.markdown(
                    f"### {company}"
                )

                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=volatility,
                        number=dict(
                            suffix="%",
                            font=dict(
                                size=27,
                                color="#F8FAFC"
                            )
                        ),
                        gauge=dict(
                            axis=dict(
                                range=[0, max_risk],
                                tickcolor="#CBD5E1"
                            ),
                            bar=dict(
                                color="#38BDF8",
                                thickness=0.25
                            ),
                            bgcolor="#17243A",
                            borderwidth=1,
                            bordercolor="#334155",
                            steps=[
                                dict(
                                    range=[0, max_risk],
                                    color="#17243A"
                                )
                            ]
                        )
                    )
                )

                gauge_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    font=dict(color="#E2E8F0"),
                    height=260,
                    margin=dict(
                        l=20,
                        r=20,
                        t=20,
                        b=10
                    )
                )

                st.plotly_chart(
                    gauge_fig,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

                st.caption(
                    "Estimated 5-day volatility"
                )

        st.divider()
        st.subheader("📊 Volatility Ranking")

        ranked = risk_df.sort_values(
            "GARCH_5D_Volatility_%",
            ascending=False
        ).reset_index(drop=True)

        ranked["Rank"] = range(1, len(ranked) + 1)

        ranked = ranked[
            [
                "Rank",
                "Company",
                "GARCH_5D_Volatility_%"
            ]
        ]

        ranked = ranked.rename(
            columns={
                "GARCH_5D_Volatility_%":
                "Estimated 5-Day Volatility (%)"
            }
        )

        st.dataframe(
            ranked.style.format({
                "Estimated 5-Day Volatility (%)": "{:.3f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("GARCH risk data is unavailable.")

    st.divider()
    st.header("📋 Complete Model Comparison")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.error("Model comparison CSV file not found!")
st.divider()

st.caption(
    "Educational research project | Not investment advice. "
    "Historical model metrics do not automatically update."
)
