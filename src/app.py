
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Stock Market Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Powered Stock Market Forecasting")
st.subheader("Risk Intelligence Dashboard")
st.caption("TCS | Infosys | Reliance")
st.divider()

project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "outputs" / "model_comparison.csv"

if file_path.exists():

    df = pd.read_csv(file_path)

    st.sidebar.header("Dashboard Controls")

    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        df["Company"].tolist()
    )

    selected = df[df["Company"] == selected_stock].iloc[0]

    st.header(f"📊 {selected_stock} Performance Overview")

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

    # Prediction model comparison
    st.header("📉 Prediction Model Comparison")

    st.write(
        f"**Selected Stock:** {selected_stock}"
    )

    st.caption(
        "Mean Absolute Percentage Error (MAPE) – "
        "Lower value indicates lower test-set prediction error."
    )

    error_df = pd.DataFrame({
        "Model": ["XGBoost", "LSTM"],
        "MAPE (%)": [
            selected["XGB_MAPE_%"],
            selected["LSTM_MAPE_%"]
        ]
    })

    fig1 = px.bar(
        error_df,
        x="MAPE (%)",
        y="Model",
        orientation="h",
        text="MAPE (%)",
        color="Model",
        color_discrete_map={
            "XGBoost": "#00B4D8",
            "LSTM": "#A78BFA"
        },
        template="plotly_dark"
    )

    max_error = max(error_df["MAPE (%)"].max(), 1)

    fig1.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig1.update_layout(
        height=330,
        xaxis=dict(
            title="Prediction Error (MAPE %)",
            range=[0, max_error * 1.35],
            showgrid=True
        ),
        yaxis_title="",
        showlegend=False,
        margin=dict(l=30, r=70, t=35, b=50),
        bargap=0.45
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    better_model = (
        "XGBoost"
        if selected["XGB_MAPE_%"] < selected["LSTM_MAPE_%"]
        else "LSTM"
        if selected["LSTM_MAPE_%"] < selected["XGB_MAPE_%"]
        else "Both models have equal MAPE"
    )

    st.info(f"Lower test-set MAPE: {better_model}")

    st.divider()

    # GARCH Risk Analysis
    st.header("⚠️ GARCH Risk Intelligence")
    st.caption("Estimated 5-trading-day volatility across all stocks.")

    risk_df = df[
        ["Company", "GARCH_5D_Volatility_%"]
    ].copy()

    risk_df = risk_df.sort_values(
        "GARCH_5D_Volatility_%"
    )

    fig2 = px.bar(
        risk_df,
        x="GARCH_5D_Volatility_%",
        y="Company",
        orientation="h",
        text="GARCH_5D_Volatility_%",
        color="Company",
        template="plotly_dark"
    )

    fig2.update_traces(
        texttemplate="%{text:.3f}%",
        textposition="outside"
    )

    fig2.update_layout(
        height=350,
        xaxis_title="Estimated Volatility (%)",
        yaxis_title="",
        margin=dict(l=30, r=60, t=40, b=50)
    )

    st.plotly_chart(fig2, use_container_width=True)

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
    "Educational research project | Not investment advice."
)



