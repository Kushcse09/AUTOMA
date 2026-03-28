import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_openai import ChatOpenAI

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Business Decision Assistant",
    page_icon="📊",
    layout="wide"
)

# ================= HEADER =================
st.title("🚀 AI Business Decision Assistant")
st.markdown(
    "Empowering **data-driven decisions** for small businesses using AI 🤖"
)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Settings")

groq_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.info("Upload a dataset to begin analysis")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:

    # ================= LOAD DATA =================
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset Loaded Successfully")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # ================= DATA PREVIEW =================
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ================= KPI DASHBOARD =================
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    if 'Sales' in df.columns:
        col1.metric("💰 Total Sales", f"{df['Sales'].sum():,.0f}")

    if 'Profit' in df.columns:
        col2.metric("📈 Total Profit", f"{df['Profit'].sum():,.0f}")

    if 'Quantity' in df.columns:
        col3.metric("📦 Total Quantity", f"{df['Quantity'].sum():,.0f}")

    # ================= BUSINESS INSIGHTS =================
    st.subheader("🧠 Business Insights")

    insights = []

    try:
        if 'Product' in df.columns and 'Sales' in df.columns:
            top_product = df.groupby('Product')['Sales'].sum().idxmax()
            insights.append(f"🏆 Top Product: **{top_product}**")

        if 'Region' in df.columns and 'Sales' in df.columns:
            best_region = df.groupby('Region')['Sales'].sum().idxmax()
            insights.append(f"🌍 Best Region: **{best_region}**")

        if 'Profit' in df.columns:
            low_profit = df[df['Profit'] < df['Profit'].mean()]
            insights.append(f"⚠️ {len(low_profit)} records have below-average profit")

    except:
        insights.append("⚠️ Unable to compute some insights (check column names)")

    for i in insights:
        st.write("👉", i)

    # ================= ALERT SYSTEM =================
    st.subheader("🚨 AI Alerts")

    try:
        if 'Profit' in df.columns and df['Profit'].mean() < 1000:
            st.error("⚠️ Low profitability detected!")

        if 'Sales' in df.columns and df['Sales'].max() > 5 * df['Sales'].mean():
            st.warning("⚠️ Sales spike anomaly detected!")

    except:
        st.info("No alerts generated")

    # ================= VISUALIZATION =================
    st.subheader("📈 Data Visualization")

    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column to visualize", numeric_cols)

        fig, ax = plt.subplots()
        df[selected_col].hist()
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for visualization")

    # ================= AI INSIGHTS =================
    st.subheader("🤖 AI Insights & Q&A")

    if not groq_key:
        st.warning("⚠️ Please enter your Groq API key in sidebar to enable AI features")
    else:
        os.environ["OPENAI_API_KEY"] = groq_key

        llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            temperature=0
        )

        # -------- AI INSIGHTS --------
        if st.button("✨ Generate AI Insights"):
            with st.spinner("Analyzing dataset..."):
                summary = df.describe().to_string()

                prompt = f"""
You are an expert business consultant.

Dataset Summary:
{summary}

Provide:
1. 3 actionable insights
2. 3 business strategies to increase profit
3. Identify risks
4. Suggest one growth opportunity

Keep it clear and practical.
"""

                try:
                    response = llm.invoke(prompt).content
                    st.success(response)
                except Exception as e:
                    st.error(f"AI Error: {e}")

        # -------- Q&A --------
        st.subheader("❓ Ask Questions")

        user_query = st.text_input("Ask a question about your data")

        if user_query:
            with st.spinner("Thinking..."):
                prompt = f"""
Dataset Summary:
{df.describe().to_string()}

Question:
{user_query}

Give a clear and business-focused answer.
"""

                try:
                    answer = llm.invoke(prompt).content
                    st.info(answer)
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👆 Upload a dataset to start your AI analysis")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built for Hackathon 🚀 | AI + Data + Impact")