import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AutoMA - AI Decision Assistant",
    layout="wide"
)

# ================= HEADER =================
st.title("🚀 AutoMA - AI Decision Assistant")
st.markdown("### Empowering Small Businesses with AI-driven Decisions (SDG 8 & 9)")

# ================= API KEY =================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API key not found. Set OPENAI_API_KEY in environment.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

# ================= CLEAN FUNCTION =================
def clean_text(text):
    return str(text).replace('\xa0', ' ').strip()

# ================= MAIN =================
if uploaded_file:

    # LOAD DATA
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)

        st.success("✅ Dataset loaded successfully")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # ================= TABS =================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🤖 AI Decision Engine",
        "🔮 Simulation",
        "💬 Ask AI"
    ])

    # ================= TAB 1: DASHBOARD =================
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Key Metrics")

        col1, col2, col3 = st.columns(3)

        if 'Sales' in df.columns:
            col1.metric("Total Sales", f"{df['Sales'].sum():,.0f}")

        if 'Profit' in df.columns:
            col2.metric("Total Profit", f"{df['Profit'].sum():,.0f}")

        if 'Quantity' in df.columns:
            col3.metric("Total Quantity", f"{df['Quantity'].sum():,.0f}")

        # BUSINESS INSIGHTS
        st.subheader("Quick Insights")

        insights = []

        try:
            if 'Product' in df.columns and 'Sales' in df.columns:
                top_product = df.groupby('Product')['Sales'].sum().idxmax()
                insights.append(f"🏆 Top Product: {top_product}")

            if 'Region' in df.columns and 'Sales' in df.columns:
                best_region = df.groupby('Region')['Sales'].sum().idxmax()
                insights.append(f"🌍 Best Region: {best_region}")

            if 'Profit' in df.columns:
                low_profit = df[df['Profit'] < df['Profit'].mean()]
                insights.append(f"⚠️ {len(low_profit)} records below average profit")

        except:
            insights.append("Unable to compute insights")

        for i in insights:
            st.write(i)

        # ALERTS
        st.subheader("Alerts")

        try:
            if 'Profit' in df.columns and df['Profit'].mean() < 1000:
                st.error("Low profitability detected")

            if 'Sales' in df.columns and df['Sales'].max() > 5 * df['Sales'].mean():
                st.warning("Sales anomaly detected")

        except:
            st.info("No alerts")

        # VISUALIZATION
        st.subheader("Visualization")

        numeric_cols = df.select_dtypes(include='number').columns

        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column", numeric_cols)

            fig, ax = plt.subplots()
            df[selected_col].hist()
            ax.set_title(f"{selected_col} Distribution")
            st.pyplot(fig)

    # ================= LLM SETUP =================
    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        temperature=0
    )

    # ================= TAB 2: AI DECISION ENGINE =================
    with tab2:
        st.subheader("🤖 AI Decision Engine")

        st.info("Designed for small businesses & local vendors")

        if st.button("Generate AI Decisions"):
            with st.spinner("Analyzing business data..."):

                summary = df.describe().to_string()

                prompt = f"""
You are an expert business consultant helping small businesses.

Dataset Summary:
{summary}

Provide:
1. Key Insights
2. Business Risks
3. Growth Opportunities
4. STRATEGIC DECISIONS (clear actions to take)
5. Priority Action Plan (step-by-step)
"""

                try:
                    response = llm.invoke(prompt).content
                    response = response.encode('utf-8', 'ignore').decode('utf-8')

                    st.success("AI Decision Output:")
                    st.write(response)

                except Exception as e:
                    st.error(f"AI Error: {e}")

    # ================= TAB 3: SIMULATION =================
    with tab3:
        st.subheader("🔮 What-If Simulation")

        if 'Sales' in df.columns:
            change = st.slider("Change Sales (%)", -50, 50, 0)

            simulated_sales = df['Sales'] * (1 + change / 100)

            st.metric("Simulated Sales", f"{simulated_sales.sum():,.0f}")

            if 'Profit' in df.columns:
                simulated_profit = df['Profit'] * (1 + change / 100)
                st.metric("Simulated Profit", f"{simulated_profit.sum():,.0f}")

        else:
            st.warning("Sales column required for simulation")

    # ================= TAB 4: Q&A =================
    with tab4:
        st.subheader("💬 Ask AI")

        user_query = st.text_input("Ask a business question")

        if user_query:
            with st.spinner("Processing..."):

                prompt = f"""
Dataset Summary:
{df.describe().to_string()}

Question:
{user_query}

Answer clearly for a non-technical business owner.
"""

                try:
                    answer = llm.invoke(prompt).content
                    answer = answer.encode('utf-8', 'ignore').decode('utf-8')

                    st.info(answer)

                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("📂 Upload a dataset to begin analysis")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built by Kushal Mahesh Handigund | AI for Social Good 🚀")
