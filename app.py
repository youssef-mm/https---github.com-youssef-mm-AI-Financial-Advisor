import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. App UI Settings
# ==========================================
st.set_page_config(page_title="AI Financial Advisor", page_icon="💰", layout="wide")
st.title("💰 Smart AI Financial Advisor")
st.markdown("Welcome! Enter your financial data, and the AI will analyze your budget in seconds.")

# ==========================================
# 2. Sidebar (Inputs)
# ==========================================
with st.sidebar:
    st.header("📊 Budget Data")
    income = st.number_input("Monthly Income (EGP):", min_value=0.0, value=15000.0, step=500.0)
    rent = st.number_input("Rent / Housing:", min_value=0.0, value=4000.0, step=100.0)
    food = st.number_input("Food / Groceries:", min_value=0.0, value=3000.0, step=100.0)
    transport = st.number_input("Transportation:", min_value=0.0, value=1000.0, step=50.0)
    others = st.number_input("Other Expenses:", min_value=0.0, value=2000.0, step=100.0)
    
    st.markdown("---")
    target_goal = st.text_input("Do you have a specific financial goal?", placeholder="e.g., Buy a laptop for 35,000 EGP")
    
    model_choice = st.selectbox(
        "🤖 Select AI Model:",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ],
        index=0,
        help="llama-3.1-8b-instant provides instant responses and is standard on Groq Free Tier."
    )
    
    # API Key Configuration: Check env first, then st.secrets, then sidebar input
    default_key = os.environ.get("GROQ_API_KEY", "")
    if not default_key:
        try:
            default_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            default_key = ""

    if not default_key:
        st.markdown("---")
        groq_api_key_input = st.text_input("🔑 Groq API Key:", type="password", placeholder="gsk_...", help="Get your free key from console.groq.com")
    else:
        groq_api_key_input = default_key

    st.markdown("---")
    run_button = st.button("Analyze Budget 🚀")

# ==========================================
# 3. Basic Calculations & Charts
# ==========================================
total_expenses = rent + food + transport + others
balance = income - total_expenses

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Account Summary")
    st.info(f"💵 **Total Income:** {income:,.0f} EGP")
    st.warning(f"📉 **Total Expenses:** {total_expenses:,.0f} EGP")
    
    if balance > 0:
        st.success(f"💰 **Remaining Balance (Surplus):** {balance:,.0f} EGP")
    else:
        st.error(f"⚠️ **Remaining Balance:** {balance:,.0f} EGP (Deficit!)")

with col2:
    st.subheader("Expenses Breakdown")
    df = pd.DataFrame({
        "Category": ["Housing", "Food", "Transport", "Others"],
        "Amount": [rent, food, transport, others]
    })
    st.dataframe(df, use_container_width=True)

st.markdown("---")

# ==========================================
# 4. AI Engine (Groq & Llama)
# ==========================================
if run_button:
    groq_api_key = groq_api_key_input.strip() if groq_api_key_input else ""
    
    if not groq_api_key:
        st.error("⚠️ Groq API Key required: Please enter your key in the sidebar or add it to Streamlit Secrets.")
        st.stop()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial advisor. 
        Analyze the budget and provide practical, actionable advice in English, structured clearly into:
        1. Financial Situation Assessment (calculate expenses to income ratio).
        2. Savings Plan (how to utilize the surplus).
        3. Smart, specific tips to reduce high expenses.
        4. A clear timeline to achieve the user's financial goal (if mentioned)."""),
        ("human", """
        Income: {income}
        Total Expenses: {total_expenses}
        Balance: {balance}
        Details: Housing {rent}, Food {food}, Transport {transport}, Others {others}.
        Financial Goal: {target_goal}
        """)
    ])

    with st.spinner("🧠 The advisor is analyzing your budget now..."):
        # Order of models to try in case selected model is retired or unavailable
        models_to_try = [model_choice]
        fallback_models = ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]
        for m in fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)

        response = None
        last_error = None
        used_model = model_choice

        for current_model in models_to_try:
            try:
                llm = ChatGroq(
                    model=current_model,
                    api_key=groq_api_key,
                    temperature=0.1
                )
                chain = prompt_template | llm
                response = chain.invoke({
                    "income": income,
                    "total_expenses": total_expenses,
                    "balance": balance,
                    "rent": rent,
                    "food": food,
                    "transport": transport,
                    "others": others,
                    "target_goal": target_goal if target_goal else "No specific goal mentioned."
                })
                used_model = current_model
                break
            except Exception as e:
                last_error = e
                continue

        if response:
            if used_model != model_choice:
                st.info(f"💡 Switched to fallback model `{used_model}` because `{model_choice}` was unavailable on Groq.")
            st.subheader("💡 Financial Advisor Recommendations:")
            st.markdown(response.content)
        else:
            st.error(f"An error occurred while processing the data: {last_error}")
