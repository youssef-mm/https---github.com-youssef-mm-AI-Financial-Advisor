import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

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
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("⚠️ Internal Error: Groq API Key not found in the system.")
        st.stop()
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.1
    )
    
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
    
    chain = prompt_template | llm

    with st.spinner("🧠 The advisor is analyzing your budget now (in seconds)..."):
        try:
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
            
            st.subheader("💡 Financial Advisor Recommendations:")
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"An error occurred while processing the data: {e}")
