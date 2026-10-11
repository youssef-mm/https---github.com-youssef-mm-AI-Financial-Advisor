import streamlit as st
import pandas as pd
import os
import groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. App UI Settings
# ==========================================
st.set_page_config(page_title="AI Financial Advisor", page_icon="💰", layout="wide")
st.title("💰 Smart AI Financial Advisor")
st.markdown("Welcome! Enter your financial data, and the AI will analyze your budget in seconds.")

# ==========================================
# Helper: Fetch Live Models from Groq API
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_groq_models(api_key: str):
    """Fetch active chat models directly from Groq using the provided API key."""
    if not api_key:
        return []
    try:
        client = groq.Groq(api_key=api_key)
        models_data = client.models.list().data
        
        # Filter for text chat models (exclude whisper audio, embeddings, guardrails, etc.)
        chat_models = []
        excluded_keywords = ["whisper", "guard", "vision", "tool-use-preview"]
        
        for m in models_data:
            mid = m.id.lower()
            if not any(k in mid for k in excluded_keywords):
                chat_models.append(m.id)
                
        # Sort so high-performing models (Llama 3.3, Llama 3.1, GPT-OSS) appear first
        def model_sort_key(name):
            n = name.lower()
            if "llama-3.3" in n:
                return 0
            if "llama-3.1-8b-instant" in n:
                return 1
            if "llama-3.1" in n:
                return 2
            if "gpt-oss" in n:
                return 3
            if "qwen" in n:
                return 4
            return 5

        chat_models.sort(key=model_sort_key)
        return chat_models
    except Exception:
        return []

# ==========================================
# 2. Sidebar (Inputs & API Configuration)
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
    
    st.markdown("---")
    st.header("⚙️ AI Configuration")
    
    # 1. API Key Resolution: check env, then secrets, then manual input
    detected_key = os.environ.get("GROQ_API_KEY", "")
    if not detected_key:
        try:
            detected_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            detected_key = ""

    if detected_key:
        groq_api_key = detected_key.strip()
        st.success("✅ Groq API Key loaded from Secrets/Environment")
    else:
        groq_api_key_input = st.text_input(
            "🔑 Groq API Key:",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at https://console.groq.com/keys"
        )
        groq_api_key = groq_api_key_input.strip() if groq_api_key_input else ""

    # 2. Dynamic Model Choices
    live_models = fetch_groq_models(groq_api_key) if groq_api_key else []
    
    default_fallback_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ]
    
    available_model_options = live_models if live_models else default_fallback_models
    
    model_choice = st.selectbox(
        "🤖 Select AI Model:",
        options=available_model_options,
        index=0,
        help="Models are automatically fetched from your active Groq account when API Key is provided."
    )
    
    st.markdown("---")
    run_button = st.button("Analyze Budget 🚀", use_container_width=True)

# ==========================================
# 3. Basic Calculations & Charts
# ==========================================
total_expenses = rent + food + transport + others
balance = income - total_expenses
expense_ratio = (total_expenses / income) if income > 0 else 0

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Account Summary")
    st.info(f"💵 **Total Income:** {income:,.0f} EGP")
    st.warning(f"📉 **Total Expenses:** {total_expenses:,.0f} EGP")
    st.info(f"📊 **Expense-to-Income Ratio:** {expense_ratio:.1%}")
    
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
# 4. AI Engine (Groq & LangChain)
# ==========================================
if run_button:
    if not groq_api_key:
        st.error("⚠️ Groq API Key required: Please enter your API key in the sidebar or add `GROQ_API_KEY` to Streamlit Secrets.")
        st.stop()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial advisor. 
        Analyze the user's budget and provide practical, actionable advice in English, structured clearly into:
        1. Financial Situation Assessment (analyze the expenses to income ratio).
        2. Savings Plan (how to utilize the surplus effectively).
        3. Smart, specific tips to optimize and reduce high expenses.
        4. A clear timeline to achieve the user's financial goal (if mentioned)."""),
        ("human", """
        Income: {income:,.0f} EGP
        Total Expenses: {total_expenses:,.0f} EGP
        Expense Ratio: {expense_ratio:.1%}
        Balance: {balance:,.0f} EGP
        Details: Housing {rent:,.0f} EGP, Food {food:,.0f} EGP, Transport {transport:,.0f} EGP, Others {others:,.0f} EGP.
        Financial Goal: {target_goal}
        """)
    ])

    with st.spinner("🧠 The advisor is analyzing your budget now..."):
        # Build candidate models starting with user choice, followed by live or fallback models
        models_to_try = [model_choice]
        for m in available_model_options:
            if m not in models_to_try:
                models_to_try.append(m)
        
        response = None
        succeeded_model = None
        attempt_errors = []

        for candidate in models_to_try:
            try:
                llm = ChatGroq(
                    model=candidate,
                    api_key=groq_api_key,
                    temperature=0.2
                )
                chain = prompt_template | llm
                response = chain.invoke({
                    "income": income,
                    "total_expenses": total_expenses,
                    "expense_ratio": expense_ratio,
                    "balance": balance,
                    "rent": rent,
                    "food": food,
                    "transport": transport,
                    "others": others,
                    "target_goal": target_goal if target_goal else "No specific goal mentioned."
                })
                succeeded_model = candidate
                break
            except Exception as e:
                err_msg = str(e)
                attempt_errors.append((candidate, err_msg))
                # If authentication failed, no model will work; abort immediately
                if "invalid_api_key" in err_msg.lower() or "401" in err_msg:
                    st.error("❌ Invalid Groq API Key! Please verify your key at https://console.groq.com/keys")
                    st.stop()
                continue

        if response:
            if succeeded_model != model_choice:
                st.warning(f"⚠️ Selected model `{model_choice}` was unavailable on Groq. Automatically switched to `{succeeded_model}`.")
            st.subheader(f"💡 Financial Advisor Recommendations (via {succeeded_model}):")
            st.markdown(response.content)
        else:
            st.error("❌ An error occurred while communicating with Groq API.")
            with st.expander("🔍 View Technical Details & Model Errors"):
                for m_name, err in attempt_errors:
                    st.markdown(f"**Model `{m_name}`:** `{err}`")
                st.info("💡 Tip: Check https://console.groq.com/docs/models to confirm available models for your account.")
