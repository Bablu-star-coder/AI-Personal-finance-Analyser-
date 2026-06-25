import streamlit as st
import pandas as pd
import numpy as np
import re
import openai

# 1. PAGE CONFIGURATION
st.set_page_config(page_title=" AI Finance Analyser", page_icon="📊", layout="wide")

st.title("🧠 AI Personal Finance Analyser")
st.caption("Track your automated 2026 SIP portfolios, ledger entries, and liquid balances instantly.")

# 2. COMPLETE STATEMENT PROCESSING & PIPELINE ENGINE
def process_and_categorize_statement(df):
    """
    Dynamically maps columns, strips character noise, preserves negative mathematical signs 
    for debits, and enforces strict day-first Indian date parsing.
    """
    # Dynamic Column Matcher
    date_keywords = ['date', 'txn date', 'transaction date', 'value date']
    desc_keywords = ['desc', 'narration', 'particular', 'transaction details', 'remarks', 'description']
    amt_keywords = ['amount', 'amt', 'volume', 'transaction amount', 'credit/debit', 'balance']

    date_col = [c for c in df.columns if any(k in c.lower() for k in date_keywords)]
    desc_col = [c for c in df.columns if any(k in c.lower() for k in desc_keywords)]
    amt_col = [c for c in df.columns if any(k in c.lower() for k in amt_keywords)]

    # Fallback indexes if labels are completely generic
    date_col = date_col[0] if date_col else df.columns[0]
    desc_col = desc_col[0] if desc_col else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    amt_col = amt_col[0] if amt_col else (df.columns[2] if len(df.columns) > 2 else df.columns[-1])

    # Extract required series
    processed_df = df[[date_col, desc_col, amt_col]].copy()
    processed_df.columns = ['Date', 'Description', 'Amount']

    # Strict Day-First Indian Date Conversion (DD-MM-YYYY)
    processed_df['Date'] = pd.to_datetime(processed_df['Date'], dayfirst=True, errors='coerce')
    
    # Capture Debit vs Credit indicators before removing symbols
    processed_df['Amount_Str'] = processed_df['Amount'].astype(str)
    processed_df['Is_Debit'] = processed_df['Amount_Str'].str.contains(r'-|DR|DEBIT', case=False, regex=True)

    # Sanitize and convert numeric values
    processed_df['Amount_Clean'] = processed_df['Amount_Str'].str.replace(r'[₹\$,\s\-]', '', regex=True)
    processed_df['Amount'] = pd.to_numeric(processed_df['Amount_Clean'], errors='coerce').fillna(0.0)

    # Re-apply negative math values strictly to Debits/Expenses
    processed_df['Amount'] = processed_df.apply(
        lambda row: -row['Amount'] if row['Is_Debit'] else row['Amount'], axis=1
    )

    # Clean intermediate tracking flags
    processed_df = processed_df.drop(columns=['Amount_Str', 'Amount_Clean', 'Is_Debit'])

    # Local Rule-Based Keyword Router
    def local_categorize(raw_desc):
        cleaned = str(raw_desc).upper()
        # Instantly clear structural string noise from layout builders
        cleaned = re.sub(r'(MISCELLANEOUS|OTHER EXPENSES)', '', cleaned).strip()
        
        if any(k in cleaned for k in ["ICCW FA", "FAILED TRANCATION", "REFUND"]):
            return "ATM Reversals & Refunds"
        elif any(k in cleaned for k in ["ICCLDHR", "INDIAN CLEARING CORP", "MONEY LIC", "MONEYLICIOUS", "RAISE SECURITIES", "DS AXISCN"]):
            return "Investments & Trading"
        elif any(k in cleaned for k in ["JIO MOBIL", "JIO PREP", "AMAZON", "SMS CHARGES", "NEXTGENFASTFAS"]):
            return "Bills & Utilities"
        elif any(k in cleaned for k in ["BY CASH", "CARDLESS DEPOSIT", "CASH DEPOSITS", "DEPOSIT"]):
            return "Cash Deposits"
        elif "ICCW" in cleaned:
            return "ATM Cash Withdrawals"
        elif any(k in cleaned for k in ["SANJAY K", "NARESH M", "BELA KUM", "BABLU KU", "MIHIR K", "GOURI PR", "RAKESH K", "ASMIT KU"]):
            return "Peer Transfers"
        elif "INT.PD" in cleaned or "INT CARD" in cleaned:
            return "Bank Interest Income"
        elif "BMTC BUS" in cleaned or any(keyword in cleaned for keyword in ["UBER", "OLA", "RAPIDO", "METRO", "TRAIN"]):
            return "Transport & Commute"
        return "Other Spending"

    processed_df['Category'] = processed_df['Description'].apply(local_categorize)
    total_net_volume = processed_df['Amount'].sum()
    
    return processed_df, total_net_volume

# 3. STREAMLIT FILE UPLOADER & INTERFACE
uploaded_file = st.file_uploader("Upload your transaction CSV/Excel Statement data", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read layout formats safely
    if uploaded_file.name.endswith('.csv'):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    # Run processing engine
    clean_df, continuous_calculated_total = process_and_categorize_statement(raw_df)

    # Split dashboard calculations accurately
    total_income = clean_df[clean_df['Amount'] > 0]['Amount'].sum()
    total_expenses = clean_df[clean_df['Amount'] < 0]['Amount'].sum()

    # 4. KPI VALUE LAYOUT DISPLAY
    st.success(f"Successfully optimized and indexed {len(clean_df)} financial transaction rows!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income Tracked (Inflows)", f"₹{total_income:,.2f}")
    col2.metric("Total Expenses (Outflows)", f"₹{abs(total_expenses):,.2f}")
    col3.metric("Net Liquid Account Balance Variance", f"₹{continuous_calculated_total:,.2f}")

    # 5. DATA EXPLORER GRAPHICS
    st.subheader("📊 Expense Distribution by Volume")
    chart_data = clean_df.groupby('Category')['Amount'].sum().reset_index()
    # Force visualization numbers positive for rendering clean bars
    chart_data['Absolute Volume'] = chart_data['Amount'].abs()
    st.bar_chart(data=chart_data, x='Category', y='Absolute Volume', use_container_width=True)

    st.subheader("🔍 Interactive Data Explorer")
    st.dataframe(clean_df, use_container_width=True)

    # 6. GENERATE SCRIPT SUMMARY PACK FOR THE LLM PROMPT
    summary_metrics = clean_df.groupby('Category')['Amount'].agg(['count', 'sum']).to_string()

    # Integrated System Prompt Template
    ai_prompt = f"""
    You are an expert Indian personal finance portfolio manager and accounting auditor reviewing a user's bank statement metrics.
    Analyze the data and structure your response EXACTLY into the specified headings. Do not modify layout tags.

    CRITICAL SYSTEM LAWS:
    1. CURRENCY: Prefix every single monetary balance with the Indian Rupee symbol (₹). Never use dollars ($).
    2. THE INVESTMENT LAW: Outflows under 'Investments & Trading' (negative sum balances) indicate wealth asset formation like Mutual Fund SIPs via ICCL. Do not describe this as a loss—praise it as disciplined capital compounding.
    3. MATHEMATICAL ACCURACY: The precise absolute net change calculated by the ledger engine is strictly ₹{continuous_calculated_total:,.2f}. Frame all text paragraphs completely around this reality.

    ---
    EXPECTED FORMAT STRUCTURE:
    
    ### 📊 Portfolio Data Summary
    (Provide a clean markdown table breaking down category distributions)

    ### 🧠 Luxuryverce AI Analytics Report
    > **Executive Financial Health Note:** (Summary of balance changes)

    #### 1. Strategic Investment & Capital Formation
    (Analysis of SIP wealth creation and broker account transactions)

    #### 2. Cash Flow & Liquidity Management
    (Analysis of ATM, cash entries, and bills stability)

    ### 📈 Actionable Portfolio Recommendations
    - (2-3 targeted portfolio health bullets)
    ---

    Metrics Dataset to Analyze:
    {summary_metrics}
    """

    # 7. EXECUTING DEPLOYED AI REVIEW
    if st.button("Run AI Financial Diagnostics"):
        st.write("### 🧠 AI Personal Finance Analyzer Running...")
        try:
            # Connects directly to your configured dashboard setup
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ai_prompt}]
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI Diagnostics Connection Error: {e}")