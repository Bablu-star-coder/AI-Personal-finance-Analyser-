import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load Environment Variables
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")

if github_token:
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
    )
else:
    st.error("Error: GITHUB_TOKEN not found in your .env file!")

# Dashboard Layout Config (Wide mode is perfect for large data tables)
st.set_page_config(page_title="AI Personal Finance Analyzer", layout="wide", page_icon="📈")
st.title("📈 AI Personal Finance Analyzer")
st.write("Upload large-scale financial statements. Optimized with Pandas vector aggregation and smart AI chunking.")

uploaded_file = st.file_uploader("Upload your Bank Statement (CSV)", type=["csv"])

@st.cache_data
def clean_currency_string(val):
    if pd.isna(val) or str(val).strip() == "":
        return 0.0
    val_str = str(val).replace(" ", "").replace(",", "").replace("?", "")
    is_negative = "-" in val_str
    numeric_parts = "".join(re.findall(r'[0-9.]', val_str))
    if not numeric_parts:
        return 0.0
    try:
        amount = float(numeric_parts)
        return -amount if is_negative else amount
    except ValueError:
        return 0.0

# Rule-based local pre-categorizer to process large rows instantly without hitting AI limits
def local_categorize(desc):
    """
    Rule-based local pre-categorizer with explicit handling for 
    ICCL Mutual Fund SIPs and ICCW ATM Failed Transaction Refunds.
    """
    cleaned_desc = str(desc).replace("Miscellaneous", "").strip().upper()
    
    # 1. ATM Failed Transactions & Reversals (ICCW FA)
    if "ICCW FA" in cleaned_desc or "FAILED TRANSACTION" in cleaned_desc or "REFUND" in cleaned_desc:
        return "ATM Reversals & Refunds"

    # 2. Mutual Fund SIPs via ICCL (Indian Clearing Corporation Ltd)
    elif "ICCLDHR" in cleaned_desc or "INDIAN CLEARING CORP" in cleaned_desc:
        return "Investments & Trading"
        
    # 3. Other Stock Market Accounts
    elif any(keyword in cleaned_desc for keyword in ["MONEY LIC", "MONEYLICIOUS", "RAISE SECURITIES", "DS AXISCN", "Raise
Se"]):
        return "Investments & Trading"
        
    # 4. Mobile Recharges & Utilities
    elif any(keyword in cleaned_desc for keyword in ["JIO MOBIL", "JIO PREP", "AMAZON", "SMS CHARGES", "NEXTGENFASTFAS" , "paytm-jiomobil" , "Flipkart"]):
        return "Bills & Utilities"
        
    # 5. Cash Transactions & Physical Deposits
    elif any(keyword in cleaned_desc for keyword in ["BY CASH", "CARDLESS DEPOSIT", "CASH DEPOSITS"]):
        return "Cash Deposits"
        
    # 6. Standard Cardless ATM Withdrawals (Successful ones)
    elif "ICCW" in cleaned_desc:
        return "ATM Cash Withdrawals"
        
    # 7. Peer-to-Peer Transfers
    elif any(keyword in cleaned_desc for keyword in ["SANJAY K", "NARESH M", "BELA KUM", "BABLU KU", "MIHIR K", "GOURI PR", "RAKESH K" ,"Mr
RAMES","Asmit ku"]):
        return "Peer Transfers"
        
    # 8. Fixed Account Interest Credits
    elif "INT.PD" in cleaned_desc or "INT CARD" in cleaned_desc:
        return "Bank Interest Income"
        
    return "Other Expenses"

if uploaded_file is not None:
    try:
        # 1. READ DATA EFFICIENTLY
        df = pd.read_csv(uploaded_file)
        df.columns = [col.strip() for col in df.columns]
        
        st.success(f"Successfully optimized and indexed {len(df)} financial transaction rows!")
        
        # 2. IDENTIFY COLUMNS
        date_col = [col for col in df.columns if 'date' in col.lower()][0] if [col for col in df.columns if 'date' in col.lower()] else df.columns[0]
        desc_col = [col for col in df.columns if 'desc' in col.lower() or 'narration' in col.lower() or 'particular' in col.lower()][0] if [col for col in df.columns if 'desc' in col.lower() or 'narration' in col.lower() or 'particular' in col.lower()] else df.columns[1]
        
        withdrawal_cols = [col for col in df.columns if 'withdrawal' in col.lower() or 'debit' in col.lower()]
        deposit_cols = [col for col in df.columns if 'deposit' in col.lower() or 'credit' in col.lower()]
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        
        # 3. VECTORIZED DATA CLEANING (Blazing fast for huge files)
        if withdrawal_cols and deposit_cols:
            w_col = withdrawal_cols[0]
            d_col = deposit_cols[0]
            df['Clean_Withdrawal'] = df[w_col].apply(clean_currency_string)
            df['Clean_Deposit'] = df[d_col].apply(clean_currency_string)
            df['Net_Amount'] = df['Clean_Deposit'] - df['Clean_Withdrawal'].abs()
        elif amount_cols:
            df['Net_Amount'] = df[amount_cols[0]].apply(clean_currency_string)
        else:
            df['Net_Amount'] = df[df.columns[2]].apply(clean_currency_string)

        # Calculate or extract Running Balance
        balance_cols = [col for col in df.columns if 'balance' in col.lower()]
        if balance_cols:
            df['Calculated_Balance'] = df[balance_cols[0]].apply(clean_currency_string)
        else:
            df['Calculated_Balance'] = df['Net_Amount'].cumsum()

        # Handle Timestamps
        df['Parsed_Date'] = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
        df_clean_dates = df.dropna(subset=['Parsed_Date']).sort_values('Parsed_Date')

        # 4. FAST LOCAL CATEGORIZATION ENGINE
        df_clean_dates['Category'] = df_clean_dates[desc_col].apply(local_categorize)

        # 5. HIGH-LEVEL KPI METRICS
        total_income = df_clean_dates[df_clean_dates['Net_Amount'] > 0]['Net_Amount'].sum()
        total_expense = df_clean_dates[df_clean_dates['Net_Amount'] < 0]['Net_Amount'].sum()
        
        last_month_spending = 0.0
        last_month_name = "N/A"
        
        if not df_clean_dates.empty:
            latest_date = df_clean_dates['Parsed_Date'].max()
            df_clean_dates['YearMonth'] = df_clean_dates['Parsed_Date'].dt.to_period('M')
            latest_month_period = latest_date.to_period('M')
            latest_month_data = df_clean_dates[df_clean_dates['YearMonth'] == latest_month_period]
            last_month_spending = latest_month_data[latest_month_data['Net_Amount'] < 0]['Net_Amount'].sum()
            last_month_name = latest_date.strftime('%B %Y')

        # Display Metrics Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Income Tracked", f"₹{total_income:,.2f}")
        col2.metric("Total Overall Expenses", f"₹{abs(total_expense):,.2f}")
        col3.metric(f"Spending in {last_month_name}", f"₹{abs(last_month_spending):,.2f}")
        latest_balance = df_clean_dates['Calculated_Balance'].iloc[-1] if not df_clean_dates.empty else 0.0
        col4.metric("Final Statement Balance", f"₹{latest_balance:,.2f}")
        
        st.markdown("---")
        
        # --- GRAPH LAYOUTS FOR BIG DATA ---
        chart_col, table_col = st.columns([1, 1])
        
        with chart_col:
            st.subheader("📊 Expense Distribution by Volume")
            # Group rows locally first before making charts to prevent browser lag
            expense_summary = df_clean_dates[df_clean_dates['Net_Amount'] < 0].groupby('Category')['Net_Amount'].sum().abs().reset_index()
            if not expense_summary.empty:
                fig_bar = px.bar(expense_summary, x='Category', y='Net_Amount', color='Category',
                                 text_auto='.2s', title="Total Spent per Category",
                                 color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No expense data found to plot.")
                
        with table_col:
            st.subheader("🔍 Interactive Data Explorer")
            # Added a quick dropdown filter so you can sift through huge files easily
            selected_cat = st.selectbox("Filter view by Category:", ["All Categories"] + list(df_clean_dates['Category'].unique()))
            
            filtered_df = df_clean_dates
            if selected_cat != "All Categories":
                filtered_df = df_clean_dates[df_clean_dates['Category'] == selected_cat]
                
            st.dataframe(filtered_df[[date_col, desc_col, 'Category', 'Net_Amount', 'Calculated_Balance']], use_container_width=True, height=260)
            
        st.markdown("---")
        
        # --- BIG DATA SMART AI INTERFACE ---
        st.subheader("🧠 AI Personal Finance Analyzer")
        if st.button("🚀 Run AI Analysis on Data Summary"):
            with st.spinner("AI is evaluating financial trends from compressed data summaries..."):
                try:
                    # SMART CHUNKING: Group data by category and calculate counts and sums
                    # This shrinks 10,000 lines into just 5 clean lines for the AI prompt!
                    ai_summary = df_clean_dates.groupby('Category').agg(
                        Transaction_Count=('Net_Amount', 'count'),
                        Total_Net_Volume=('Net_Amount', 'sum')
                    ).reset_index().to_string(index=False)

                    prompt = f"""
You are an expert Indian financial analytics assistant.
Analyze the following summarized banking transaction dataset.

CRITICAL INSTRUCTIONS FOR CURRENCY FORMATTING:
- NEVER use the dollar sign ($) anywhere in your analysis.
- Keep the exact raw numeric values from the data but swap the symbol. For example, if you calculate 4,200, display it strictly as \u20b94,200 (DO NOT convert \u20b94,200 into $50 USD).
- Every single monetary transaction value, budget cap, or total breakdown figure MUST be prefixed with the Indian Rupee symbol (\u20b9).

Data Summary:
{ai_summary}
"""
                    
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a strategic corporate financial advisor."},
                            {"role": "user", "content": prompt}
                        ],
                        model="gpt-4o-mini"
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.choices[0].message.content)
                    
                except Exception as ai_err:
                    st.error(f"AI Matrix Exception: {ai_err}")
                    
    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.info("💡 Please upload your CSV file to test true database auto-categorization.")
