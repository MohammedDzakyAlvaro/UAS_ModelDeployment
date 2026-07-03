"""
app.py
Streamlit deployment untuk Credit Score Classification.
Menampilkan form input nasabah dan memprediksi Credit Score (Poor/Standard/Good).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from inference import CreditScoreInference


# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return CreditScoreInference(
        model_path  = os.path.join(BASE_DIR, "artifacts/best_model.pkl"),
        config_path = os.path.join(BASE_DIR, "artifacts/preprocess_config.pkl"),
    )

engine = load_engine()
config = engine.config


# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
    .subtitle     { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .result-good  { background: #d4edda; border-left: 5px solid #28a745;
                    padding: 1rem 1.5rem; border-radius: 8px; }
    .result-std   { background: #fff3cd; border-left: 5px solid #ffc107;
                    padding: 1rem 1.5rem; border-radius: 8px; }
    .result-poor  { background: #f8d7da; border-left: 5px solid #dc3545;
                    padding: 1rem 1.5rem; border-radius: 8px; }
    .result-title { font-size: 1.5rem; font-weight: 700; margin: 0; }
    .result-desc  { font-size: 0.95rem; margin-top: 0.3rem; }
    .section-hdr  { font-size: 1.05rem; font-weight: 600;
                    color: #2c3e50; margin: 1.2rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Credit Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan data nasabah pada panel kiri untuk memprediksi Credit Score (Poor / Standard / Good)</div>', unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────
# Sidebar — Input Form
# ─────────────────────────────────────────────
st.sidebar.header("Data Nasabah")

OCCUPATION_OPTIONS      = ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
                            'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
                            'Accountant', 'Architect', 'Mechanic', 'Musician', 'Writer']
CREDIT_MIX_OPTIONS      = ['Good', 'Standard', 'Bad']
PAYMENT_MIN_OPTIONS     = ['Yes', 'No']
PAY_BEHAVIOUR_OPTIONS   = ['High_spent_Small_value_payments', 'High_spent_Medium_value_payments',
                            'High_spent_Large_value_payments', 'Low_spent_Small_value_payments',
                            'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments']
MONTH_OPTIONS           = ['January','February','March','April','May','June',
                            'July','August','September','October','November','December']
LOAN_TYPE_OPTIONS       = ['Auto Loan', 'Personal Loan', 'Student Loan', 'Mortgage Loan',
                            'Payday Loan', 'Home Equity Loan', 'Not Specified']

# ── Personal Info ──────────────────────────────────
st.sidebar.markdown("**Informasi Pribadi**")
age        = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
month      = st.sidebar.selectbox("Month", MONTH_OPTIONS, index=0)
occupation = st.sidebar.selectbox("Occupation", OCCUPATION_OPTIONS, index=2)

# ── Income & Balance ───────────────────────────────
st.sidebar.markdown("**Pendapatan & Saldo**")
annual_income            = st.sidebar.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
monthly_inhand_salary    = st.sidebar.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=4000.0, step=100.0)
monthly_balance          = st.sidebar.number_input("Monthly Balance ($)", min_value=0.0, value=500.0, step=50.0)
amount_invested_monthly  = st.sidebar.number_input("Amount Invested Monthly ($)", min_value=0.0, value=200.0, step=50.0)

# ── Credit Accounts ────────────────────────────────
st.sidebar.markdown("**Rekening & Kartu Kredit**")
num_bank_accounts = st.sidebar.slider("Num Bank Accounts", 0, 20, 3)
num_credit_card   = st.sidebar.slider("Num Credit Card", 0, 20, 3)
num_of_loan       = st.sidebar.slider("Num of Loan", 0, 20, 1)
type_of_loan      = st.sidebar.selectbox("Type of Loan", LOAN_TYPE_OPTIONS, index=0)
total_emi         = st.sidebar.number_input("Total EMI per Month ($)", min_value=0.0, value=200.0, step=10.0)

# ── Credit Behaviour ───────────────────────────────
st.sidebar.markdown("**Perilaku Kredit**")
interest_rate            = st.sidebar.slider("Interest Rate (%)", 1, 100, 12)
delay_from_due           = st.sidebar.slider("Delay from Due Date (days)", 0, 100, 5)
num_delayed_payment      = st.sidebar.slider("Num of Delayed Payment", 0, 30, 2)
changed_credit_limit     = st.sidebar.slider("Changed Credit Limit", -20.0, 30.0, 5.0, step=0.5)
num_credit_inquiries     = st.sidebar.slider("Num Credit Inquiries", 0, 20, 3)
outstanding_debt         = st.sidebar.number_input("Outstanding Debt ($)", min_value=0.0, value=500.0, step=50.0)
credit_utilization_ratio = st.sidebar.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0, step=0.5)

# ── Credit History ─────────────────────────────────
st.sidebar.markdown("**Riwayat Kredit**")
credit_history_years  = st.sidebar.slider("Credit History — Years", 0, 40, 5)
credit_history_months = st.sidebar.slider("Credit History — Months (0-11)", 0, 11, 0)
credit_history_age    = credit_history_years * 12 + credit_history_months   # numeric (months)

credit_mix            = st.sidebar.selectbox("Credit Mix", CREDIT_MIX_OPTIONS, index=0)
payment_of_min_amount = st.sidebar.selectbox("Payment of Min Amount", PAYMENT_MIN_OPTIONS, index=0)
payment_behaviour     = st.sidebar.selectbox("Payment Behaviour", PAY_BEHAVIOUR_OPTIONS, index=3)


# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
predict_btn = st.sidebar.button("Predict Credit Score", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────
col_main, col_info = st.columns([2, 1])

with col_main:
    if predict_btn:
        input_data = {
            'Age':                      age,
            'Annual_Income':            annual_income,
            'Monthly_Inhand_Salary':    monthly_inhand_salary,
            'Num_Bank_Accounts':        num_bank_accounts,
            'Num_Credit_Card':          num_credit_card,
            'Interest_Rate':            interest_rate,
            'Num_of_Loan':              num_of_loan,
            'Delay_from_due_date':      delay_from_due,
            'Num_of_Delayed_Payment':   num_delayed_payment,
            'Changed_Credit_Limit':     changed_credit_limit,
            'Num_Credit_Inquiries':     num_credit_inquiries,
            'Credit_Mix':               credit_mix,
            'Outstanding_Debt':         outstanding_debt,
            'Credit_Utilization_Ratio': credit_utilization_ratio,
            'Credit_History_Age':       credit_history_age,      # already in months
            'Payment_of_Min_Amount':    payment_of_min_amount,
            'Total_EMI_per_month':      total_emi,
            'Amount_invested_monthly':  amount_invested_monthly,
            'Payment_Behaviour':        payment_behaviour,
            'Monthly_Balance':          monthly_balance,
            'Month':                    month,
            'Occupation':               occupation,
            'Type_of_Loan':             type_of_loan,
        }

        with st.spinner("Memprediksi credit score..."):
            result = engine.predict(input_data)

        label = result['label']

        # Display result
        if label == 'Good':
            css_cls  = "result-good"
            desc     = "Nasabah memiliki riwayat kredit yang sangat baik. Risiko gagal bayar rendah."
        elif label == 'Standard':
            css_cls  = "result-std"
            desc     = "Nasabah memiliki kredit yang cukup baik namun ada beberapa area yang perlu diperbaiki."
        else:
            css_cls  = "result-poor"
            desc     = "Nasabah memiliki riwayat kredit yang buruk. Risiko gagal bayar tinggi."

        st.markdown(f"""
        <div class="{css_cls}">
            <p class="result-title">Credit Score: <strong>{label}</strong></p>
            <p class="result-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar chart
        if 'probabilities' in result:
            st.markdown("#### Probabilitas per Kelas")
            proba_df = pd.DataFrame(
                list(result['probabilities'].items()),
                columns=['Credit Score', 'Probability']
            ).sort_values('Probability', ascending=False)
            st.bar_chart(proba_df.set_index('Credit Score'))

        # Input summary
        st.markdown("#### Ringkasan Input")
        display_input = {k: v for k, v in input_data.items()}
        display_input['Credit_History_Age'] = f"{credit_history_years}Y {credit_history_months}M ({credit_history_age} months)"
        st.dataframe(pd.DataFrame([display_input]).T.rename(columns={0: 'Value'}), use_container_width=True)

    else:
        st.info("Isi data nasabah di panel kiri, lalu klik **Predict Credit Score**.")

        


with col_info:

    st.markdown("### Deskripsi Label")
    st.markdown("""
    | Label | Deskripsi |
    |-------|-----------|
    | **Good** | Nasabah dengan riwayat kredit sangat baik |
    | **Standard** | Nasabah dengan kredit cukup baik |
    | **Poor** | Nasabah dengan riwayat kredit buruk |
    """)

