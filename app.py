'''streamlit run app.py'''
import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
if os.path.exists("preprocessor.pkl") and os.path.exists("model.pkl"):
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
else:
    st.error("⚠️ Model files missing! Please ensure 'preprocessor.pkl' and 'model.pkl' are in the working directory.")
    st.stop()

# ---------------- ADVANCED CSS & ANIMATIONS ----------------
st.markdown(
"""
<style>
/* Modern Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background Canvas */
.stApp {
    background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, #0f172a 50%, #020617 100%);
    background-attachment: fixed;
}

/* Main Container Padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Title Styling with Animated Shimmer Gradient */
.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #e0e7ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite, fadeInDown 1.2s ease-out;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 35px;
    letter-spacing: 0.5px;
    animation: fadeIn 1.5s ease-out;
}

/* Glassmorphism Cards */
.card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    color: #f8fafc;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: fadeInUp 0.8s ease-out;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 30px 60px rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.4);
}

.card h2 {
    color: #ffffff;
    font-size: 22px;
    margin-bottom: 8px;
}

/* Prediction Cards Customization */
.result-card-danger {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.25) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    animation: pulseDanger 2s infinite alternate;
}

.result-card-success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    animation: pulseSuccess 2s infinite alternate;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 10px;
}

/* Glowing Neon Button */
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%);
    color: #ffffff;
    border: none;
    height: 60px;
    font-size: 20px;
    font-weight: 700;
    border-radius: 16px;
    cursor: pointer;
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4);
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.6);
    background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
}

.stButton button:active {
    transform: translateY(1px);
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.7);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Custom Progress Bar Glow */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    border-radius: 10px;
}

/* Animations Keyframes */
@keyframes shine {
    to { background-position: 200% center; }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulseDanger {
    0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.2); }
    100% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.5); }
}

@keyframes pulseSuccess {
    0% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
    100% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.5); }
}
</style>
""",
unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown(
"""
<div class="main-title">
🤖 Customer Churn Intelligence
</div>
<div class="subtitle">
Predict churn risks accurately using high-performance Machine Learning models
</div>
""",
unsafe_allow_html=True
)

# ---------------- DASHBOARD CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
    """
    <div class="card">
        <h2>👤 Profile</h2>
        <p style="color: #94a3b8; font-size: 14px;">Configure customer demographic parameters and behavior data in the sidebar.</p>
    </div>
    """,
    unsafe_allow_html=True)

with c2:
    st.markdown(
    """
    <div class="card">
        <h2>🧠 Intelligence Engine</h2>
        <p style="color: #94a3b8; font-size: 14px;">Random Forest Model pre-trained for multi-variate churn probability classification.</p>
    </div>
    """,
    unsafe_allow_html=True)

with c3:
    st.markdown(
    """
    <div class="card">
        <h2>📊 Live Analytics</h2>
        <p style="color: #94a3b8; font-size: 14px;">Instant retention risk assessment complete with dynamic confidence scoring.</p>
    </div>
    """,
    unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Parameters")

age = st.sidebar.slider("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.number_input("Tenure (Months)", 0, 100, 12)
usage_frequency = st.sidebar.number_input("Usage Frequency", 0, 100, 15)
support_calls = st.sidebar.number_input("Support Calls", 0, 50, 2)
payment_delay = st.sidebar.number_input("Payment Delay (Days)", 0, 100, 5)
subscription_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
total_spend = st.sidebar.number_input("Total Spend ($)", 0, 10000, 500)
last_interaction = st.sidebar.number_input("Last Interaction (Days Ago)", 0, 100, 10)

# ---------------- PREDICTION SECTION ----------------
if st.button("🚀 Run Churn Prediction Analysis"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    try:
        transformed = preprocessor.transform(input_data)
        prediction = model.predict(transformed)
        probability = model.predict_proba(transformed)[0][1]

        st.write("")

        if prediction[0] == 1:
            st.markdown(
            f"""
            <div class="result-card-danger">
                <div class="result-title" style="color: #fca5a5;">
                    ⚠️ High Risk: Customer Likely to Churn
                </div>
                <p style="font-size: 20px; color: #f8fafc; margin-top: 10px;">
                    Predicted Churn Probability: <b>{probability:.2%}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
            )
        else:
            st.markdown(
            f"""
            <div class="result-card-success">
                <div class="result-title" style="color: #6ee7b7;">
                    ✅ Low Risk: Customer Retained
                </div>
                <p style="font-size: 20px; color: #f8fafc; margin-top: 10px;">
                    Predicted Churn Probability: <b>{probability:.2%}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
            )

        st.write("")
        st.markdown("<p style='color: #cbd5e1; font-weight: 600;'>Churn Risk Score Bar:</p>", unsafe_allow_html=True)
        st.progress(float(probability))

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Footer
st.markdown(
"""
<br><hr style="border: 0; height: 1px; background: rgba(255,255,255,0.1);"><br>
<center style="color: #64748b; font-size: 14px;">
    Crafted with ❤️ using Python | Scikit-Learn | Streamlit
</center>
""",
unsafe_allow_html=True
)