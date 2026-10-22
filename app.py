
import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="CreditWise | Loan Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL CSS
# ==================================================
st.markdown("""
<style>
/* Main application */
.stApp {
    background-color: #f3f6fb;
    color: #172b4d;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* All normal text */
p, label, li, h1, h2, h3, h4 {
    color: #172b4d;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 36px;
    font-weight: 750;
    margin: 0;
}

.hero p {
    color: #e2edf5 !important;
    font-size: 16px;
    margin-top: 10px;
    margin-bottom: 0;
}

/* Section headings */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #102a43 !important;
    margin: 10px 0 5px 0;
}

.section-subtitle {
    color: #526780 !important;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Input labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #172b4d !important;
    font-weight: 600 !important;
}

/* Text and number inputs */
div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #172b4d !important;
    -webkit-text-fill-color: #172b4d !important;
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] input,
div[data-baseweb="select"] div,
div[data-baseweb="select"] span {
    color: #172b4d !important;
}

/* Dropdown popup options */
div[role="listbox"],
div[role="option"] {
    background-color: #ffffff !important;
    color: #172b4d !important;
}

div[role="option"]:hover {
    background-color: #e5f3f5 !important;
}

/* Form cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    border: 1px solid #dbe4ee !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dbe4ee;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #172b4d !important;
}

/* Main buttons */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 52px;
    background: #087e8b !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

div[data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #075e68 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eaf0f7;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #172b4d !important;
}

/* Expanders and captions */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dbe4ee;
    border-radius: 10px;
}

[data-testid="stCaptionContainer"] p {
    color: #526780 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #627d98;
    font-size: 13px;
    padding: 15px 0;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================
@st.cache_resource
def load_model():
    saved = joblib.load("creditwise_model.pkl")
    return saved["model"], saved["label_encoder"]

try:
    model, label_encoder = load_model()
except Exception as error:
    st.error("Could not load the trained model.")
    st.info(
        "Check that creditwise_model.pkl is in the same folder "
        "as app.py and contains the model and label_encoder."
    )
    st.code(str(error))
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("# 💳 CreditWise")
    st.caption("LOAN INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("### 🏠 Loan Prediction")
    st.write("Enter applicant information to predict the loan status.")

    st.divider()
    st.markdown("### ⚙️ Model Information")
    st.write("**Algorithm:** XGBoost")
    st.write("**Task:** Binary Classification")
    st.write("**Output:** Approved / Rejected")

    st.divider()
    st.caption(
        "Educational machine learning project. "
        "Predictions are estimates, not actual lending decisions."
    )

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <h1>CreditWise Loan Prediction</h1>
    <p>
        Intelligent loan assessment powered by Machine Learning.
        Enter applicant details to generate a prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# APPLICANT FORM
# ==================================================
with st.form("creditwise_application"):

    # Applicant information
    st.markdown(
        '<div class="section-title">👤 Applicant Information</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Basic personal and employment information'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            no_of_dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=20,
                value=0
            )

            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"]
            )

        with col2:
            self_employed = st.selectbox(
                "Self Employed",
                ["Yes", "No"]
            )

            cibil_score = st.number_input(
                "CIBIL Score",
                min_value=300,
                max_value=900,
                value=700
            )

    st.write("")

    # Financial information
    st.markdown(
        '<div class="section-title">💰 Financial Information</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Income, loan amount and repayment term'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            income_annum = st.number_input(
                "Annual Income",
                min_value=0,
                value=500000,
                step=50000
            )

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0,
                value=1000000,
                step=50000
            )

        with col2:
            loan_term = st.number_input(
                "Loan Term",
                min_value=1,
                value=10
            )

            st.caption(
                "Use the same currency and units as your training dataset."
            )

    st.write("")

    # Asset information
    st.markdown(
        '<div class="section-title">🏠 Asset Information</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Applicant property and bank asset values'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            residential_assets_value = st.number_input(
                "Residential Assets Value",
                min_value=0,
                value=0,
                step=50000
            )

            commercial_assets_value = st.number_input(
                "Commercial Assets Value",
                min_value=0,
                value=0,
                step=50000
            )

        with col2:
            luxury_assets_value = st.number_input(
                "Luxury Assets Value",
                min_value=0,
                value=0,
                step=50000
            )

            bank_asset_value = st.number_input(
                "Bank Asset Value",
                min_value=0,
                value=0,
                step=50000
            )

    st.write("")
    submitted = st.form_submit_button(
        "🔍  Predict Loan Status",
        use_container_width=True
    )

# ==================================================
# PREDICTION RESULT
# ==================================================
if submitted:

    input_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    try:
        prediction = model.predict(input_data)
        result = label_encoder.inverse_transform(prediction)[0]

        st.divider()
        st.markdown(
            '<div class="section-title">📊 Assessment Result</div>',
            unsafe_allow_html=True
        )

        if result == "Approved":
            st.success("✅ Predicted Loan Status: APPROVED")
            st.write(
                "The model predicts that this applicant belongs "
                "to the approved loan class."
            )
        else:
            st.error("❌ Predicted Loan Status: REJECTED")
            st.write(
                "The model predicts that this applicant belongs "
                "to the rejected loan class."
            )

        with st.expander("📄 View Applicant Details"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This is a model prediction, not a guaranteed loan decision. "
            "Actual lenders may use additional criteria."
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.code(str(error))

# ==================================================
# FOOTER
# ==================================================
st.divider()

st.markdown("""
<div class="footer">
    CreditWise Loan Prediction System<br>
    Built with Python · Streamlit · Scikit-learn · XGBoost
</div>
""", unsafe_allow_html=True)
