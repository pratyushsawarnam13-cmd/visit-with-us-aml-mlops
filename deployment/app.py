import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    layout="centered"
)

st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

model_path = hf_hub_download(
    repo_id="Pratyush130194/visit-with-us-tourism-model",
    filename="tourism_model.pkl"
)

model = joblib.load(model_path)

st.sidebar.header("Customer Information")

def user_input_features():
    data = {
        "Age": st.sidebar.slider("Age", 18, 70, 35),
        "TypeofContact": st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"]),
        "CityTier": st.sidebar.selectbox("City Tier", [1, 2, 3]),
        "DurationOfPitch": st.sidebar.slider("Duration Of Pitch", 1, 60, 15),
        "Occupation": st.sidebar.selectbox(
            "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        ),
        "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "NumberOfPersonVisiting": st.sidebar.slider("Number Of Person Visiting", 1, 5, 2),
        "NumberOfFollowups": st.sidebar.slider("Number Of Followups", 0, 10, 2),
        "ProductPitched": st.sidebar.selectbox(
            "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"]
        ),
        "PreferredPropertyStar": st.sidebar.selectbox("Preferred Property Star", [3, 4, 5]),
        "MaritalStatus": st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        "NumberOfTrips": st.sidebar.slider("Number Of Trips", 0, 10, 2),
        "Passport": st.sidebar.selectbox("Passport", [0, 1]),
        "PitchSatisfactionScore": st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3),
        "OwnCar": st.sidebar.selectbox("Own Car", [0, 1]),
        "NumberOfChildrenVisiting": st.sidebar.slider("Number Of Children Visiting", 0, 3, 0),
        "Designation": st.sidebar.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        ),
        "MonthlyIncome": st.sidebar.number_input(
            "Monthly Income", min_value=10000, max_value=200000, value=50000
        )
    }
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer is likely to purchase (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Customer is unlikely to purchase (Probability: {probability:.2f})")
