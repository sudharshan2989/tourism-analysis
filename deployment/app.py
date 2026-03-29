import streamlit as st
import pandas as pd
import joblib


# Download and load the trained model
model_path = hf_hub_download(repo_id="sudharshanc/tourism-analysis", filename="best_tourism_analysis_model_v1.joblib")
model = joblib.load(model_path)
'''
# Load the trained model (saved earlier with joblib.dump)
model = joblib.load("best_tourism_analysis_model_v1.joblib")
'''
# Streamlit UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the newly introduced **Wellness Tourism Package** 
based on their demographic and engagement details. Please enter the customer information below to get a prediction.
""")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
monthly_income = st.number_input("Monthly Income (USD)", min_value=0, max_value=100000, value=2000)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=30)
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
designation = st.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "AVP", "VP"])

passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'MonthlyIncome': monthly_income,
    'DurationOfPitch': duration_pitch,
    'NumberOfTrips': num_trips,
    'NumberOfFollowups': num_followups,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfChildrenVisiting': num_children_visiting,
    'PitchSatisfactionScore': pitch_score,
    'TypeofContact': typeof_contact,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'ProductPitched': product_pitched,
    'CityTier': city_tier,
    'PreferredPropertyStar': preferred_star,
    'Designation': designation,
    'Passport': passport,
    'OwnCar': own_car
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Customer is likely to purchase the package (probability: {probability:.2f})")
    else:
        st.error(f"❌ Customer is unlikely to purchase the package (probability: {probability:.2f})")
