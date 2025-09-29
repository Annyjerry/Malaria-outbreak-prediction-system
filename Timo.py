
import streamlit as st
import joblib
import numpy as np

# Load the saved Naive Bayes model
model = joblib.load("malaria_outbreak.pkl")

# Feature names and descriptions
feature_descriptions = {
    "Rainfall": "Total rainfall in the region (in mm). High rainfall often creates conditions favorable for mosquito breeding.",
    "MinTemperature": "Minimum temperature recorded in the region (in °C). Extreme temperatures can affect mosquito survival rates.",
    "MaxTemperature": "Maximum temperature recorded in the region (in °C). High temperatures can accelerate mosquito life cycles.",
    "RelativeHumidity_1": "Relative humidity percentage recorded at 8:00 AM. Higher humidity supports mosquito activity.",
    "RelativeHumidity_2": "Relative humidity percentage recorded at 2:00 PM. Changes in humidity levels influence mosquito behavior.",
    "MVP": "Mean Vector Population, representing mosquito density in the area. Higher values indicate increased transmission risk.",
    "MalariaCases": "The number of malaria cases recorded in the region. High case numbers indicate an ongoing or imminent outbreak."
}

# Streamlit interface
st.title("Malaria Outbreak Prediction")
st.markdown(
    """
    This app predicts the likelihood of a malaria outbreak (High Risk or Low Risk) based on environmental, mosquito population, and malaria case data.
    """
)

st.sidebar.header("Feature Information")
for feature, description in feature_descriptions.items():
    st.sidebar.markdown(f"**{feature}**: {description}")

# Collect user input
st.header("Input Features")
inputs = {}
for feature, description in feature_descriptions.items():
    inputs[feature] = st.number_input(
        f"{feature}", 
        help=description, 
        min_value=0.0, 
        step=1.0, 
        format="%.2f"
    )

# Convert user input to NumPy array
input_values = np.array(list(inputs.values())).reshape(1, -1)

# Predict outcome
if st.button("Predict"):
    prediction = model.predict(input_values)[0]
    outcome = "High Risk" if prediction == 1 else "Low Risk"
    st.subheader("Prediction Outcome")
    st.write(f"The predicted likelihood of malaria outbreak is **{outcome}**.")
