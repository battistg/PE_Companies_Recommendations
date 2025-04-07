import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(page_title="Altor Deal Sourcing Classifier", layout="wide")

# ----------------------------------
# Sidebar: App description and instructions
# ----------------------------------
st.sidebar.title("Altor Deal Sourcing Classifier")
st.sidebar.image("1714028331090.jpeg", use_container_width=True)
st.sidebar.markdown("""
**App Description:**

This internal application is designed exclusively for **Altor**. It leverages a pre-trained Random Forest model to classify companies as **High Potential** or **Low Potential** for deal sourcing. This tool aids our investment team in quickly identifying promising opportunities based on key financial and operational metrics.

**Instructions:**
- Enter the details of the new company in the form.
- Click **"Classify Company"** to receive the prediction.
- The app will display the classification along with the model’s confidence level.
""")

# ----------------------------------
# Main page
# ----------------------------------
st.title("Altor Deal Sourcing Classifier")
st.write("### Please enter the details of the new company:")

# Input fields for company details
company_name = st.text_input("Company Name", "NewCompanyX")
founded_year = st.number_input("Year Founded", min_value=1990, max_value=2024, value=2018)
num_employees = st.number_input("Number of Employees", min_value=1, value=50)
revenue_last_year = st.number_input("Revenue Last Year (in million €)", min_value=0.0, value=2.5)
revenue_growth_3y = st.number_input("3-Year Revenue Growth (%)", min_value=-100.0, value=60.0)
funding_raised = st.number_input("Total Funding Raised (in million €)", min_value=0.0, value=5.0)
num_funding_rounds = st.number_input("Number of Funding Rounds", min_value=1, max_value=10, value=3)
has_patents = st.selectbox("Has Patents?", ["No", "Yes"]) == "Yes"
has_top_tier_investor = st.selectbox("Has Top-Tier Investor?", ["No", "Yes"]) == "Yes"
founder_prev_exit = st.selectbox("Founder Has Previous Exit?", ["No", "Yes"]) == "Yes"
linkedin_followers = st.number_input("Instagram Followers", min_value=0, value=12000)
web_traffic_rank = st.number_input("Web Traffic Rank (lower is better)", min_value=1, value=300000)
press_mentions_last_6m = st.number_input("Press Mentions in Last 6 Months", min_value=0, value=5)
country = st.selectbox("Country", ['Germany', 'France', 'Sweden', 'Italy', 'Netherlands', 'Spain'])
sector = st.selectbox("Sector", ['Fintech', 'Healthtech', 'SaaS', 'AI', 'eCommerce', 'GreenTech'])

# ----------------------------------
# Load the pre-trained model and feature columns
# ----------------------------------
model = joblib.load("deal_sourcing_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ----------------------------------
# Prediction button
# ----------------------------------
if st.button("Classify Company"):
    # Create a dictionary with input values
    input_data = {
        'founded_year': founded_year,
        'num_employees': num_employees,
        'revenue_last_year': revenue_last_year,
        'revenue_growth_3y': revenue_growth_3y,
        'funding_raised': funding_raised,
        'num_funding_rounds': num_funding_rounds,
        'has_patents': has_patents,
        'has_top_tier_investor': has_top_tier_investor,
        'founder_prev_exit': founder_prev_exit,
        'instagram_followers': linkedin_followers,
        'web_traffic_rank': web_traffic_rank,
        'press_mentions_last_6m': press_mentions_last_6m,
        # Include categorical variables to be encoded later
        'country': country,
        'sector': sector
    }
    
    # Convert the dictionary into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply one-hot encoding for categorical variables: 'country' and 'sector'
    input_df = pd.get_dummies(input_df, columns=['country', 'sector'], drop_first=True)
    
    # Ensure that the input DataFrame has all the required feature columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder the columns to match the training order
    input_df = input_df[feature_columns]
    
    # Make the prediction using the pre-trained model
    prediction = model.predict(input_df)[0]
    # Assuming the class 'True' corresponds to "High Potential"
    probability = model.predict_proba(input_df)[0][1]
    
    # Display the result
    label = "High Potential" if prediction else "Low Potential"
    st.success(f"Company **{company_name}** is classified as: **{label}**")
    st.info(f"Model Confidence: **{probability * 100:.1f}%**")
