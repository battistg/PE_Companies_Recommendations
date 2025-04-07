import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Set the page configuration
st.set_page_config(page_title="Altor Deal Sourcing Classifier", layout="wide")

# ----------------------------------
# Sidebar: Navigation, description, instructions and threshold selection
# ----------------------------------
menu = st.sidebar.radio("Navigation", ["Prediction Input", "Model Insights"])

st.sidebar.image("altor_logo.jpeg", use_container_width=True)
st.sidebar.markdown("""
**App Description:**

This internal application is designed exclusively for **Altor**, a private equity firm based in Stockholm. It leverages a pre-trained Random Forest model to classify companies as **High Potential** or **Low Potential** in the final stage of deal sourcing. The tool helps our investment team by providing a clear, data-driven second opinion to support decision-making.

**Instructions:**
- In the **Prediction Input** tab, enter the company details and adjust the confidence threshold.
- Click **"Classify Company"** to receive the prediction along with an explanation of the model’s decision.
- In the **Model Insights** tab, view overall model explanations and feature importance.

**Company Website:** [altor.com](https://www.altor.com)
""")

threshold = st.sidebar.slider("Select confidence threshold", 0.50, 0.95, 0.80, 0.01)
st.sidebar.markdown("""
**Threshold Levels Explanation:**

- **0.50 (High Sensitivity – Discover):** The model is very sensitive, flagging many companies as High Potential, which helps in uncovering opportunities but may include more false positives.
- **0.80 (Balanced Approach):** A balanced setting that aims to capture the most promising leads while keeping false positives reasonable.
- **0.95 (High Precision):** The model is very strict, classifying a company as High Potential only when extremely confident, reducing false positives but potentially missing some viable opportunities.
""")

# ----------------------------------
# Prediction Input Page
# ----------------------------------
if menu == "Prediction Input":
    st.title("Altor Deal Sourcing Classifier - Prediction Input")
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
    linkedin_followers = st.number_input("LinkedIn Followers", min_value=0, value=12000)
    web_traffic_rank = st.number_input("Web Traffic Rank (lower is better)", min_value=1, value=300000)
    press_mentions_last_6m = st.number_input("Press Mentions in Last 6 Months", min_value=0, value=5)
    country = st.selectbox("Country", ['Germany', 'France', 'Sweden', 'Italy', 'Netherlands', 'Spain'])
    sector = st.selectbox("Sector", ['Fintech', 'Healthtech', 'SaaS', 'AI', 'eCommerce', 'GreenTech'])

    # Load the pre-trained model and feature columns
    model = joblib.load("deal_sourcing_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

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
            'linkedin_followers': linkedin_followers,
            'web_traffic_rank': web_traffic_rank,
            'press_mentions_last_6m': press_mentions_last_6m,
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
        
        # Obtain the prediction probability for the "High Potential" class
        proba = model.predict_proba(input_df)[0][1]
        
        # Apply the user-selected threshold
        prediction = proba >= threshold
        label = "High Potential" if prediction else "Low Potential"
        
        # Display the result
        st.success(f"Company **{company_name}** is classified as: **{label}**")
        st.info(f"Model Confidence: **{proba * 100:.1f}%** (Threshold: {threshold * 100:.1f}%)")
        
        # SHAP explanation for the current input
        st.markdown("### SHAP Explanation for This Prediction")
        explainer = shap.TreeExplainer(model)
        shap_values_input = explainer.shap_values(input_df)
        fig_shap, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values_input[1], input_df, plot_type="bar", show=False)
        st.pyplot(fig_shap)
        st.markdown("""
        **Note:** The SHAP plot above illustrates the contribution of each feature for the current prediction.
        Features with larger absolute values have a stronger influence on the model's decision.
        """)

# ----------------------------------
# Model Insights Page
# ----------------------------------
elif menu == "Model Insights":
    st.title("Altor Deal Sourcing Classifier - Model Insights")
    st.write("### Explore Overall Model Explanations")
    
    # Load the pre-trained model and feature columns
    model = joblib.load("deal_sourcing_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    
    st.subheader("Interactive Feature Importance")
    # Create a DataFrame for feature importances
    importance = model.feature_importances_
    df_feat_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Plotly bar chart for interactive feature importance
    fig = px.bar(df_feat_imp.head(10), x="Importance", y="Feature", orientation="h",
                 title="Top 10 Features Influencing the Prediction")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("SHAP Model Explanation (Summary)")
    # For demonstration, we use a dummy sample with default values (1 for all features)
    dummy_data = {col: 1 for col in feature_columns}
    dummy_df = pd.DataFrame([dummy_data])
    explainer = shap.TreeExplainer(model)
    shap_values_dummy = explainer.shap_values(dummy_df)
    fig_dummy, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_values_dummy[1], dummy_df, plot_type="bar", show=False)
    st.pyplot(fig_dummy)
    st.markdown("""
    **Note:** The above SHAP plot is generated using a dummy sample to illustrate the overall impact of features on the model’s predictions.
    In a production scenario, this would be based on a representative sample of live data.
    """)
