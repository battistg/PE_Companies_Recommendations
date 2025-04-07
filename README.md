# AI-Based Deal Sourcing Prototype

This repository contains the code and synthetic dataset used to build a prototype machine learning model and a Streamlit web application for deal sourcing support. The project was developed as part of an academic module with the goal of helping **Altor Equity Partners** private equity firm to identify high-potential companies more efficiently.

## Project Overview

The prototype uses a Random Forest classifier trained on a synthetic dataset of European startups to classify companies as either High Potential or Low Potential. A Streamlit-based interface allows users to input company data manually, adjust a confidence threshold, and receive instant predictions.

*This is a Proof of Concept (PoC) version and is intended for demonstration and research purposes only.*



## App Features

- Manual company data input (form-based)
- Adjustable prediction confidence threshold (0.50, 0.80, 0.95)
- Real-time output: classification + model confidence
- Clean and intuitive user interface

Try the live version: https://pe-deal-sourcing.streamlit.app



## How It Works

- The model was trained on a synthetic dataset of 1,000 startups, created to reflect real-world investment signals.
- Categorical features were one-hot encoded, and some missing values were imputed using Generative AI for demonstration purposes.
- The app uses a pre-trained Random Forest model to generate predictions in real time.



## Repository Structure

```bash
PE_Companies_Recommendations/
├── random_forest.py                # Model training and export
├── streamlit_app_deal_sourcing.py  # Streamlit app code
├── EU_Startup_DealSourcing.csv     # Synthetic dataset
├── feature_columns.pkl             # Feature list used in training
├── deal_sourcing_model.pkl         # Trained model (Random Forest)
├── requirements.txt                # Libraries requirements
├── altor_logo.png                  # Company logo
```

## Requirements

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app_deal_sourcing.pyy
```

## License & Acknowledgments

This project was developed for academic purposes at **Trinity College Dublin**, as part of the Big Data & AI in Business module (2024/2025).

Developed by: **Giovanni Battistella**

Module Leader: **Prof. Max Darby**

⸻
