import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Client Home Predictor", layout="wide")

# --- LOAD MODELS (Using Pickle) ---
@st.cache_resource
def load_assets():
    # Ensure 'housing_models.pkl' is in the same folder
    with open('models.pkl', 'rb') as f:
        assets = pickle.load(f)
    return assets

assets = load_assets()
lr_model = assets["linear_model"]
dt_model = assets["tree_model"]
scaler = assets["scaler"]

# --- MAIN INTERFACE ---
st.title("🏡 Personal House Value Predictor")
st.write("Adjust the house features below to match the property you are looking for.")

# Create two columns for a clean look
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🛠️ Customize Your Specific House")
    
    # 1. Rooms - Most important for clients
    rm = st.slider("How many rooms do you need? (RM)", 3.0, 9.0, 6.0, help="Average number of rooms per house.")
    
    # 2. Neighborhood Status
    lstat = st.slider("Neighborhood Social Status % (LSTAT)", 1.0, 40.0, 12.0, help="Percentage of the population considered lower status.")
    
    # 3. River View
    chas = st.radio("Do you want a view of the Charles River? (CHAS)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # 4. Pollution
    nox = st.select_slider("Preferred Air Quality (NOX)", options=[0.4, 0.5, 0.6, 0.7, 0.8], value=0.5, help="Nitric oxide concentration (lower is cleaner).")

    # 5. Crime Rate
    crim = st.number_input("Maximum acceptable Crime Rate (CRIM)", 0.0, 100.0, 3.6)

    # 6. Education
    ptratio = st.slider("Preferred Pupil-Teacher Ratio (PTRATIO)", 12.0, 22.0, 18.0, help="Lower means smaller class sizes in local schools.")

    # 7. Tax
    tax = st.number_input("Property Tax Rate (TAX)", 180, 711, 408)

    # Note: We fill the other 6 technical attributes with dataset averages 
    # so the client doesn't get overwhelmed with math questions.
    features = {
        'CRIM': crim, 'ZN': 11.3, 'INDUS': 11.1, 'CHAS': chas,
        'NOX': nox, 'RM': rm, 'AGE': 68.5, 'DIS': 3.7,
        'RAD': 9.5, 'TAX': tax, 'PTRATIO': ptratio,
        'B': 356.6, 'LSTAT': lstat
    }
    user_df = pd.DataFrame(features, index=[0])

with col_right:
    st.subheader("🚀 Prediction Result")
    
    # Model Selection for the Client
    model_choice = st.selectbox("Select Model Logic", ["Linear Regression", "Decision Tree"])
    
    # Process and Predict
    user_scaled = scaler.transform(user_df)
    
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(user_scaled)
        color = "#1f77b4"
    else:
        prediction = dt_model.predict(user_scaled)
        color = "#2ca02c"
    
    # Final Display
    final_price = prediction[0] * 1000
    st.markdown(f"### Predicted Market Value:")
    st.markdown(f"<h1 style='color: {color}; text-align: center;'>${final_price:,.2f}</h1>", unsafe_allow_html=True)
    
    st.divider()
    st.write("**Why this price?**")
    if rm > 7:
        st.write("✅ High room count is significantly increasing the value.")
    if lstat > 20:
        st.write("⚠️ High LSTAT % is pulling the predicted value down.")
    if chas == 1:
        st.write("🌊 River access is adding a premium to the price.")

# --- DATA TABLE AT THE BOTTOM ---
with st.expander("See Raw Input Data (Technical)"):
    st.write(user_df)