# app1.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

# ---------------------------
# Load & Train Model (Backend)
# ---------------------------

# Load dataset
df = pd.read_csv("water_potability.csv")

# Features
feature_columns = ['ph','Hardness','Solids','Chloramines','Sulfate',
                   'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']

# Handle missing values
imputer = KNNImputer(n_neighbors=5)
df[feature_columns] = imputer.fit_transform(df[feature_columns])

# Split data
X = df[feature_columns]
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_res, y_train_res)

# ---------------------------
# Prediction Function
# ---------------------------

def predict_water_potability(user_input, threshold=0.5):
    ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity = user_input

    # Rule-based quick safety checks
    if ph < 6.5 or ph > 8.5:
        return "Not Drinkable ❌", 0.0
    if solids > 2000 or chloramines > 10 or sulfate > 400:
        return "Not Drinkable ❌", 0.0
    if turbidity > 5 or trihalomethanes > 100:
        return "Not Drinkable ❌", 0.0

    # ML model prediction
    user_df = pd.DataFrame([user_input], columns=feature_columns)
    user_scaled = scaler.transform(user_df)
    prob = model.predict_proba(user_scaled)[0][1]
    prediction = "Drinkable ✅" if prob >= threshold else "Not Drinkable ❌"
    return prediction, prob

# ---------------------------
# Streamlit Frontend
# ---------------------------

st.set_page_config(page_title="💧 AquaSentinel", page_icon="💧", layout="centered")

st.title("💧 H2O Insight: Predictive Water Safety & Treatment Guide")
st.write("Select a mode to analyze your water quality:")

# Create tabs
tab1, tab2 = st.tabs(["Basic Mode (pH only)", "Advanced Mode (Full Analysis)"])

# ---------------------------
# Tab 1: Basic Mode
# ---------------------------
with tab1:
    st.subheader("🔹 Basic Mode - Quick Check with pH")
    st.write("pH is the most common and easy test for water safety. Enter your water pH value:")

    ph_basic = st.number_input("pH (0 - 14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)

    if st.button("Check pH Safety", key="basic"):
        if ph_basic < 6.5:
            st.error("❌ Not Safe - Too Acidic")
            st.markdown("<h4 style='color:red;'>⚠️ Acidic water may cause corrosion, stomach issues, and bad taste.</h4>", unsafe_allow_html=True)
        elif ph_basic > 8.5:
            st.error("❌ Not Safe - Too Alkaline")
            st.markdown("<h4 style='color:red;'>⚠️ Alkaline water may cause deposits, bitter taste, and health risks.</h4>", unsafe_allow_html=True)
        else:
            st.success("✅ Safe - Ideal Range")
            st.markdown("<h4 style='color:green;'>🌿 Water is within safe range (6.5 - 8.5). Likely drinkable!</h4>", unsafe_allow_html=True)

# ---------------------------
# Tab 2: Advanced Mode
# ---------------------------
with tab2:
    st.subheader("🔹 Advanced Mode - Full Water Quality Analysis")
    st.write("Provide detailed water parameters for ML-based prediction:")

    # Input fields in horizontal layout
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
        hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
        solids = st.number_input("Solids (ppm)", min_value=0.0, value=1000.0)

    with col2:
        chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=3.0)
        sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=200.0)
        conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0)

    with col3:
        organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
        trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=50.0)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0)

    if st.button("🔍 Predict Potability", key="advanced"):
        user_input = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        prediction, probability = predict_water_potability(user_input)

        st.subheader("Prediction Result")
        if "✅" in prediction:
            st.success(prediction)
        else:
            st.error(prediction)

        st.progress(int(probability * 100))
        st.write(f"**Probability of being Drinkable: {probability*100:.2f}%**")
