# app1.py

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter

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
# Helper: Extract dominant color
# ---------------------------
def extract_dominant_color(image, k=1):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img)
    most_common = Counter(kmeans.labels_).most_common(1)[0][0]
    return kmeans.cluster_centers_[most_common]


# ---------------------------
# Map color to pH (simplified scale)
# ---------------------------
def color_to_ph(rgb_color):
    # Simplified reference (pH test strip color chart)
    ph_scale = {
        4: (255, 0, 0),       # Red
        6: (255, 165, 0),     # Orange
        7: (255, 255, 0),     # Yellow
        8: (0, 128, 0),       # Green
        10: (0, 0, 255),      # Blue
        12: (128, 0, 128)     # Purple
    }
    # Find nearest match by Euclidean distance
    min_dist, best_ph = float("inf"), 7
    for ph_val, ref_rgb in ph_scale.items():
        dist = np.linalg.norm(np.array(rgb_color) - np.array(ref_rgb))
        if dist < min_dist:
            min_dist, best_ph = dist, ph_val
    return best_ph


# ---------------------------
# Streamlit Frontend
# ---------------------------

st.set_page_config(page_title="💧 AquaSentinel", page_icon="💧", layout="centered")

st.title("💧 H2O Insight: Predictive Water Safety & Treatment Guide")
st.write("Select a mode to analyze your water quality:")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Basic Mode (pH only)",
    "Advanced Mode (Full Analysis)",
    "📷 Upload Mode (pH Strip Image)",
    "📸 Live Camera Scan"
])

# ---------------------------
# Tab 1: Basic Mode
# ---------------------------
with tab1:
    st.subheader(" Basic Mode - Quick Check with pH")
    ph_basic = st.number_input("pH (0 - 14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)

    if st.button("Check pH Safety", key="basic"):
        if ph_basic < 6.5:
            st.error("❌ Not Safe - Too Acidic")
        elif ph_basic > 8.5:
            st.error("❌ Not Safe - Too Alkaline")
        else:
            st.success("✅ Safe - Ideal Range ")

# ---------------------------
# Tab 2: Advanced Mode
# ---------------------------
with tab2:
    st.subheader("Advanced Mode - Full Water Quality Analysis")
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

    if st.button("Predict Potability", key="advanced"):
        user_input = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        prediction, probability = predict_water_potability(user_input)

        st.subheader("Prediction Result")
        st.write(prediction)
        st.progress(int(probability * 100))
        st.write(f"**Probability of being Drinkable: {probability*100:.2f}%**")

# ---------------------------
# Tab 3: Camera Mode
# ---------------------------
import numpy as np
import cv2
import streamlit as st
from sklearn.cluster import KMeans

# Extract dominant color using KMeans
def extract_dominant_color(image, k=1):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.reshape((-1, 3))  # Flatten
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img)
    return tuple(map(int, kmeans.cluster_centers_[0]))

# Better calibrated pH color mapping (from strip chart)
ph_color_scale = {
    (200, 0, 0): 1,     # Dark Red
    (220, 40, 0): 2,    # Red-Orange
    (255, 90, 0): 3,    # Orange-Red
    (255, 150, 0): 4,   # Orange
    (255, 210, 50): 5,  # Yellow-Orange
    (240, 240, 100): 6, # Yellow
    (180, 220, 140): 7, # Light Green → Neutral
    (100, 180, 100): 8, # Green
    (80, 160, 180): 9,  # Green-Blue
    (60, 130, 200): 10, # Sky Blue
    (40, 90, 180): 11,  # Blue
    (20, 50, 160): 12,  # Dark Blue
    (100, 40, 180): 13, # Violet
    (130, 0, 130): 14   # Purple
}

# Function to map detected RGB to closest reference pH
def color_to_ph(rgb):
    min_dist = float("inf")
    closest_ph = None
    for ref_rgb, ph in ph_color_scale.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))  # Euclidean distance
        if dist < min_dist:
            min_dist = dist
            closest_ph = ph
    return closest_ph

# Tab 3 UI (replace your old one)
with tab3:
    st.subheader("Upload a photo of your pH strip")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Uploaded pH Strip")

        # Extract dominant color
        dominant_rgb = extract_dominant_color(img)
        predicted_ph = color_to_ph(dominant_rgb)

        st.write(f"Detected Color: {tuple(map(int, dominant_rgb))}")
        st.success(f"Estimated pH from strip: {predicted_ph}")

        # Safety check
        if predicted_ph < 6.5:
            st.error("❌ Not Safe - Too Acidic")
        elif predicted_ph > 8.5:
            st.error("❌ Not Safe - Too Alkaline")
        else:
            st.success("✅ Safe - Within Drinking Range")



# ---------------------------
# Tab 4: Live Camera / Upload Hybrid
# ---------------------------
with tab4:
    st.subheader("📸 Live Camera or Upload pH Strip Image")

    # Option to take a live photo
    camera_image = st.camera_input("Take a photo of your pH strip")

    # Option to upload an image as fallback
    uploaded_file = st.file_uploader("Or upload a photo of your pH strip", type=["jpg", "png", "jpeg"])

    image_to_process = None

    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)
    elif uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)

    if image_to_process is not None:
        st.image(image_to_process, channels="BGR", caption="Processed pH Strip Image")

        # Extract dominant color
        dominant_rgb = extract_dominant_color(image_to_process)
        predicted_ph = color_to_ph(dominant_rgb)

        st.write(f"Detected Color: {tuple(map(int, dominant_rgb))}")
        st.success(f"Estimated pH from strip: {predicted_ph}")

        # Safety check
        if predicted_ph < 6.5:
            st.error("❌ Not Safe - Too Acidic")
        elif predicted_ph > 8.5:
            st.error("❌ Not Safe - Too Alkaline")
        else:
            st.success("✅ Safe - Within Drinking Range")
