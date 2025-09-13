# 💧 AquaSentinel: Predictive Water Safety & Treatment Guide

## **Project Overview**
**AquaSentinel** is an interactive web application built with **Streamlit** that allows users to analyze and predict the **potability of water** using both **machine learning** and **color-based pH strip detection**.  

The application helps users **assess water quality quickly**, get **safety warnings**, and receive **treatment suggestions** for unsafe water.  

This tool is ideal for **home users, researchers, or field technicians** who want to monitor water quality without expensive equipment.

---

## **Features**

### **1. Basic Mode**
- Quick check using **pH value only**.
- Warns if water is **too acidic** or **too alkaline**.

### **2. Advanced Mode**
- Full analysis of water quality parameters:
  - pH
  - Hardness
  - Solids (ppm)
  - Chloramines (ppm)
  - Sulfate (mg/L)
  - Conductivity (µS/cm)
  - Organic Carbon
  - Trihalomethanes (µg/L)
  - Turbidity (NTU)
- Uses a **Random Forest ML model** to predict if water is **drinkable**.
- Displays **probability of being potable**.
- Gives **instant warnings** for unsafe values.

### **3. Upload Mode (pH Strip Image)**
- Users can **upload a photo of a pH test strip**.
- Application extracts the **dominant color** using **KMeans clustering**.
- Maps color to approximate **pH value**.
- Provides **visual warnings** for acidic or alkaline water.

### **4. Live Camera Scan**
- Take a **live photo of a pH strip** using your webcam.
- Option to **upload an image** as a fallback.
- Real-time **color detection** and **safety overlay**.
- Provides **predicted pH** and **safety status** directly on the image.

---

## **Technologies & Libraries Used**
- **Python 3.12+**
- **Streamlit** – Frontend web interface.
- **Pandas & NumPy** – Data handling and preprocessing.
- **Scikit-learn** – Machine learning (Random Forest) and preprocessing.
- **Imbalanced-learn (SMOTE)** – Handling class imbalance.
- **KNNImputer** – Handling missing values.
- **OpenCV** – Image processing & dominant color extraction.
- **Matplotlib / Seaborn** – Optional visualization.
- **KMeans Clustering** – Color extraction from pH strips.

---

## **Dataset**
- **Dataset:** `water_potability.csv`
- Contains **9 water quality features** and a **Potability label (0 or 1)**.
- Source: Public water potability datasets available on **Kaggle / UCI ML Repository**.






