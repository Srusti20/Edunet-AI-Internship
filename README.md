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

## **Installation Instructions**

Installation Instructions
1. Create a Virtual Environment
python -m venv venv

2. Activate the Environment (Windows)
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt


Example requirements.txt:

streamlit
pandas
numpy
scikit-learn
imbalanced-learn
opencv-python
matplotlib
seaborn

4. Run the App
streamlit run app1.py


The app will open in your browser (usually at http://localhost:8501
).

Usage Instructions

Basic Mode: Enter pH and check safety.

Advanced Mode: Enter all water parameters → Predict Potability → View probability.

Upload Mode: Upload a pH strip image → View estimated pH and safety warnings.

Live Camera Mode: Take a live photo of your pH strip → See real-time overlay with pH and warnings.

Screenshots
Basic Mode
<img width="1577" height="823" alt="Screenshot 2025-09-13 234212" src="https://github.com/user-attachments/assets/891eef73-ccf5-474c-8d83-a9bb8cb95599" />

Advanced Mode
<img width="1068" height="870" alt="Screenshot 2025-09-13 234459" src="https://github.com/user-attachments/assets/3042cfbb-ce95-43ce-a0f1-3ef6c5da155c" />

Upload Mode

<img width="1000" height="499" alt="Screenshot 2025-09-13 234508" src="https://github.com/user-attachments/assets/0fecbd73-b145-4c63-8392-ad6210c89a4e" />
<img width="1129" height="776" alt="Screenshot 2025-09-13 234529" src="https://github.com/user-attachments/assets/ce91355a-e4ba-42a0-a8ff-f1467194228c" />



Live Camera Mode
<img width="999" height="828" alt="Screenshot 2025-09-13 234657" src="https://github.com/user-attachments/assets/6e5d7e24-e03a-46f5-8666-4e989fe8f2e7" />
<img width="939" height="756" alt="Screenshot 2025-09-13 234706" src="https://github.com/user-attachments/assets/c1c8b8f9-4288-407e-9bfd-f1b533ab536b" />




Future Enhancements

Treatment recommendations for unsafe water.

Water quality trends over time.

PDF report generation for analysis.
