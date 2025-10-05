# ------------------- Diabetes Detection Streamlit Dashboard -------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from io import BytesIO
import time # For the loading animation

# ------------------ CUSTOM CSS (ENHANCED) ------------------
st.markdown("""
    <style>
    /* Gradient Background for the main body/app */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%); /* Light blue gradient */
    }
    
    /* Main Content Container Styling */
    .main {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.1); /* Stronger shadow */
    }

    /* Primary Button Styling - Green/Teal */
    .stButton>button {
        background-color: #00796B; /* Teal */
        color: white;
        font-weight: bold;
        border-radius: 12px;
        height: 50px; /* Slightly taller */
        width: 100%;
        transition: all 0.3s ease; /* Smooth transition for hover */
        border: none;
        box-shadow: 0 4px 6px rgba(0, 121, 107, 0.3);
    }
    .stButton>button:hover {
        background-color: #004d40; /* Darker Teal */
        transform: translateY(-2px); /* Slight lift on hover */
    }
    
    /* Headers Styling */
    h1, h2, h3 {
        color: #004d40; /* Dark Teal for headers */
        border-bottom: 2px solid #b2dfdb; /* Underline effect */
        padding-bottom: 5px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc { /* Targeting sidebar components */
        background-color: #e0f2f1; /* Lighter Teal for sidebar */
        border-right: 3px solid #004d40;
    }
    
    /* Metric styling */
    .css-1r6dm1k {
        border: 1px solid #00796B;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Custom Alert for Prediction */
    .stAlert {
        border-radius: 10px;
        font-size: 1.1em;
        font-weight: 600;
    }
    
    /* Custom Spinner/Loader Look */
    .stSpinner > div > div {
        border-top-color: #00796B !important;
    }

    </style>
""", unsafe_allow_html=True)

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Advanced Diabetes Detection Dashboard", page_icon="💖", layout="wide")

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model():
    # Placeholder for model loading. Replace with actual logic.
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("🚨 Error: Model file 'diabetes_model.pkl' not found. Ensure it's in the correct path.")
        return None

model = load_model()

# ------------------ SIDEBAR ------------------
st.sidebar.title("🩺 Control Panel")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", ["🏠 Home", "📈 Prediction", "📊 Data Insights", "💬 Health Chatbot", "💡 Health Tips"])
st.sidebar.markdown("---")
st.sidebar.info("Model: LogReg/RF | Source: Pima Indians Dataset")


# ------------------ HOME ------------------
if page == "🏠 Home":
    st.title("💖 Advanced AI-Powered Diabetes Detection Dashboard")
    st.image("https://img.freepik.com/free-vector/diabetes-awareness-concept-illustration_114360-20545.jpg",
              caption="Focusing on Health and Prevention", use_container_width=True)
    
    st.markdown("""
        ## Welcome to Your Health Hub! 
        This dashboard uses a **Machine Learning model** to assess diabetes risk based on key biometric data.
        
        ### Key Features:
        - **Single Prediction:** Get instant risk assessment with feature impact (SHAP).
        - **Batch Prediction:** Upload a CSV for population-level analysis.
        - **Interactive Chatbot (New!):** Get instant answers to common health questions.
        - **Data Insights:** Visualize data distribution and correlations.
        
        Click on **'📈 Prediction'** to get started!
    """)
    st.divider()
    
# ------------------ PREDICTION ------------------
elif page == "📈 Prediction":
    st.title("🔬 Individual Risk Assessment")

    # Animation Placeholder: Use a custom Lottie or just a simple markdown animation
    st.markdown("### Input Patient Vitals")

    # Input Fields in Columns
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.slider("Pregnancies", 0, 17, 1, help="Number of times pregnant")
        Glucose = st.slider("Glucose Level (mg/dL)", 40, 250, 120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
        BloodPressure = st.slider("Blood Pressure (mmHg)", 0, 122, 70, help="Diastolic blood pressure")
        SkinThickness = st.slider("Skin Thickness (mm)", 0, 99, 20, help="Triceps skin fold thickness")
    with col2:
        Insulin = st.slider("Insulin (mu U/ml)", 0, 846, 80, help="2-Hour serum insulin")
        BMI = st.slider("BMI", 0.0, 67.1, 25.0, help="Body mass index")
        DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5, help="Genetic predisposition score")
        Age = st.slider("Age", 21, 81, 30, help="Patient's age in years")

    st.markdown("---")
    if st.button("🚀 Analyze Risk and Explain"):
        if model is None:
            st.error("Cannot proceed: Model not loaded.")
        else:
            with st.spinner('Calculating risk and feature impact...'):
                time.sleep(1) # Simulate a small processing time for the animation

                input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                         Insulin, BMI, DiabetesPedigreeFunction, Age]])
                try:
                    prediction = model.predict(input_data)[0]
                    # Check if model has predict_proba
                    if hasattr(model, 'predict_proba'):
                        risk_score = model.predict_proba(input_data)[0][1] * 100
                        confidence = np.max(model.predict_proba(input_data)) * 100
                    else:
                        risk_score = prediction * 100 # Default to 100% if positive
                        confidence = 100 # Assume high confidence for binary model without probability

                    st.markdown("---")
                    st.subheader("✅ Prediction & Confidence")
                    
                    col_res, col_conf = st.columns(2)
                    with col_res:
                        if prediction == 1:
                            st.error("🚨 **RESULT: Likely Diabetic**")
                        else:
                            st.success("🎉 **RESULT: Likely Non-Diabetic**")

                    with col_conf:
                        st.metric("Estimated Risk (%)", f"{risk_score:.2f}%", 
                                  delta_color="off", help="Probability of being diabetic")
                        st.metric("Model Confidence", f"{confidence:.2f}%", 
                                  delta_color="off", help="Model's certainty in its prediction")


                    # **IMPROVED: Gauge/Radial Bar for Risk** (Using matplotlib for custom visualization)
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plt.rcParams['font.size'] = 12
                    
                    # Create a simple "gauge" using bar plot
                    risk_level = risk_score / 100
                    colors = ['#4BB543' if risk_score < 30 else '#F39C12' if risk_score < 60 else '#FF4B4B']
                    
                    ax.barh([0], [risk_level], color=colors, height=0.5)
                    ax.barh([0], [1 - risk_level], left=[risk_level], color='#cccccc', height=0.5)
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xticks(np.linspace(0, 1, 6))
                    ax.set_xticklabels([f'{int(i*100)}%' for i in np.linspace(0, 1, 6)])
                    ax.set_yticks([])
                    ax.set_title(f"Risk Gauge: {risk_score:.2f}%", pad=15)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.divider()

                    # Recommendations
                    st.subheader("📌 Actionable Recommendations")
                    if risk_score < 30:
                        st.info("🟢 **Low Risk:** Maintain a healthy lifestyle, retest annually.")
                    elif risk_score < 60:
                        st.warning("🟡 **Moderate Risk:** Regular monitoring of Glucose/BMI is crucial. Consult a doctor for preventative measures.")
                    else:
                        st.error("🔴 **High Risk:** **Immediate consultation** with a healthcare professional is strongly advised.")

                    # SHAP Explanation - More robust try/except
                    st.divider()
                    st.subheader("🧠 Model Explainability (Feature Impact)")
                    
                    # Create a DataFrame for SHAP
                    df_single = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
                    
                    try:
                        explainer = shap.TreeExplainer(model) # Use TreeExplainer if a tree-based model
                        shap_values = explainer.shap_values(df_single)
                        
                        # Assuming binary classification, get values for the positive class (1)
                        if isinstance(shap_values, list):
                            shap_values_for_plot = shap_values[1]
                        else:
                            shap_values_for_plot = shap_values

                        # Waterfall Plot for single prediction
                        fig_shap = plt.figure()
                        shap.waterfall_plot(shap.Explanation(values=shap_values_for_plot[0], 
                                                           base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                                           data=df_single.iloc[0].values, 
                                                           feature_names=df_single.columns.tolist()),
                                            show=False)
                        st.pyplot(fig_shap)
                        plt.close(fig_shap)
                        
                    except Exception as shap_e:
                        st.info(f"⚠️ SHAP explanation not available for this model type/setup: {shap_e}")
                        
                except Exception as e:
                    st.error(f"Prediction failed: Check input data or model compatibility. Error: {e}")

# ------------------ DATA INSIGHTS ------------------
elif page == "📊 Data Insights":
    st.title("📈 Batch Analysis & Data Visualizations")
    
    # **NEW FEATURE: Download Template**
    example_df = pd.DataFrame({
        'Pregnancies': [1, 5, 0], 'Glucose': [120, 180, 100], 'BloodPressure': [70, 90, 60],
        'SkinThickness': [35, 0, 25], 'Insulin': [0, 300, 50], 'BMI': [33.6, 45.0, 22.1],
        'DiabetesPedigreeFunction': [0.627, 0.45, 0.35], 'Age': [50, 42, 25]
    })
    csv_template = example_df.to_csv(index=False).encode('utf-8')
    st.download_button("📂 Download CSV Template", data=csv_template, file_name="diabetes_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("📂 Upload CSV for Batch Prediction (8 features required)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # **Data Cleaning/Prep Suggestion (Advanced Feature)**
        if st.checkbox("Show Data Cleaning Summary"):
            zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            zero_counts = (df[zero_cols] == 0).sum()
            st.warning(f"Columns with 0 values (potentially missing data): {zero_counts.to_dict()}")
        
        st.success(f"✅ Data Uploaded Successfully. Shape: {df.shape}")
        st.dataframe(df.head(10))

        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        missing_cols = [col for col in features if col not in df.columns]
        
        if missing_cols:
            st.error(f"🚨 Missing mandatory columns: {missing_cols}. Please check your CSV file structure.")
        else:
            # Batch prediction
            if model is None:
                st.error("Cannot perform batch prediction: Model not loaded.")
            else:
                with st.spinner('Performing batch prediction...'):
                    # Data imputation (basic mean/median for 0 values) - OPTIONAL but good practice
                    df_clean = df[features].replace({col: df[col].mean() for col in ['Glucose', 'BloodPressure', 'BMI'] if (df[col]==0).any()}, inplace=False)

                    df['Risk (%)'] = model.predict_proba(df_clean)[:, 1]*100
                    df['Prediction'] = np.where(df['Risk (%)'] > 50, 'Diabetic', 'Non-Diabetic')
                    st.success("Batch Prediction Complete!")

                    st.markdown("### 📈 Summary Statistics")
                    st.write(df.describe().T) # Transposed for better view

                    st.markdown("### 🧾 Prediction Results (Sample)")
                    st.dataframe(df[['Pregnancies','Glucose','BMI','Age','Risk (%)','Prediction']].head(10))

                    st.markdown("### 📊 Key Visualizations")
                    
                    # Correlation Heatmap (Refined)
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.markdown("#### 🔥 Feature Correlation")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        sns.heatmap(df[features + ['Risk (%)']].corr(), annot=False, cmap="viridis", linewidths=.5, ax=ax)
                        st.pyplot(fig)
                        plt.close()

                    # Distribution Plot: Age (Refined)
                    with col_viz2:
                        st.markdown("#### 👥 Age Distribution by Prediction")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        sns.histplot(data=df, x='Age', hue='Prediction', multiple='stack', bins=20, kde=True, 
                                     palette={'Diabetic':'#FF4B4B','Non-Diabetic':'#4BB543'}, ax=ax)
                        st.pyplot(fig)
                        plt.close()
                        
                    st.divider()
                    
                    # Scatter Plot: BMI vs Glucose (Retained)
                    st.markdown("#### 🩸 BMI vs Glucose Scatter Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x='BMI', y='Glucose', hue='Prediction', data=df,
                                    size='Risk (%)', sizes=(20, 200), alpha=0.7,
                                    palette={'Diabetic':'#E74C3C','Non-Diabetic':'#27AE60'}, ax=ax)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Download processed CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Analyzed Results", data=csv, file_name="Diabetes_Batch_Analysis.csv", mime="text/csv")
    else:
        st.info("📁 Upload a CSV file structured with the 8 input features to run batch analysis and unlock visualizations.")

# ------------------ HEALTH CHATBOT (NEW FEATURE) ------------------
elif page == "💬 Health Chatbot":
    st.title("💬 Your Health Assistant")
    st.markdown("""
        Hi! I'm your **AI Health Assistant**. I can answer common questions about diabetes, symptoms, and prevention.
        *Disclaimer: I am an AI; always consult a doctor for medical advice.*
    """)
    st.divider()

    # --- Simple Placeholder Chat Logic ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about diabetes, symptoms, or prevention..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Basic Bot Response Logic (Replace with actual LLM call in a real app) ---
        response = ""
        prompt_lower = prompt.lower()
        if "symptom" in prompt_lower or "sign" in prompt_lower:
            response = "Common symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, and blurred vision."
        elif "prevent" in prompt_lower or "avoid" in prompt_lower:
            response = "To help prevent Type 2 diabetes, focus on maintaining a healthy weight, eating a balanced diet rich in fiber, and getting regular exercise (at least 150 minutes per week)."
        elif "bmi" in prompt_lower or "weight" in prompt_lower:
            response = "BMI (Body Mass Index) is a measure of body fat based on your height and weight. Maintaining a healthy BMI (typically under 25) significantly lowers diabetes risk."
        elif "insulin" in prompt_lower:
            response = "Insulin is a hormone produced by the pancreas that regulates the amount of glucose in the blood. In diabetes, either the body doesn't produce enough insulin, or it can't use it effectively."
        else:
            response = "I'm sorry, I can only answer general questions about diabetes, prevention, and related health topics right now. For specific advice, please consult a medical professional."

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    # ---------------------------------------------
    
# ------------------ HEALTH TIPS (Retained and Refined) ------------------
elif page == "💡 Health Tips":
    st.title("💡 Lifestyle & Wellness Strategies")
    st.markdown("""
        ## Take Control of Your Health! 
        These tips can significantly reduce your risk of developing (or managing) Type 2 Diabetes.
    """)
    
    st.divider()

    col_tip1, col_tip2, col_tip3 = st.columns(3)

    with col_tip1:
        st.header("🍎 Nutrition Focus")
        st.markdown("""
        - **Prioritize Fiber:** Eat plenty of fruits, non-starchy vegetables, and whole grains (oats, brown rice).
        - **Limit Sugar:** Avoid all sugary drinks (sodas, juices) and highly processed foods.
        - **Healthy Fats:** Choose unsaturated fats like avocados, nuts, and olive oil over saturated and trans fats.
        """)

    with col_tip2:
        st.header("🏃‍♂️ Physical Activity")
        st.markdown("""
        - **Cardio is Key:** Aim for **150 minutes** of moderate-intensity aerobic exercise per week (e.g., brisk walking, swimming).
        - **Strength Training:** Incorporate muscle-building exercises 2-3 times a week; muscle helps absorb glucose.
        - **Move Often:** Don't sit for long periods. Get up and stretch or walk every 30 minutes.
        """)

    with col_tip3:
        st.header("🧘‍♀️ Monitoring & Wellness")
        st.markdown("""
        - **Know Your Numbers:** Regularly check your **BMI**, **Blood Pressure**, and **Cholesterol**.
        - **Quality Sleep:** 7-9 hours of consistent sleep helps regulate glucose metabolism.
        - **Stress Management:** Chronic stress elevates cortisol, which can raise blood sugar. Practice mindfulness or deep breathing.
        """)
        
    st.divider()
    st.caption("Disclaimer: This information is for educational purposes only and is not a substitute for medical advice.")

# ------------------ FOOTER ------------------
st.markdown("<hr><center>© 2025 Advanced Diabetes Detection Dashboard | Developed using Streamlit & Machine Learning</center>", unsafe_allow_html=True)