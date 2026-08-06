import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Jet Engine Hospital", layout="wide")
st.title("✈️ Jet Engine Hospital: Early Warning System")
st.write("Predicting turbofan engine failure before it happens (Model trained on FD001).")

# --- 2. LOAD DATA AND MODELS ---
@st.cache_data
def load_data():
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    # Using your exact path!
    df = pd.read_csv(r"C:\Users\kouro\Jet_Engine_Project\data\test_FD001.txt", sep='\s+', header=None, names=cols)
    return df, cols

@st.cache_resource
def load_models():
    # Load the files we just created in the notebook
    scaler = joblib.load(r"C:\Users\kouro\Jet_Engine_Project\scaler.pkl")
    rul_model = joblib.load(r"C:\Users\kouro\Jet_Engine_Project\rul_model.pkl")
    clf_model = joblib.load(r"C:\Users\kouro\Jet_Engine_Project\clf_model.pkl")
    return scaler, rul_model, clf_model

df, cols = load_data()
scaler, rul_model, clf_model = load_models()
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# --- 3. SIDEBAR (USER CONTROLS) ---
st.sidebar.header("Control Panel")
selected_engine = st.sidebar.selectbox("Select Engine ID:", df['engine_id'].unique())

engine_data = df[df['engine_id'] == selected_engine].copy()
max_cycle = int(engine_data['cycle'].max())

current_cycle = st.sidebar.slider("Current Flight Cycle (Age):", min_value=1, max_value=max_cycle, value=max_cycle)

# --- 4. DATA PROCESSING FOR AI ---
history = engine_data[engine_data['cycle'] <= current_cycle].copy()

# Scale and feature engineer exactly like the notebook
history[sensor_cols] = scaler.transform(history[sensor_cols])
history_grouped = history[sensor_cols].rolling(window=5, min_periods=1).mean()
history_grouped.columns = [f"{c}_mean" for c in sensor_cols]
features_df = pd.concat([history, history_grouped], axis=1)

features = sensor_cols + [f"{c}_mean" for c in sensor_cols]
current_state = features_df.iloc[-1:][features]

# --- 5. AI PREDICTIONS ---
predicted_rul = int(rul_model.predict(current_state)[0])
failure_risk = clf_model.predict_proba(current_state)[0][1] * 100 

# Simulated Anomaly Score (Deviation from healthy baseline)
# We use sensor 11's scaled value as a proxy for an anomaly score for the dashboard
anomaly_score = abs(current_state['sensor_11'].values[0]) 

# --- 6. DECISION LOGIC (CONTINUE / INSPECT / STOP) ---
if failure_risk > 70 or predicted_rul < 15:
    status = "🛑 STOP"
    color = "red"
    reason = "Critical: High probability of failure within 30 cycles or RUL < 15."
elif failure_risk > 35 or predicted_rul < 35 or anomaly_score > 2.0:
    status = "⚠️ INSPECT"
    color = "orange"
    reason = "Warning: Elevated risk or anomaly detected. Schedule maintenance soon."
else:
    status = "✅ CONTINUE"
    color = "green"
    reason = "Safe: Engine operating normally. Next review next cycle."

# --- 7. DASHBOARD VISUALS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<h4 style='text-align: center;'>Remaining Useful Life</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>{predicted_rul} Cycles</h2>", unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='text-align: center;'>30-Cycle Risk</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {color};'>{failure_risk:.1f}%</h2>", unsafe_allow_html=True)

with col3:
    st.markdown("<h4 style='text-align: center;'>Anomaly Score</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: gray;'>{anomaly_score:.2f}</h2>", unsafe_allow_html=True)

with col4:
    st.markdown("<h4 style='text-align: center;'>Action Required</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {color};'>{status}</h2>", unsafe_allow_html=True)
    
st.caption(f"**Trigger Reason:** {reason}")
st.divider()

# --- 8. ENGINE TIMELINE PLOT ---
st.subheader("Engine Sensor Timeline (Sensor 11)")
st.write("Sensor 11 is highly correlated with engine HPC degradation.")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(history['cycle'], history['sensor_11'], label='Sensor 11 (Scaled)', color='blue')
ax.scatter(current_cycle, history.iloc[-1]['sensor_11'], color='red', s=100, label='Current Cycle', zorder=5)

ax.set_xlabel("Flight Cycle")
ax.set_ylabel("Sensor Value")
ax.legend()
st.pyplot(fig)