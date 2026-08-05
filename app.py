import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Jet Engine Hospital", layout="wide")
st.title("✈️ Jet Engine Hospital: Early Warning System")
st.write("Predicting turbofan engine failure before it happens.")

# --- 2. LOAD DATA AND MODELS ---
# We use @st.cache_data so the app doesn't reload the file every time you click a button
@st.cache_data
def load_data():
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    # We load the test data to simulate real-time engine monitoring
    df = pd.read_csv('./data/test_FD001.txt', sep='\s+', header=None, names=cols)
    return df, cols

@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    rul_model = joblib.load('rul_model.pkl')
    clf_model = joblib.load('clf_model.pkl')
    return scaler, rul_model, clf_model

df, cols = load_data()
scaler, rul_model, clf_model = load_models()
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# --- 3. SIDEBAR (USER CONTROLS) ---
st.sidebar.header("Control Panel")
# User selects an engine to monitor
selected_engine = st.sidebar.selectbox("Select Engine ID:", df['engine_id'].unique())

# Filter data just for that engine
engine_data = df[df['engine_id'] == selected_engine].copy()
max_cycle = int(engine_data['cycle'].max())

# User selects the current day (cycle) using a slider
current_cycle = st.sidebar.slider("Current Flight Cycle (Age):", min_value=1, max_value=max_cycle, value=max_cycle)

# --- 4. DATA PROCESSING FOR AI ---
# Filter data up to the cycle the user selected
history = engine_data[engine_data['cycle'] <= current_cycle].copy()

# Apply the exact same math we did in training!
history[sensor_cols] = scaler.transform(history[sensor_cols])

# Calculate 5-day rolling average
history_grouped = history[sensor_cols].rolling(window=5, min_periods=1).mean()
history_grouped.columns = [f"{c}_mean" for c in sensor_cols]
features_df = pd.concat([history, history_grouped], axis=1)

features = sensor_cols + [f"{c}_mean" for c in sensor_cols]

# Grab ONLY the last row (the current day) to make a prediction
current_state = features_df.iloc[-1:][features]

# --- 5. AI PREDICTIONS ---
predicted_rul = int(rul_model.predict(current_state)[0])
# predict_proba gets the actual percentage/probability of failure
failure_risk = clf_model.predict_proba(current_state)[0][1] * 100 

# --- 6. DECISION LOGIC (CONTINUE / INSPECT / STOP) ---
if failure_risk > 70 or predicted_rul < 15:
    status = "🛑 STOP"
    color = "red"
    reason = "Critical: High probability of failure within 30 cycles or RUL < 15."
elif failure_risk > 35 or predicted_rul < 35:
    status = "⚠️ INSPECT"
    color = "orange"
    reason = "Warning: Elevated risk detected. Schedule maintenance soon."
else:
    status = "✅ CONTINUE"
    color = "green"
    reason = "Safe: Engine operating normally. Next review next cycle."

# --- 7. DASHBOARD VISUALS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<h3 style='text-align: center;'>Remaining Useful Life</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{predicted_rul} Cycles</h1>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<h3 style='text-align: center;'>30-Cycle Failure Risk</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{failure_risk:.1f}%</h1>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<h3 style='text-align: center;'>Action Required</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{status}</h1>", unsafe_allow_html=True)
    st.caption(f"**Reason:** {reason}")

st.divider()

# --- 8. ENGINE TIMELINE PLOT ---
st.subheader("Engine Sensor Timeline (Sensor 11)")
st.write("Sensor 11 is highly correlated with engine degradation.")

fig, ax = plt.subplots(figsize=(10, 3))
# Plot the history up to current day
ax.plot(history['cycle'], history['sensor_11'], label='Sensor 11 (Scaled)', color='blue')
# Put a red dot on the current day
ax.scatter(current_cycle, history.iloc[-1]['sensor_11'], color='red', s=100, label='Current Cycle', zorder=5)

ax.set_xlabel("Flight Cycle")
ax.set_ylabel("Sensor Value")
ax.legend()
st.pyplot(fig)