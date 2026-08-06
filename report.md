

# Jet Engine Hospital: Early Warning System
**Machine Learning Capstone Technical Report**
**Dataset:** NASA C-MAPSS (FD001, FD003, FD004)

## 1. Problem Framing and Business Context
The goal of this project was to build a multi-task early-warning system for commercial turbofan engines using the NASA C-MAPSS dataset. In the aviation industry, maintenance is governed by asymmetric costs. A false alarm (predicting an engine will fail when it is actually healthy) costs the airline money in unnecessary inspections and downtime. However, a missed warning (failing to predict a breakdown) can result in catastrophic mid-air failure. 

To address this, I designed a system that doesn't just predict the exact Remaining Useful Life (RUL) of the engine, but also calculates a 30-cycle failure risk probability. These outputs are combined to give mechanics a clear CONTINUE, INSPECT, or STOP recommendation.

## 2. Data Audit and Leakage Controls
Before training any models, I audited the FD001 dataset. The dataset consists of multivariate time-series data without headers, containing an engine ID, cycle time, 3 operational settings, and 21 sensor measurements.
![alt text](image.png)

I noticed early on that some sensors (like Sensor 1, Sensor 5, and Sensor 16) were completely flatline across the entire lifespan of the engines. While I initially kept them, in a future optimized build, dropping these zero-variance sensors would speed up training.

### Preventing Data Leakage
The grading rubric was strict about avoiding data leakage. I implemented two hard rules:
1. **Strict Engine-Level Split:** I split the data by `engine_id` (70 for training, 30 for validation) *before* doing any scaling or feature engineering. If I had done a random row split, data from the same engine would end up in both training and testing, which allows the model to cheat.
2. **Handling the Test Set Trap:** The training data is "run-to-failure," meaning the final row is the actual death of the engine. However, the test data is censored—it stops randomly. I had to manually calculate the true RUL for the test set by mapping the final observed cycle of each engine to the `RUL_FD001.txt` answer key. This ensured my final metrics were completely honest.

## 3. Time-Series Feature Engineering
Raw sensor data fluctuates heavily due to noise. To help the model see the underlying degradation trends, I engineered rolling window features. 

I applied a 5-cycle trailing rolling mean to all 21 sensors. **Crucially, I only used trailing windows, not centered windows.** Using a centered window would mean the average at Cycle 10 includes data from Cycle 11 and 12. This violates causality (time travel) and would instantly invalidate the model.

Finally, I used a `StandardScaler` to normalize the sensor readings. I fit the scaler strictly on the 70 training engines, and only used `.transform()` on the validation and test sets. 

## 4. Model Selection and Architecture
I framed the problem using two distinct machine learning tasks:
* **Regression (Predicting exact RUL):** I used a `RandomForestRegressor` (50 estimators, max depth 10). I chose Random Forest because tree-based models handle non-linear sensor degradation very well without needing complex mathematical transformations.
* **Classification (30-Cycle Warning):** I used `LogisticRegression` as a transparent baseline. This model predicts a binary outcome: will the engine fail in the next 30 flight cycles?

## 5. Stage 1 Results (FD001 Foundation)
After tuning the models on the validation set, I ran a final evaluation against the unseen official Test dataset.

**Final Test Metrics (FD001):**
* **Regression MAE:** 35.78 cycles
* **Regression RMSE:** 47.06 cycles
* **Classification Precision (30-cycle):** 0.81 (81%)

**Error Analysis:** 
An MAE of ~35 cycles sounds high at first glance, but it makes logical sense when you look at the engine lifecycle. When an engine is brand new (e.g., cycle 10), the sensors are perfectly healthy. The model has no way of knowing if this specific engine will last 150 cycles or 250 cycles, so it guesses the average, resulting in a high error margin early on. However, as the engine enters its final 50 cycles, the sensor degradation becomes obvious, and the model's accuracy tightens significantly. 

Because my Logistic Regression achieved 81% precision, it means that when the system warns a mechanic that failure is imminent within 30 days, it is correct 8 out of 10 times. This low false-alarm rate minimizes the "Early-warning burden" (unnecessary maintenance costs) for the airline.

## 6. Stage 2 and Bonus Findings (FD003 & FD004)
To test the robustness of my pipeline, I stressed it against FD003 and FD004. I deliberately kept my preprocessing and model architecture exactly the same to see how it would handle new complexities.

* **Stage 2 (FD003 - Multiple Fault Modes):** My MAE jumped to **58.54**. This dataset introduces a second failure mode (Fan Degradation) alongside the original HPC Degradation. Because my Random Forest was a single global model, it struggled to compromise between two totally different sensor signatures. 
* **Bonus (FD004 - Multiple Faults + 6 Operating Conditions):** My MAE stayed high at **54.69**. FD004 includes changes in altitude and throttle (operating settings 1, 2, and 3). Because my `StandardScaler` was applied globally, it completely failed to account for these environmental shifts. 

**Conclusion on Generalization:** A basic pipeline cannot generalize to FD004. To fix this in the future, I would need to implement "condition-aware normalization"—clustering the data by the 6 operating regimes first, and then scaling the sensors independently within each regime.

## 7. Dashboard Logic and Local Deployment

To translate model outputs into business actions, I built an interactive Streamlit dashboard. 

**Decision Logic:**
* **STOP (Red):** Triggered if 30-day failure risk is > 70% OR predicted RUL is < 15 cycles.
* **INSPECT (Amber):** Triggered if failure risk is > 35% OR predicted RUL is < 35 cycles OR the anomaly score (deviation of Sensor 11) spikes above 2.0.
* **CONTINUE (Green):** Engine is healthy.
---
***Three colors for same engine in different Filght Cycles***


**Green** for engine id = 35 and Current Flight Cycle (Age) = 152

![alt text](image-1.png)

**Yellow** for engine id = 35 and Current Flight Cycle (Age) = 163

![alt text](image-2.png)

**Red** for engine id = 35 and Current Flight Cycle (Age) = 192

![alt text](image-3.png)


**Deployment Constraints (Addressing the Hugging Face Requirement):**
While the original project scope required deploying this dashboard to Hugging Face Spaces, I ran into persistent environment issues. Specifically, the `.pkl` artifact files created on my Windows machine using local `joblib` and `scikit-learn` versions caused fatal `ValueError` dtype crashes when attempting to unpickle them on the Linux-based cloud environments used by Streamlit Cloud and Hugging Face. 

To ensure the delivery of a working, demonstrable product, I abandoned the cloud deployment and successfully hosted the application locally. By utilizing a local Python virtual environment (`venv`), I ensured all package dependencies remained perfectly aligned, allowing the app to run seamlessly on `localhost:8501`.

