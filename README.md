# TECHSOPHY-67B9
Insurance Portfolio Risk Management Dashboard
# 🔴 Real-Time Insurance Risk Monitoring Dashboard

> ⚡️ A powerful, auto-refreshing dashboard to simulate and visualize **live insurance claims** and **portfolio risk exposure** in real-time.

---

## 🚨 Real-Time Claims Simulation (Core Feature)

This dashboard stands out by simulating **live insurance claims** every 5 seconds. It helps insurers:
- Visualize the **impact of incoming claims** instantly
- See **total processed claims update live**
- Watch the **claims feed and recent policies evolve** in real-time
- Prepare for **risk spikes and high-claim events** dynamically

This creates a near real-world environment for **risk managers** and **underwriters** to practice decision-making under pressure.

---

## 📊 What Does the Dashboard Do?

The app provides insurers with:
- 📍 A complete **portfolio overview** by region and policy type
- 📈 **Risk analysis** using correlation heatmaps and clustering
- 💥 **Stress testing** using Monte Carlo simulations
- 🔴 A **real-time dashboard** that auto-refreshes every 5 seconds, simulating dynamic market activity

Built using **Streamlit**, this dashboard combines clarity, interactivity, and real-world modeling in one seamless interface.

---

## 🧠 Key Features

✅ Real-Time Claim Updates  
✅ Live Claims Feed  
✅ Auto-refreshing Risk Visualizations  
✅ Risk Clustering with Alerts  
✅ Monte Carlo Stress Testing  
✅ CSV Export for Reporting

---

## 📁 File Structure Explained

| File / Folder                       | Purpose                                                                 |
|------------------------------------|-------------------------------------------------------------------------|
| `insurance_risk_management.py`     | 🚀 Main Streamlit app with all 5 pages and real-time claim simulation   |
| `insurance_risk_management_st.ipynb` | Notebook version for testing logic and visualizations                  |
| `insurance_policies.csv`           | Original static insurance policy data                                   |
| `streamlit_insurance_policies.csv` | Updated data reflecting live claims in real-time                        |
| `type_aggregation.csv`             | Saved results: coverage by policy type                                  |
| `region_aggregation.csv`           | Saved results: coverage by region                                       |
| `stress_test_results.csv`          | Monte Carlo simulation outputs                                          |
| `.ipynb_checkpoints/`              | Auto-generated Jupyter checkpoints (ignored in `.gitignore`)           |
| `README.md`                        | 📄 This documentation file                                              |

---

## 📊 Dashboard Pages

1. **🏠 Home** — Overview and feature guide  
2. **📊 Overview** — Portfolio breakdown by policy type and region  
3. **📈 Risk Analysis** — Correlation heatmaps and KMeans clustering  
4. **💥 Stress Testing** — Monte Carlo simulation of catastrophic losses  
5. **🔴 Real-Time Dashboard** — Auto-refreshing feed of simulated claims and risk totals

---

## 🛠️ Built With

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Scikit-learn** (for clustering)
- **Plotly** (for interactive charts)
- **streamlit_autorefresh** (for live updates)
- **Jupyter Notebook** (for prototyping)

---

