import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import time
import uuid
from streamlit_autorefresh import st_autorefresh

# Page settings
st.set_page_config(page_title='Insurance Risk Management Dashboard', layout='wide')

# Custom CSS
st.markdown('''
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f5f5f5; color: #333; }
        .header { background-color: #007bff; color: white; padding: 14px; text-align: center; border-radius: 6px; margin-bottom: 10px; }
        .stButton>button { background-color: #007bff; color: white; border-radius: 5px; padding: 6px 12px; }
        .metric-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
    </style>
''', unsafe_allow_html=True)

# Universal Session Initialization (runs once)
if 'policies' not in st.session_state:
    np.random.seed(42)
    n_policies = 1000
    policies_df = pd.DataFrame({
        'policy_id': [str(uuid.uuid4()) for _ in range(n_policies)],
        'policy_type': np.random.choice(['Auto', 'Home', 'Health', 'Commercial'], n_policies),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_policies),
        'coverage_amount': np.random.uniform(50000, 1000000, n_policies),
        'premium': np.random.uniform(1000, 10000, n_policies),
        'risk_score': np.random.uniform(0, 1, n_policies),
        'claims': np.zeros(n_policies),
        'last_updated': [time.time()] * n_policies
    })
    policies_df.to_csv('streamlit_insurance_policies.csv', index=False)
    st.session_state.policies = policies_df

if 'claims' not in st.session_state:
    st.session_state.claims = pd.DataFrame({
        'policy_id': pd.Series(dtype='str'),
        'claim_amount': pd.Series(dtype='float64'),
        'claim_time': pd.Series(dtype='float64')
    })

# Data functions
def get_policies():
    return st.session_state.policies

def simulate_real_time_claims():
    policies = st.session_state.policies
    new_claims = pd.DataFrame({
        'policy_id': np.random.choice(policies['policy_id'], size=5),
        'claim_amount': np.random.uniform(10000, 200000, 5),
        'claim_time': [time.time()] * 5
    })
    st.session_state.claims = pd.concat([st.session_state.claims, new_claims], ignore_index=True)
    for policy_id, claim_amount in zip(new_claims['policy_id'], new_claims['claim_amount']):
        policies.loc[policies['policy_id'] == policy_id, 'claims'] += claim_amount
    policies.to_csv('streamlit_insurance_policies.csv', index=False)
    return new_claims

def aggregate_risk(df):
    type_agg = df.groupby('policy_type').agg({
        'coverage_amount': 'sum', 'premium': 'sum', 'risk_score': 'mean', 'claims': 'sum'
    }).reset_index()
    type_agg['exposure_ratio'] = type_agg['coverage_amount'] / type_agg['premium']
    region_agg = df.groupby('region').agg({
        'coverage_amount': 'sum', 'premium': 'sum', 'risk_score': 'mean', 'claims': 'sum'
    }).reset_index()
    region_agg['exposure_ratio'] = region_agg['coverage_amount'] / region_agg['premium']
    return type_agg, region_agg

def correlation_analysis(df):
    pivot = df.pivot_table(values='coverage_amount', index='policy_type', columns='region', aggfunc='sum').fillna(0)
    corr_matrix = pivot.corr()
    return px.imshow(corr_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1, title='Correlation of Coverage Amount by Region'), corr_matrix

def detect_risk_concentrations(df):
    features = df[['coverage_amount', 'risk_score']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['risk_cluster'] = kmeans.fit_predict(scaled_features)
    high_risk = df[df['risk_cluster'] == df['risk_cluster'].value_counts().idxmax()]
    alerts = []
    if high_risk['coverage_amount'].sum() > df['coverage_amount'].sum() * 0.3:
        alerts.append(f"⚠️ High risk concentration: ₹ {high_risk['coverage_amount'].sum():,.2f} in cluster {df['risk_cluster'].value_counts().idxmax()}")
    return df, alerts

def monte_carlo_simulation(df, n_simulations=1000):
    claim_probs = df['risk_score'].values * 0.1
    claim_amounts = np.random.uniform(10000, 500000, (n_simulations, len(df)))
    events = np.random.rand(n_simulations, len(df)) < claim_probs
    simulated_losses = (claim_amounts * events).sum(axis=1)
    return pd.DataFrame(simulated_losses, columns=['total_claims'])

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Overview", "📈 Risk Analysis", "💥 Stress Testing", "🔴 Real-Time Dashboard"])

# Main Content
st.markdown('<div class="header">Insurance Portfolio Risk Management Dashboard</div>', unsafe_allow_html=True)
policies = get_policies()





if page == "🏠 Home":
    st.header("📖 Welcome to the Dashboard!")

    st.write("""
        This dashboard helps insurance companies monitor the health and risk levels of their active insurance policies in real-time.
        Each section in the sidebar is like a different 'page' offering a unique view into the data. Here’s what each page does:
    """)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Overview
    st.markdown("""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #333333;">
        <h3 style="color: #007bff;">📊 Overview</h3>
        <p style="font-size: 16px;">This page shows a summary of the total insurance coverage distributed across different 
        <strong>Policy Types</strong> (like Auto, Home, Health, Commercial) and <strong>Regions</strong> (North, South, East, West).</p>
        <p style="font-size: 16px;">You’ll see simple bar charts to quickly compare which categories have the highest coverage amounts 
        and premiums. It helps managers instantly grasp how their insurance portfolio is spread out.</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk Analysis
    st.markdown("""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #333333;">
        <h3 style="color: #007bff;">📈 Risk Analysis</h3>
        <p style="font-size: 16px;">This section helps identify how different regions and policy types are related by displaying a 
        <strong>correlation heatmap</strong> of coverage amounts.</p>
        <p style="font-size: 16px;">It also uses <strong>clustering techniques</strong> to group policies based on their 
        <strong>coverage amounts</strong> and <strong>risk scores</strong>. An alert appears if too much of your portfolio 
        is concentrated in a single high-risk cluster — a sign of potential trouble for insurers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stress Testing
    st.markdown("""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #333333;">
        <h3 style="color: #007bff;">💥 Stress Testing</h3>
        <p style="font-size: 16px;">This page runs a <strong>Monte Carlo simulation</strong> — a process where we simulate thousands 
        of future scenarios by randomly generating possible insurance claims based on each policy’s risk score.</p>
        <p style="font-size: 16px;">It helps managers estimate the total potential claim payouts under worst-case or unusual 
        scenarios and prepare accordingly for financial risks.</p>
    </div>
    """, unsafe_allow_html=True)

    # Real-Time Dashboard
    st.markdown("""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #333333;">
        <h3 style="color: #007bff;">🔴 Real-Time Dashboard</h3>
        <p style="font-size: 16px;">This page simulates real-time insurance claim events, automatically refreshing every 5 seconds.</p>
        <p style="font-size: 16px;">You’ll see a live feed of new claims, recently added policies, and a real-time total of all claims processed.
        It provides a sense of how active and risky your claims pipeline currently is.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 16px; color: #dddddd;'>👉 Use the sidebar to switch between these pages and explore your insurance data portfolio from different angles.</p>", unsafe_allow_html=True)

elif page == "📊 Overview":
    st.header("📊 Portfolio Overview")
    type_agg, region_agg = aggregate_risk(policies)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Exposure by Policy Type")
        st.plotly_chart(px.bar(type_agg, x='policy_type', y='coverage_amount', title='Exposure by Policy Type', text_auto='.2s'), use_container_width=True)
    with col2:
        st.subheader("Exposure by Region")
        st.plotly_chart(px.bar(region_agg, x='region', y='coverage_amount', title='Exposure by Region', text_auto='.2s'), use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.download_button("📥 Download Policies CSV", policies.to_csv(index=False), "policies.csv", "text/csv")

elif page == "📈 Risk Analysis":
    st.header("📈 Risk Analysis")
    fig_corr, _ = correlation_analysis(policies)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.subheader("Risk Concentrations")
    policies, alerts = detect_risk_concentrations(policies)
    st.plotly_chart(px.scatter(policies, x='coverage_amount', y='risk_score', color='risk_cluster',
                               hover_data=['policy_id', 'policy_type', 'region'], title='Risk Clusters'), use_container_width=True)
    for alert in alerts:
        st.error(alert)

elif page == "💥 Stress Testing":
    st.header("💥 Stress Testing")
    n_simulations = st.slider("Number of Simulations", 100, 2000, 1000)
    simulated_losses = monte_carlo_simulation(policies, n_simulations)
    st.plotly_chart(px.histogram(simulated_losses, x='total_claims', nbins=50, title='Total Claims Distribution'), use_container_width=True)

elif page == "🔴 Real-Time Dashboard":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #ffe6e6, #ffcccc); padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(255,0,0,0.1);">
            <h2 style="color: #b30000; font-weight: bold;">🔴 Real-Time Claims Dashboard</h2>
    """, unsafe_allow_html=True)

    st_autorefresh(interval=5000, key="auto-refresh")
    new_claims = simulate_real_time_claims()
    total_claims = st.session_state.claims['claim_amount'].sum()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h4 style='color: #660000;'>Live Claims Updates</h4>", unsafe_allow_html=True)
        st.dataframe(new_claims.tail(10), use_container_width=True)
    with col2:
        st.markdown("<h4 style='color: #660000;'>Latest Policies</h4>", unsafe_allow_html=True)
        st.dataframe(policies.tail(10), use_container_width=True)

    st.markdown(f"""
        <div class='metric-box' style="background: #fff5f5; border: 2px solid #ffcccc;">
            <h4 style="color: #800000;">Total Claims Processed</h4>
            <h2 style="color: #cc0000;">₹ {total_claims:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

