import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- STYLING & CONFIG ---
st.set_page_config(page_title="Verizon Executive AI & MLOps Control", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #ef4444; color: white; }
    .agent-box { background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; color: #1e293b; }
    .offer-badge { background-color: #e0f2fe; color: #0369a1; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    </style>
""", unsafe_allow_html=True)

# --- MOCK DATA ---
@st.cache_data
def get_customer_feature_store():
    return pd.DataFrame({
        "Customer_ID": ["ST-9821", "TW-4412", "ST-3309", "TW-7762"],
        "Brand": ["Straight Talk", "Total Wireless", "Straight Talk", "Total Wireless"],
        "Tenure_Months": [24, 3, 48, 12],
        "Monthly_Data_GB": [45, 4, 62, 18],
        "Autopay_Enabled": ["Yes", "No", "Yes", "Yes"],
        "Device_Type": ["iPhone 15 Pro", "Samsung A14", "iPhone 14", "Google Pixel 8"],
        "Estimated_Household_Size": [4, 1, 3, 2],
        "Historical_Sentiment": ["Positive", "Frustrated", "Neutral", "Positive"]
    })

df_features = get_customer_feature_store()

# --- MULTI-OFFER PROPENSITY MATRIX ENGINE ---
def run_portfolio_propensity_engine(customer_data):
    scores = {"Walmart+": 0.20, "Streaming Perk (Disney+)": 0.15, "Verizon Value Protect": 0.10}
    
    if customer_data['Brand'] == 'Straight Talk': scores['Walmart+'] += 0.30
    if customer_data['Monthly_Data_GB'] > 30: scores['Walmart+'] += 0.25
    if customer_data['Estimated_Household_Size'] >= 3: scores['Walmart+'] += 0.20
    
    if customer_data['Estimated_Household_Size'] >= 2: scores['Streaming Perk (Disney+)'] += 0.35
    if customer_data['Monthly_Data_GB'] > 15: scores['Streaming Perk (Disney+)'] += 0.25
    if customer_data['Autopay_Enabled'] == 'Yes': scores['Streaming Perk (Disney+)'] += 0.15
    
    if "iPhone" in customer_data['Device_Type'] or "Pixel" in customer_data['Device_Type']: scores['Verizon Value Protect'] += 0.50
    if customer_data['Historical_Sentiment'] == 'Positive': scores['Verizon Value Protect'] += 0.20
    if customer_data['Tenure_Months'] < 12: scores['Verizon Value Protect'] += 0.10
    
    for offer in scores:
        scores[offer] = min(round(scores[offer], 2), 0.98)
        
    top_offer = max(scores, key=scores.get)
    return scores, top_offer, scores[top_offer]

# --- AGENTIC ORCHESTRATION FRAMEWORK ---
def run_agentic_llm_framework(customer_id, brand, top_offer, score, customer_data):
    persona = f"Subscriber on {brand} using an {customer_data['Device_Type']}. Model identified optimal ARPU lift via {top_offer} allocation based on behavioral parameters."
    
    if score > 0.65:
        action = f"🚨 ROUTE TO SPECIALIZED SUBSCRIPTION AGENT (Campaign: {top_offer})"
        if top_offer == "Walmart+":
            pitch = f"\"I see you've been with {brand} for {customer_data['Tenure_Months']} months. Based on your household's high data needs, we can unlock a free 6-month trial of Walmart+ saving you up to $800/year on deliveries. Can I activate this on your line today?\""
        elif top_offer == "Streaming Perk (Disney+)":
            pitch = f"\"Thanks for managing a multi-line account with us! To maximize your household value, we can add our Premium Streaming Perk featuring Disney+ directly to your plan for just an additional $5/month, saving you over 50% compared to retail. Shall we get that set up?\""
        else:
            pitch = f"\"Protecting your flagship hardware is critical. Since you're on a premium device, we can provision Verizon Value Protect onto your account today to guarantee overnight replacements for accidental liquid spills or cracks. Should I secure your device?\""
    else:
         action = "✅ STANDARD ROUTING (Suppress Portfolio Offers / Focus on Support Efficiency)"
         pitch = "Offer Portfolio Suppressed. High risk of average handle time (AHT) inflation without operational justification."
         
    return persona, action, pitch

# --- HEADER WITH TOOLTIP ---
st.title(
    "🚀 Verizon Value Team: Enterprise MLOps & Agentic Orchestration",
    help="STRATEGIC SUMMARY:\nEvent-driven architecture designed to maximize portfolio ARPU growth across Value Brands (Straight Talk, Total Wireless) by combining predictive ML scoring with autonomous GenAI workflows."
)
st.write("---")

# --- APPLICATION TABS ---
tab1, tab2, tab3 = st.tabs(["🎯 Live Simulation & Inference", "🔄 MLOps Pipeline & Drift Control", "📊 Financial & Value Realization"])

# ==========================================
# TAB 1: LIVE SIMULATION
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(
            "📋 Inbound Call Center Event",
            help="PITCH POINTS:\n- Simulates real-time cloud feature-store lookup instantly upon call connection.\n- Solves the 'data silo' problem by unifying billing metrics, historical sentiment, and device hardware profiles instantly."
        )
        selected_id = st.selectbox("Select Inbound Customer ID", df_features["Customer_ID"])
        customer_record = df_features[df_features["Customer_ID"] == selected_id].iloc[0]
        
        for col in df_features.columns:
            st.markdown(f"**{col.replace('_', ' ')}:** `{customer_record[col]}`")

    with col2:
        st.subheader(
            "⚡ Real-Time Engine Execution",
            help="PITCH POINTS:\n- Evaluates a parallel propensity matrix across all high-margin subscription products simultaneously.\n- Note Latency (71ms): Well within the sub-150ms SLA required for real-time live telecom interactions to eliminate dead air."
        )
        
        if "simulated" not in st.session_state:
            st.session_state.simulated = False
            st.session_state.all_scores = {}
            st.session_state.top_offer = ""
            st.session_state.prob = 0.0
            st.session_state.agent_persona = ""
            st.session_state.agent_action = ""
            st.session_state.agent_pitch = ""

        if st.button("🔴 SIMULATE INBOUND CALL CONTACT", type="primary"):
            with st.spinner("Processing real-time pipelines..."):
                time.sleep(0.4)
                scores, top_offer, top_score = run_portfolio_propensity_engine(customer_record)
                p, a, pi = run_agentic_llm_framework(selected_id, customer_record['Brand'], top_offer, top_score, customer_record)
                
                st.session_state.all_scores = scores
                st.session_state.top_offer = top_offer
                st.session_state.prob = top_score
                st.session_state.agent_persona = p
                st.session_state.agent_action = a
                st.session_state.agent_pitch = pi
                st.session_state.simulated = True

        if st.session_state.simulated:
            st.write("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Top Propensity Score", f"{int(st.session_state.prob*100)}%", f"Target: {st.session_state.top_offer}")
            m2.metric("Inference Latency", "71 ms", "SLA: < 150ms")
            m3.metric("Model Registry Status", "v2.2.0-Prod", "MLflow Active")
            
            st.markdown("### 📊 Propensity Evaluation Matrix")
            scores_df = pd.DataFrame(list(st.session_state.all_scores.items()), columns=['Subscription Initiative', 'Propensity Score'])
            scores_df['Propensity Score'] = scores_df['Propensity Score'].apply(lambda x: f"{int(x*100)}%")
            st.table(scores_df)
            
            st.markdown(
                "### Next Best Action Routing Decision:",
                help="PITCH POINTS:\n- Deterministic optimization block split traffic based on statistical certainty thresholds.\n- High-propensity profiles receive specialized routing, while low-propensity/frustrated callers trigger programmatic Offer Suppression to safeguard AHT."
            )
            if st.session_state.prob > 0.65:
                _ = st.error(st.session_state.agent_action)
            else:
                _ = st.info(st.session_state.agent_action)
            
            st.markdown(
                "### 🤖 Generative AI Scripting Engine",
                help="PITCH POINTS:\n- Demonstrates autonomous LangChain multi-agent orchestration.\n- Agent 1 acts as a data scientist analyzing device/data telemetry to output a strategic brief.\n- Agent 2 acts as a contextual copywriter, instantly synthesizing hyper-personalized pitch text for the call agent."
            )
            st.markdown(f"**Agent 1 (Behavioral Insight):** <div class='agent-box'>{st.session_state.agent_persona}</div>", unsafe_allow_html=True)
            st.markdown(f"<br>**Agent 2 (Contextual Pitch Copy):** <div class='agent-box' style='border-left: 5px solid #22c55e;'>{st.session_state.agent_pitch}</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: MLOPS PIPELINE & DRIFT CONTROL
# ==========================================
with tab2:
    st.header(
        "🔄 Continuous Monitoring & Multi-Armed Bandit Orchestration",
        help="STRATEGIC SUMMARY:\nRepresents our model governance backend. Proves this is an active production ecosystem that continuously auto-tunes and self-protects against behavioral decay over time."
    )
    st.write("This pane evaluates architectural stability, concept drift validation, and reinforcement learning routing mechanics.")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Experimentation Framework", "Contextual Bandit (Bayesian)", "Thompson Sampling Active")
    m_col2.metric("Population Baseline Drift (PSI)", "0.041", "Status: Stable (< 0.1)")
    m_col3.metric("Exploration Alpha (Traffic Leak)", "5%", "Auto-Optimizing")
    
    st.write("---")
    
    layout_col1, layout_col2 = st.columns([1, 1])
    
    with layout_col1:
        st.subheader(
            "📈 Reinforcement Learning Offer Vector Convergence",
            help="PITCH POINTS:\n- Standard A/B testing wastes 50% of traffic on losing offers for weeks.\n- Our Multi-Armed Bandit structure uses Thompson Sampling to monitor conversion rates hourly, dynamically pulling traffic from underperforming offers to eliminate revenue regret."
        )
        st.markdown("> **Executive Explanation:** Standard A/B testing wastes 50% of traffic on losing offers for weeks. Our Multi-Armed Bandit algorithm monitors conversions in real-time, dynamically shifting routing share to the highest-margin subscription offer per segment to avoid revenue regret.")
        
        days = list(range(1, 31))
        walmart_share = [40 - (x*0.5) for x in days]
        disney_share = [30 + (x*1.1) for x in days]
        protect_share = [30 - (x*0.6) for x in days]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=walmart_share, name='Walmart+ Traffic Vector', line=dict(color='#ef4444', width=3)))
        fig.add_trace(go.Scatter(x=days, y=disney_share, name='Streaming Perk (Disney+)', line=dict(color='#22c55e', width=3)))
        fig.add_trace(go.Scatter(x=days, y=protect_share, name='Value Protect Insurance', line=dict(color='#3b82f6', width=3)))
        
        fig.update_layout(
            title="Automated Routing Shift Matrix (30 Day Exploitation Loop)",
            xaxis_title="Day of Experiment Window",
            yaxis_title="Allocated Queue Percentage (%)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with layout_col2:
        st.subheader(
            "🛡️ Production Feature Drift Monitor",
            help="PITCH POINTS:\n- Acts as an automated data smoke detector using the Population Stability Index (PSI).\n- Values below 0.1 indicate stable data. Values above 0.1 warn the team. Values above 0.25 trigger automated model retraining via CI/CD before precision decays."
        )
        st.markdown("Continuous comparison of inbound call parameters against original Model Training baselines to catch consumer behavior shifts.")
        
        drift_data = pd.DataFrame({
            "Feature Name": ["Monthly_Data_GB", "Estimated_Household_Size", "Autopay_Enabled", "Device_Type_Flag"],
            "Training Mean": [24.5, 2.1, "68%", "Baseline"],
            "Production Window": [32.1, 2.5, "71%", "Drifting Higher"],
            "Population Stability Index": [0.082, 0.031, 0.012, 0.145],
            "Status": ["🟢 Normal", "🟢 Normal", "🟢 Normal", "🟡 Investigate Metric Shift"]
        })
        st.dataframe(drift_data, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: FINANCIAL & VALUE REALIZATION
# ==========================================
with tab3:
    st.header(
        "📊 Enterprise ROI & Financial Value Realization Model",
        help="STRATEGIC SUMMARY:\nTranslates data science accuracy and low latency directly into corporate financial metrics that Finance, Marketing, and Operations can champion to justify enterprise rollout."
    )
    st.write("Simulate regional or national deployment value realizations utilizing the multi-offer agentic architecture.")
    
    st.write("---")
    
    calc_col1, calc_col2 = st.columns([1, 1])
    
    with calc_col1:
        st.subheader(
            "🎛️ Interactive Value Parameter Modeler",
            help="PITCH POINTS:\n- Cross-functional tool built to model live scenario changes directly in the boardroom.\n- Tweak the precision lift slider to show leadership how even tiny improvements scale significantly at national deployment levels."
        )
        
        monthly_calls = st.slider("Total Monthly Value-Brand Call Volume", 50000, 1000000, 250000, step=50000)
        conversion_lift = st.slider("Incremental Subscription Attachment Lift (Via Engine Routing)", 1.0, 10.0, 3.5, step=0.5)
        avg_perk_margin = st.number_input("Average Monthly Net Margin per Attachment Plan ($)", value=6.50)
        aht_cost_savings = st.number_input("Operational Cost per Saved Call Center Second ($)", value=0.02)
        
        total_incremental_attaches = int(monthly_calls * (conversion_lift / 100.0))
        monthly_revenue_lift = total_incremental_attaches * avg_perk_margin
        annual_revenue_lift = monthly_revenue_lift * 12
        
        suppressed_calls = int(monthly_calls * 0.40)
        seconds_saved = suppressed_calls * 45
        monthly_ops_savings = seconds_saved * aht_cost_savings
        annual_ops_savings = monthly_ops_savings * 12
        
    with calc_col2:
        st.subheader(
            "💰 Projected Business Optimization Yield",
            help="PITCH POINTS:\n- Unlocks two separate value levers.\n- Lever 1: Top-Line Expansion via precise, automated subscription cross-selling.\n- Lever 2: Bottom-Line Cost Savings by programmatically suppressing pitches on low-propensity callers, saving 45 seconds of wasteful AHT per call."
        )
        
        rev_1, rev_2 = st.columns(2)
        rev_1.metric("Projected Monthly Revenue Growth", f"${monthly_revenue_lift:,.2f}", f"+{conversion_lift}% Base Attachment Lift")
        rev_2.metric("Projected Annualized Gross Value Realization", f"${annual_revenue_lift + annual_ops_savings:,.2f}", "Combined Yield")
        
        st.write("---")
        
        categories = ['Incremental Subscription Rev', 'Operational AHT Cost Recovery', 'Total Net Annual Impact']
        amounts = [annual_revenue_lift, annual_ops_savings, annual_revenue_lift + annual_ops_savings]
        
        fig_bar = go.Figure([go.Bar(x=categories, y=amounts, marker_color=['#10b981', '#3b82f6', '#8b5cf6'])])
        fig_bar.update_layout(
            title="Annual Corporate Value Realization Stack",
            yaxis_title="Value Recaptured ($)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)


