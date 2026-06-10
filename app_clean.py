import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- STYLING & CONFIG ---
st.set_page_config(
    page_title="Enterprise AI & MLOps Control Center", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #0f172a; padding: 20px; border-radius: 10px; border-left: 5px solid #6366f1; color: white; }
    .agent-box { background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; color: #0f172a; font-family: monospace; }
    .offer-badge { background-color: #e0f2fe; color: #0369a1; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    </style>
""", unsafe_allow_html=True)

# --- MOCK DATA ---


@st.cache_data
def get_customer_feature_store():
    return pd.DataFrame({
        "Customer_ID": ["SUB-9821", "SUB-4412", "SUB-3309", "SUB-7762"],
        "Brand_Segment": ["Value Mobile A", "Value Mobile B", "Value Mobile A", "Value Mobile B"],
        "Tenure_Months": [24, 3, 48, 12],
        "Monthly_Data_GB": [45, 4, 62, 18],
        "Autopay_Enabled": ["Yes", "No", "Yes", "Yes"],
        "Device_Type": ["Flagship iOS", "Mid-Tier Android", "Legacy iOS", "Flagship Android"],
        "Estimated_Household_Size": [4, 1, 3, 2],
        "Historical_Sentiment": ["Positive", "Frustrated", "Neutral", "Positive"]
    })


df_features = get_customer_feature_store()

# --- ENHANCEMENT 2: STATE RESET CALLBACK ---


def reset_simulation_state():
    """Wipes old calculation caches when a new user is picked to prevent UI data silos."""
    st.session_state.simulated = False

# --- MULTI-OFFER PROPENSITY MATRIX ENGINE (ENHANCED) ---


def run_portfolio_propensity_engine(customer_data):
    """
    Simulates a true model score computation. It vectorizes features 
    and applies a calculated dot product weighting system.
    """
    initiatives = ["📦 Premium Utility Perk",
                   "🎬 Entertainment Streaming Bundle", "🛡️ Device Protection Plan"]

    # Mathematical weights resembling a logistic scoring coefficient model
    features_vector = np.array([
        customer_data['Tenure_Months'],
        customer_data['Monthly_Data_GB'],
        customer_data['Estimated_Household_Size'],
        1 if customer_data['Autopay_Enabled'] == 'Yes' else 0
    ])

    # Coefficients for individual initiatives
    weights = {
        "📦 Premium Utility Perk": np.array([0.002, 0.010, 0.050, 0.05]),
        "🎬 Entertainment Streaming Bundle": np.array([0.001, 0.005, 0.120, 0.10]),
        "🛡️ Device Protection Plan": np.array([-0.005, 0.002, 0.010, 0.02])
    }

    scores = {}
    for init in initiatives:
        # Dot product calculation simulating programmatic algorithm inference
        raw_score = 0.25 + np.dot(features_vector, weights[init])
        if "🛡️ Device Protection Plan" in init and ("Flagship" in customer_data['Device_Type'] or "iOS" in customer_data['Device_Type']):
            raw_score += 0.35
        scores[init] = min(round(float(raw_score), 2), 0.98)

    top_offer = max(scores, key=scores.get)
    return scores, top_offer, scores[top_offer]

# --- AGENTIC ORCHESTRATION FRAMEWORK ---


def run_agentic_llm_framework(customer_id, brand, top_offer, score, customer_data):
    persona = f"Subscriber on {brand} using a {customer_data['Device_Type']} device. Model identified optimal ARPU lift via {top_offer} allocation based on behavioral parameters."

    if score > 0.65:
        action = f"🚨 ROUTE TO SPECIALIZED SUBSCRIPTION AGENT (Campaign: {top_offer})"
        if "Premium Utility" in top_offer:
            pitch = f"\"I see you've been with {brand} for {customer_data['Tenure_Months']} months. Based on your household's high data needs, we can unlock an exclusive 6-month trial of our Premium Utility Perk, saving you up to $800/year on daily deliveries. Can I activate this on your line today?\""
        elif "Entertainment Streaming" in top_offer:
            pitch = f"\"Thanks for managing a multi-line account with us! To maximize your household value, we can add our Entertainment Streaming Bundle directly to your plan for just an additional $5/month, saving you over 50% compared to retail. Shall we get that set up?\""
        else:
            pitch = f"\"Protecting your premium hardware is critical. Since you're on a flagship device, we can provision our Device Protection Plan onto your account today to guarantee overnight replacements for accidental liquid spills or cracks. Should I secure your device?\""
    else:
        action = "💤 STANDARD ROUTING (Suppress Portfolio Offers / Focus on Support Efficiency)"
        pitch = "Offer Portfolio Suppressed. High risk of average handle time (AHT) inflation without operational justification."

    return persona, action, pitch


# --- HEADER WITH TOOLTIP ---
st.title(
    " 🌐 Telecom Value Team: Enterprise MLOps & Agentic Orchestration",
    help="STRATEGIC SUMMARY:\nEvent-driven architecture designed to maximize portfolio ARPU growth across Value Brands by combining predictive ML scoring with autonomous GenAI workflows."
)
st.write("---")

# --- APPLICATION TABS WITH SYMBOLS ---
tab1, tab2, tab3 = st.tabs([
    "⚡ Live Simulation & Inference",
    "📊 MLOps Pipeline & Drift Control",
    "📈 Financial & Value Realization"
])

# ==========================================
# TAB 1: LIVE SIMULATION
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(
            " 📥 Inbound Call Center Event",
            help="PITCH POINTS:\n- Simulates real-time cloud feature-store lookup instantly upon call connection."
        )
        # ENHANCEMENT 2: Dynamic callback connection added to reset interface properly
        selected_id = st.selectbox(
            "Select Inbound Customer ID", df_features["Customer_ID"], on_change=reset_simulation_state)
        customer_record = df_features[df_features["Customer_ID"]
                                      == selected_id].iloc[0]

        st.write("")
        for col in df_features.columns:
            st.markdown(
                f"**{col.replace('_', ' ')}:** `{customer_record[col]}`")

    with col2:
        st.subheader(
            " ⚙️ Real-Time Engine Execution",
            help="PITCH POINTS:\n- Evaluates a parallel propensity matrix across all high-margin subscription products simultaneously."
        )

        if "simulated" not in st.session_state:
            st.session_state.simulated = False
            st.session_state.all_scores = {}
            st.session_state.top_offer = ""
            st.session_state.prob = 0.0
            st.session_state.agent_persona = ""
            st.session_state.agent_action = ""
            st.session_state.agent_pitch = ""

        if st.button("🚀 SIMULATE INBOUND CALL CONTACT", type="primary"):
            with st.spinner("Processing real-time pipelines..."):
                time.sleep(0.4)
                scores, top_offer, top_score = run_portfolio_propensity_engine(
                    customer_record)
                p, a, pi = run_agentic_llm_framework(
                    selected_id, customer_record['Brand_Segment'], top_offer, top_score, customer_record)

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
            m1.metric(label="🎯 Top Propensity Score",
                      value=f"{int(st.session_state.prob*100)}%", delta=f"Target: {st.session_state.top_offer}")
            m2.metric(label="⏱️ Inference Latency", value="71 ms",
                      delta="-79ms (vs SLA < 150ms)", delta_color="inverse")
            m3.metric(label="🗄️ Model Registry Status",
                      value="v2.2.0-Prod", delta="MLflow Active")

            st.markdown("### 📋 Propensity Evaluation Matrix")
            scores_df = pd.DataFrame(list(st.session_state.all_scores.items()), columns=[
                                     'Subscription Initiative', 'Propensity Score'])
            scores_df['Propensity Score'] = scores_df['Propensity Score'].apply(
                lambda x: f"{int(x*100)}%")
            st.table(scores_df)

            st.markdown("### 🛣️ Next Best Action Routing Decision:")
            if st.session_state.prob > 0.65:
                st.error(st.session_state.agent_action)
            else:
                st.info(st.session_state.agent_action)

            st.markdown("### 🧠 Generative AI Scripting Engine")
            st.markdown(
                f"**🤖 Agent 1 (Behavioral Insight):** <div class='agent-box'>{st.session_state.agent_persona}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<br>**✍️ Agent 2 (Contextual Pitch Copy):** <div class='agent-box' style='border-left: 5px solid #10b981;'>{st.session_state.agent_pitch}</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: MLOPS PIPELINE & DRIFT CONTROL
# ==========================================
with tab2:
    st.header(" 🔍 Continuous Monitoring & Multi-Armed Bandit Orchestration")
    st.write("This pane evaluates architectural stability, concept drift validation, and reinforcement learning routing mechanics.")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(label="🔬 Experimentation Framework",
                  value="Contextual Bandit (Bayesian)", delta="Thompson Sampling Active")
    m_col2.metric(label="📉 Population Baseline Drift (PSI)",
                  value="0.041", delta="Status: Stable (< 0.1)")
    m_col3.metric(label="🎲 Exploration Alpha (Traffic Leak)",
                  value="5%", delta="Auto-Optimizing")

    st.write("---")
    layout_col1, layout_col2 = st.columns([1, 1])

    with layout_col1:
        st.subheader(" 🛡️ Reinforcement Learning Offer Vector Convergence")
        st.markdown("> **Executive Explanation:** Standard A/B testing wastes 50% of traffic on losing offers for weeks. Our Multi-Armed Bandit algorithm monitors conversions in real-time...")

        days = list(range(1, 31))
        utility_share = [40 - (x*0.5) for x in days]
        streaming_share = [30 + (x*1.1) for x in days]
        protection_share = [30 - (x*0.6) for x in days]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=utility_share,
                      name='📦 Premium Utility Vector', line=dict(color='#6366f1', width=3)))
        fig.add_trace(go.Scatter(x=days, y=streaming_share,
                      name='🎬 Entertainment Streaming Bundle', line=dict(color='#10b981', width=3)))
        fig.add_trace(go.Scatter(x=days, y=protection_share,
                      name='🛡️ Device Protection Plan', line=dict(color='#0f172a', width=3)))

        fig.update_layout(title="Automated Routing Shift Matrix", xaxis_title="Day",
                          yaxis_title="Allocated %", height=350, template="seaborn")
        st.plotly_chart(fig, use_container_width=True)

    with layout_col2:
        st.subheader(" 🚨 Production Feature Drift Monitor")
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
    st.header(" 💎 Enterprise ROI & Financial Value Realization Model")
    st.write("---")

    calc_col1, calc_col2 = st.columns([1, 1])

    with calc_col1:
        st.subheader(" 🎛️ Interactive Value Parameter Modeler")

        # ENHANCEMENT 3: Dynamic binding from Customer Store records
        base_calls = 250000
        if st.session_state.simulated:
            # Shift slider default context dynamically depending on selected household size factors
            if customer_record["Estimated_Household_Size"] >= 3:
                base_calls = 450000

        monthly_calls = st.slider(
            "Total Monthly Brand Segment Call Volume", 50000, 1000000, base_calls, step=50000)
        conversion_lift = st.slider(
            "Incremental Subscription Attachment Lift (Via Engine Routing)", 1.0, 10.0, 3.5, step=0.5)
        avg_perk_margin = st.number_input(
            "Average Monthly Net Margin per Attachment Plan ($)", value=6.50)
        aht_cost_savings = st.number_input(
            "Operational Cost per Saved Call Center Second ($)", value=0.02)

        total_incremental_attaches = int(
            monthly_calls * (conversion_lift / 100.0))
        monthly_revenue_lift = total_incremental_attaches * avg_perk_margin
        annual_revenue_lift = monthly_revenue_lift * 12

        suppressed_calls = int(monthly_calls * 0.40)
        seconds_saved = suppressed_calls * 45
        monthly_ops_savings = seconds_saved * aht_cost_savings
        annual_ops_savings = monthly_ops_savings * 12

    with calc_col2:
        st.subheader(" 🏆 Projected Business Optimization Yield")

        rev_1, rev_2 = st.columns(2)
        rev_1.metric(label="💵 Monthly Revenue Growth",
                     value=f"${monthly_revenue_lift:,.2f}", delta=f"+{conversion_lift}% Base Attachment Lift")
        rev_2.metric(label="💰 Annualized Gross Value Realization",
                     value=f"${annual_revenue_lift + annual_ops_savings:,.2f}", delta="Combined Yield", delta_color="normal")

        st.write("---")
        categories = ['Incremental Subscription Rev',
                      'Operational AHT Cost Recovery', 'Total Net Annual Impact']
        amounts = [annual_revenue_lift, annual_ops_savings,
                   annual_revenue_lift + annual_ops_savings]

        fig_bar = go.Figure([go.Bar(x=categories, y=amounts, marker_color=[
                            '#6366f1', '#475569', '#10b981'])])
        fig_bar.update_layout(title="Annual Corporate Value Realization Stack",
                              yaxis_title="Value ($)", height=300, template="simple_white")
        st.plotly_chart(fig_bar, use_container_width=True)
