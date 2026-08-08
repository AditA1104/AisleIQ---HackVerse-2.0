import streamlit as st
import pandas as pd
import numpy as np
import time
from streamlit_autorefresh import st_autorefresh

# Refresh the dashboard every 1000 milliseconds (1 second)
st_autorefresh(interval=1000, key="datarefresh")

# Page configuration
st.set_page_config(
    page_title="AisleIQ - Retail Analytics & Foot Traffic",
    page_icon="🛒",
    layout="wide"
)

# App Header
st.title("🛒 AisleIQ: Smart Retail Analytics & Heatmapping")
st.markdown("Real-time customer dwell time tracking, zone analytics, and automated inventory insights.")

# Sidebar Controls
st.sidebar.header("Control Panel")
selected_zone = st.sidebar.selectbox(
    "Select Monitored Zone",
    ["Aisle 1 (Snacks & Beverages)", "Aisle 2 (Fresh Produce)", "Aisle 3 (Electronics)", "Checkout Queue"]
)

dwell_threshold = st.sidebar.slider("Dwell Alert Threshold (Minutes)", min_value=1, max_value=15, value=5)

# Main Dashboard Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current Shoppers in Zone", value="12", delta="+3 from last hr")

with col2:
    st.metric(label="Average Dwell Time", value="4.2 mins", delta="-0.5 mins")

with col3:
    st.metric(label="Active Assistance Alerts", value="1", delta="Action Required", delta_color="inverse")

st.markdown("---")

# Section: Live Heatmap / Feed Simulation & Analytics
tab1, tab2 = st.tabs(["📊 Traffic & Dwell Analytics", "🤖 AI Restock & Staffing Insights"])

with tab1:
    st.subheader(f"Traffic Trends for {selected_zone}")
    
    # Generate mock chart data representing foot traffic over hours
    chart_data = pd.DataFrame(
        np.random.randn(20, 3) * 5 + 20,
        columns=['Morning Rush', 'Afternoon Lull', 'Evening Peak']
    )
    st.line_chart(chart_data)
    
    st.info("💡 **Tip:** High dwell times combined with low purchases in specific zones trigger automated notifications for your floor assistants.")

with tab2:
    st.subheader("IBM Granite AI Recommendation Engine")
    
    if st.button("Generate Real-time Inventory Advice"):
        with st.spinner("Analyzing stock levels and foot traffic patterns..."):
            time.sleep(1.5) # Simulate processing delay
            st.success("Analysis Complete!")
            st.write(f"**Recommendation for {selected_zone}:** Foot traffic is up 35% in this section today, primarily driven by organic snack interest. Consider moving secondary stock closer to the aisle entrance to optimize throughput and reduce customer hesitation.")
    else:
        st.markdown("Click the button above to query the AI assistant for live operational strategy based on store metrics.")

# Footer
st.markdown("---")
st.caption("AisleIQ Hackathon Project | Powered by OpenCV, Ultralytics, and Streamlit")