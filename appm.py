import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- BACKEND AI IMPORTS ---
try:
    from backend import trigger_dwell_time_alert, summarize_daily_trends
except ImportError:
    def trigger_dwell_time_alert(aisle_name="Electronics", dwell_seconds=48):
        return f"🚨 Alert: Customer stalled in {aisle_name} for {dwell_seconds} seconds! Floor assistance recommended."

    def summarize_daily_trends(heatmap_data):
        return (
            f"📊 AI Daily Summary: High shopper dwell observed in {heatmap_data.get('Electronics_Zone', 'N/A')}. "
            f"Queue velocity peaked at {heatmap_data.get('Checkout_Queue', 'N/A')}. Recommended staffing adjustments applied."
        )

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="aisleIQ.",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. INITIALIZE SESSION STATE ---
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"

if "toolbar_open" not in st.session_state:
    st.session_state.toolbar_open = False

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "show_ai_summary" not in st.session_state:
    st.session_state.show_ai_summary = False

# --- 3. DYNAMIC SIDEBAR VISIBILITY & ANIMATION CONTROL ---
st.markdown(
    """
    <style>
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarExpandButton"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

if not st.session_state.splash_done or not st.session_state.toolbar_open:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: block !important;
            min-width: 320px !important;
            max-width: 320px !important;
            animation: slideInToolbar 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

# --- 4. DYNAMIC THEME COLOR PALETTE SELECTION ---
is_subpage = st.session_state.active_tab != "Overview"

if st.session_state.dark_mode:
    bg_color = "#0D1117" if not is_subpage else "#161B22"
    main_text = "#F0F6FC"
    border_col = "#30363D"
    box_shadow_col = "#000000"
    sidebar_bg = "#161B22"
    card_light_bg = "#1F2937"
    card_dark_bg = "#111827"
    accent_brand = "#58A6FF"
    btn_bg = "#21262D"
    splash_bg = "radial-gradient(circle at center, #21262D 0%, #0D1117 100%)"
else:
    bg_color = "#A8D0E6" if is_subpage else "#F3F0E8"
    main_text = "#030B33"
    border_col = "#030B33"
    box_shadow_col = "#030B33"
    sidebar_bg = "#030B33"
    card_light_bg = "#D4F4F7"
    card_dark_bg = "#030B33"
    accent_brand = "#007A93"
    btn_bg = "#030B33"
    splash_bg = "radial-gradient(circle at center, #F9F7F1 0%, #F3F0E8 100%)"

app_animation = "none !important"
bg_transition = "none !important" if is_subpage else "background-color 0.4s ease"

# --- 5. CUSTOM STYLING & ANIMATIONS ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@400;500&display=swap');

    @keyframes slideInToolbar {{
        0% {{ transform: translateX(-100%); opacity: 0; }}
        100% {{ transform: translateX(0); opacity: 1; }}
    }}

    @keyframes logoPulseGlow {{
        0%, 100% {{ 
            box-shadow: 10px 10px 0px {border_col}, 0 0 0px rgba(0, 122, 147, 0); 
            transform: translateY(0px);
        }}
        50% {{ 
            box-shadow: 14px 14px 0px {border_col}, 0 0 30px rgba(34, 157, 176, 0.45); 
            transform: translateY(-4px);
        }}
    }}

    @keyframes badgeSlideUp {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    .stApp {{
        background-color: {bg_color} !important;
        color: {main_text} !important;
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 1.15rem;
        transition: {bg_transition};
    }}

    .main .block-container {{
        animation: {app_animation};
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    h1, h2, h3, h4, label, p, span {{
        color: {main_text} !important;
    }}

    h1, h2, h3, h4 {{
        font-weight: 600 !important;
    }}

    /* COMPACT POP-UP TOAST NOTIFICATION WITH WHITE TEXT */
    div[data-testid="stToast"] {{
        background-color: #161B22 !important;
        border: 2px solid {accent_brand} !important;
        border-radius: 10px !important;
        box-shadow: 3px 3px 0px {box_shadow_col} !important;
        padding: 6px 14px !important;
        max-width: 280px !important;
        min-height: 42px !important;
    }}

    div[data-testid="stToast"] * {{
        color: #FFFFFF !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.88rem !important;
        line-height: 1.25 !important;
    }}

    /* ALL TOOLBAR ARROWS & DROPDOWNS FORCED TO WHITE */
    section[data-testid="stSidebar"] svg,
    section[data-testid="stSidebar"] svg path,
    section[data-testid="stSidebar"] svg circle,
    section[data-testid="stSidebar"] svg polygon,
    section[data-testid="stSidebar"] svg g,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg path,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg path {{
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }}

    /* LIGHT HAMBURGER MENU BUTTON WITH DARK LINES */
    div[data-testid="stColumn"] button[key="open_toolbar_btn"] {{
        background: #F3F0E8 !important;
        color: #030B33 !important;
        border: 3px solid #030B33 !important;
        border-radius: 12px !important;
        padding: 0px !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        box-shadow: 4px 4px 0px {box_shadow_col} !important;
        width: 52px !important;
        height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        margin-bottom: 20px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    div[data-testid="stColumn"] button[key="open_toolbar_btn"] p {{
        color: #030B33 !important;
        font-weight: 800 !important;
    }}

    div[data-testid="stColumn"] button[key="open_toolbar_btn"]:hover {{
        background: #FFFFFF !important;
        color: #000000 !important;
        box-shadow: 6px 6px 0px {box_shadow_col} !important;
        transform: translate(-2px, -2px);
    }}

    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 4px solid {accent_brand};
        padding-top: 20px;
    }}

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
        font-family: 'Poppins', sans-serif;
    }}

    section[data-testid="stSidebar"] [data-testid="stExpander"] {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid {accent_brand} !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }}

    section[data-testid="stSidebar"] summary p {{
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #FFFFFF !important;
    }}

    .sidebar-brand {{
        font-size: 1.8rem;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        padding-bottom: 10px;
        border-bottom: 2px solid {accent_brand};
        margin-bottom: 20px;
        letter-spacing: 0.05em;
    }}

    .splash-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 78vh;
        text-align: center;
        background: {splash_bg};
        border: 4px solid {border_col};
        border-radius: 32px;
        padding: 40px;
        box-shadow: 12px 12px 0px {box_shadow_col};
        position: relative;
        overflow: hidden;
    }}

    .splash-logo {{
        font-family: 'Poppins', sans-serif;
        font-size: 5.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, {accent_brand} 0%, {sidebar_bg} 100%);
        padding: 26px 68px;
        border: 4px solid {border_col};
        border-radius: 28px;
        animation: logoPulseGlow 2.5s ease-in-out infinite;
        margin-bottom: 28px;
    }}

    .splash-subtitle {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFFFFF !important;
        background: {sidebar_bg};
        padding: 12px 32px;
        border: 3px solid {border_col};
        border-radius: 16px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        box-shadow: 4px 4px 0px {accent_brand};
        animation: badgeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.25s backwards;
    }}

    div[data-testid="stProgress"] > div > div {{
        background-color: {border_col} !important;
        border: 2px solid {border_col} !important;
        border-radius: 10px !important;
        height: 14px !important;
    }}

    div[data-testid="stProgress"] > div > div > div {{
        background: linear-gradient(90deg, {accent_brand}, #229DB0) !important;
        border-radius: 8px !important;
    }}

    .app-header-clean {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 10px 0px 24px 0px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .app-brand-clean {{
        font-family: 'Poppins', sans-serif;
        font-size: 4.8rem;
        font-weight: 700;
        color: {main_text} !important;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1;
    }}

    div.stButton > button[key="live_feed_redirect_btn"] {{
        background: linear-gradient(135deg, #FF3B30 0%, #B71C1C 100%) !important;
        color: #FFFFFF !important;
        padding: 12px 28px !important;
        border-radius: 40px !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        border: 2px solid #5A1818 !important;
        box-shadow: 3px 3px 0px {box_shadow_col} !important;
        margin-bottom: 0px !important;
    }}

    div.stButton > button[key="live_feed_redirect_btn"]:hover {{
        background: linear-gradient(135deg, #E53935 0%, #880E4F 100%) !important;
        color: #FFFFFF !important;
        transform: translate(-2px, -2px) !important;
        box-shadow: 5px 5px 0px {box_shadow_col} !important;
    }}

    .card-green-light {{
        background-color: {card_light_bg} !important;
        border: 3px solid {border_col};
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 20px;
        box-shadow: 6px 6px 0px {box_shadow_col};
        color: {main_text} !important;
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .card-green-light:hover {{
        transform: translateY(-2px);
        box-shadow: 8px 8px 0px {box_shadow_col};
    }}

    .card-green-dark {{
        background-color: {card_dark_bg} !important;
        border: 3px solid {border_col};
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 20px;
        box-shadow: 6px 6px 0px {box_shadow_col};
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .card-green-dark,
    .card-green-dark *,
    .card-green-dark .card-title,
    .card-green-dark .card-value,
    .card-green-dark .card-subtext,
    .card-green-dark h1,
    .card-green-dark p {{
        color: #FFFFFF !important;
    }}

    .card-green-dark:hover {{
        transform: translateY(-2px);
        box-shadow: 8px 8px 0px {box_shadow_col};
    }}

    .card-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .card-value {{
        font-family: 'Poppins', sans-serif;
        font-size: 3.6rem;
        font-weight: 600;
        line-height: 1.1;
        margin-bottom: 6px;
    }}

    .card-subtext {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.15rem;
        font-weight: 500;
    }}

    .stButton > button {{
        border-radius: 16px !important;
        border: 3px solid {border_col} !important;
        background: {btn_bg} !important;
        color: #FFFFFF !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        padding: 16px 28px !important;
        box-shadow: 5px 5px 0px {box_shadow_col} !important;
        transition: all 0.15s ease-in-out !important;
        margin-bottom: 20px !important;
    }}

    .stButton > button,
    .stButton > button * {{
        color: #FFFFFF !important;
    }}

    .stButton > button:hover {{
        background: {accent_brand} !important;
        color: #FFFFFF !important;
        box-shadow: 8px 8px 0px {box_shadow_col} !important;
        transform: translate(-2px, -2px);
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# ANIMATED SPLASH SCREEN
# ==========================================
if not st.session_state.splash_done:
    st.markdown(
        """
        <div class="splash-container">
            <div class="splash-logo">aisleIQ.</div>
            <div class="splash-subtitle">⚡ Retail Intelligence Engine</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

    st.session_state.splash_done = True
    st.rerun()

# ==========================================
# SIDEBAR TOOLBAR
# ==========================================
with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'>TOOLBAR</div>", unsafe_allow_html=True
    )

    with st.expander("🎨 THEME & DISPLAY", expanded=False):
        dark_mode_toggle = st.toggle(
            "🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_switch"
        )
        if dark_mode_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode_toggle
            st.rerun()

    with st.expander("🛠️ MODULE SWITCHER", expanded=False):
        nav_selection = st.radio(
            "Select View",
            ["Overview", "Heatmap Analytics", "Staff Dispatch", "Live Tracking"],
            index=[
                "Overview",
                "Heatmap Analytics",
                "Staff Dispatch",
                "Live Tracking",
            ].index(
                {
                    "Overview": "Overview",
                    "Heatmap": "Heatmap Analytics",
                    "Dispatch": "Staff Dispatch",
                    "Live Tracking": "Live Tracking",
                }.get(st.session_state.active_tab, "Overview")
            ),
        )

        if nav_selection == "Overview" and st.session_state.active_tab != "Overview":
            st.session_state.active_tab = "Overview"
            st.rerun()
        elif (
            nav_selection == "Heatmap Analytics"
            and st.session_state.active_tab != "Heatmap"
        ):
            st.session_state.active_tab = "Heatmap"
            st.rerun()
        elif (
            nav_selection == "Staff Dispatch"
            and st.session_state.active_tab != "Dispatch"
        ):
            st.session_state.active_tab = "Dispatch"
            st.rerun()
        elif (
            nav_selection == "Live Tracking"
            and st.session_state.active_tab != "Live Tracking"
        ):
            st.session_state.active_tab = "Live Tracking"
            st.rerun()

    with st.expander("⚙️ QUICK CONTROLS", expanded=False):
        st.toggle("Auto-Refresh Feed", value=True)
        st.selectbox(
            "Floor Zone", ["Zone A (Snacks)", "Zone B (Produce)", "Zone C (Bakery)"]
        )
        st.slider(
            "Alert Sensitivity", 10, 60, 45, help="Dwell duration trigger in seconds"
        )

    st.markdown("---")
    st.markdown(
        "<div style='color: #FFFFFF !important; font-weight: 600;'>STATUS:"
        " ONLINE</div>",
        unsafe_allow_html=True,
    )

# ==========================================
# HAMBURGER MENU BUTTON ☰
# ==========================================
col_btn, col_empty = st.columns([1, 15])
with col_btn:
    if st.button("☰", key="open_toolbar_btn", help="Toggle Menu"):
        st.session_state.toolbar_open = not st.session_state.toolbar_open
        st.rerun()

# ==========================================
# DYNAMIC MAIN HEADER
# ==========================================
if st.session_state.active_tab == "Overview":
    header_title = "aisleIQ."
elif st.session_state.active_tab == "Heatmap":
    header_title = "Heatmap Analytics"
elif st.session_state.active_tab == "Dispatch":
    header_title = "Staff Dispatch"
else:
    header_title = "Live Tracking"

# SHOW LIVE FEED BUTTON ONLY ON OVERVIEW PAGE
if st.session_state.active_tab == "Overview":
    col_title, col_live_btn = st.columns([4, 1])
    with col_title:
        st.markdown(
            f'<div class="app-header-clean"><span class="app-brand-clean">{header_title}</span></div>',
            unsafe_allow_html=True,
        )
    with col_live_btn:
        st.markdown(
            '<div style="padding-top: 15px;"></div>', unsafe_allow_html=True
        )
        if st.button("● LIVE FEED", key="live_feed_redirect_btn"):
            st.session_state.active_tab = "Live Tracking"
            st.rerun()
else:
    st.markdown(
        f'<div class="app-header-clean"><span class="app-brand-clean">{header_title}</span></div>',
        unsafe_allow_html=True,
    )

# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
if st.session_state.active_tab == "Overview":

    customer_is_confused = True

    if customer_is_confused:
        alert_text = trigger_dwell_time_alert(
            aisle_name="Electronics", dwell_seconds=48
        )
        st.warning(alert_text)
        st.toast(alert_text, icon="🚨")

    st.markdown("### 🗂️ SELECT APP MODULE")
    n1, n2 = st.columns(2)
    with n1:
        if st.button(
            "📹 OPEN HEATMAP ANALYTICS",
            key="nav_heatmap_btn",
            use_container_width=True,
        ):
            st.session_state.active_tab = "Heatmap"
            st.rerun()
    with n2:
        if st.button(
            "🚨 OPEN STAFF DISPATCH",
            key="nav_dispatch_btn",
            use_container_width=True,
        ):
            st.session_state.active_tab = "Dispatch"
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("# 📌 LIVE FLOOR METRICS")

    st.markdown(
        """
        <div class="card-green-light">
            <div class="card-title">👥 ACTIVE SHOPPERS IN STORE</div>
            <div class="card-value">14</div>
            <div class="card-subtext">📈 +2 vs 5m ago</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card-green-dark">
            <div class="card-title">🚨 STALLED SHOPPERS ALERT</div>
            <div class="card-value">1 Alert</div>
            <div class="card-subtext">⚠️ Requires Floor Staff</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card-green-light">
            <div class="card-title">⏱️ AVERAGE DWELL TIME</div>
            <div class="card-value">38 Secs</div>
            <div class="card-subtext">📉 -4s vs Target</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card-green-dark">
            <div class="card-title">🔥 TOP HOTSPOT ZONE</div>
            <div class="card-value">Shelf B</div>
            <div class="card-subtext">📊 82% Focus Share</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("# 💡 EXECUTIVE INSIGHTS")

    if not st.session_state.show_ai_summary:
        if st.button("Generate Daily AI Summary", key="ai_summary_trigger_btn"):
            st.session_state.show_ai_summary = True
            st.rerun()
    else:
        mock_heatmap_data = {
            "Electronics_Zone": "52 mins total dwell",
            "Checkout_Queue": "Peak 6 people at 2 PM",
        }
        summary = summarize_daily_trends(mock_heatmap_data)
        st.info(summary)

        if st.button("❌ Hide Daily AI Summary", key="hide_ai_summary_btn"):
            st.session_state.show_ai_summary = False
            st.rerun()

# ==========================================
# TAB 2: HEATMAP ANALYTICS
# ==========================================
elif st.session_state.active_tab == "Heatmap":
    if st.button("⬅️ BACK TO OVERVIEW", key="back_from_heatmap"):
        st.session_state.active_tab = "Overview"
        st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    selected_camera = st.selectbox(
        "Active Feed Selector",
        [
            "Aisle 2 - Electronics Zone",
            "Aisle 1 - Fresh Produce",
            "Aisle 3 - Personal Care",
        ],
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="card-green-light">
            <h1 style="font-family: 'Poppins', sans-serif; font-size: 2rem; margin-top: 0; font-weight: 600; color: {main_text};">Spatial Footstep Density</h1>
            <p style="font-size: 1.2rem; margin-bottom: 0; color: {main_text};">Real-time camera depth grid showing dwell hotspots.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    np.random.seed(12)
    grid_data = np.random.exponential(scale=12, size=(10, 14))
    grid_data[3:6, 5:9] += 50

    fig_map = px.imshow(
        grid_data,
        labels=dict(x="Aisle Width", y="Shelf Depth", color="Dwell (s)"),
        color_continuous_scale=[
            "#D4F4F7",
            "#A2E2E8",
            "#229DB0",
            "#007A93",
            "#030B33",
        ],
    )
    fig_map.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=main_text, family="Poppins", size=16),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ==========================================
# TAB 3: STAFF DISPATCH
# ==========================================
elif st.session_state.active_tab == "Dispatch":
    if st.button("⬅️ BACK TO OVERVIEW", key="back_from_dispatch"):
        st.session_state.active_tab = "Overview"
        st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.error(
        "🚨 **ACTIVE ALERT:** Customer stalled in Electronics Zone needs floor assistance!"
    )

    st.markdown(
        f"""
        <div class="card-green-dark">
            <h1 style="font-size: 2.2rem; margin-top: 0; font-weight: 600; color: #FFFFFF !important;">📱 AUTOMATED FLOOR BROADCAST</h1>
            <p style="margin-bottom: 20px; color: #FFFFFF !important;">Message dispatched to floor assistant handset:</p>
            <div style="background: {btn_bg}; color: #FFFFFF !important; border: 3px solid {border_col}; border-radius: 14px; padding: 24px; font-size: 1.3rem; font-weight: 600; box-shadow: 4px 4px 0px {box_shadow_col};">
                "Shopper stalled in <strong style='color: #FFFFFF !important;'>Electronics Zone</strong> for <strong style='color: #FFFFFF !important;'>48s</strong>. Please check pricing display."
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("✅ MARK INCIDENT AS RESOLVED", use_container_width=True):
        st.success("Incident marked resolved and logged in store operations ledger!")

# ==========================================
# TAB 4: LIVE TRACKING
# ==========================================
elif st.session_state.active_tab == "Live Tracking":
    if st.button("⬅️ BACK TO OVERVIEW", key="back_from_livetracking"):
        st.session_state.active_tab = "Overview"
        st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="background: {btn_bg}; border: 3px dashed {border_col}; border-radius: 20px; height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #FFFFFF; text-align: center; padding: 20px; box-shadow: 6px 6px 0px {box_shadow_col};">
            <div style="font-size: 4rem; margin-bottom: 12px;">📹</div>
            <div style="font-size: 1.8rem; font-weight: 600;">Video Stream Placeholder</div>
            <div style="font-size: 1.15rem; color: #A8D0E6; margin-top: 8px;">Insert your teammate's CCTV video tracking code here later.</div>
        </div>
    """,
        unsafe_allow_html=True,
    )