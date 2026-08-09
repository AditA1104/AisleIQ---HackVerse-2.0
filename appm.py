import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- BACKEND AI IMPORTS ---
# NOTE: aisletracker.py (tracker) and this dashboard run as SEPARATE
# processes, so backend.RECENT_ALERTS_LOG is NOT shared between them.
# We must use load_recent_alerts_from_disk() (reads alerts_log.json),
# NOT get_live_alerts() (in-memory, same-process only) — otherwise real
# alerts fired by the tracker never show up here.
try:
    from backend import summarize_daily_trends, load_recent_alerts_from_disk
except ImportError:
    def summarize_daily_trends(traffic_data: dict) -> str:
        return (
            f"AI Daily Summary: High shopper dwell observed in {traffic_data.get('Electronics_Zone', 'N/A')}. "
            f"Queue velocity peaked at {traffic_data.get('Checkout_Queue', 'N/A')}. Recommended staffing adjustments applied."
        )

    def load_recent_alerts_from_disk():
        return []

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="aisleIQ.",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NAVIGATION MAP & STATE SYNC HELPER ---
TAB_MAP = {
    "Overview": "Dashboard",
    "Heatmap": "Heatmap Analytics",
    "Dispatch": "Staff Dispatch",
    "Live Tracking": "Live Tracking",
    "Notifications": "Notifications",
}
REVERSE_TAB_MAP = {v: k for k, v in TAB_MAP.items()}

def navigate_to(tab_key: str):
    """Centralized navigation helper that sets active tab and triggers rerun."""
    st.session_state.active_tab = tab_key
    st.rerun()

def on_nav_dropdown_change():
    """Callback triggered when the user manually changes the sidebar selectbox."""
    selected_label = st.session_state.nav_select_dropdown
    st.session_state.active_tab = REVERSE_TAB_MAP.get(selected_label, "Overview")

# --- 2. INITIALIZE SESSION STATE & SYNC BEFORE WIDGET INSTANTIATION ---
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"

# Keep selectbox state synced BEFORE the sidebar widget is rendered
st.session_state.nav_select_dropdown = TAB_MAP.get(st.session_state.active_tab, "Dashboard")

if "toolbar_open" not in st.session_state:
    st.session_state.toolbar_open = False

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "show_ai_summary" not in st.session_state:
    st.session_state.show_ai_summary = False

# Timestamp-based dedup for toast pop-ups (NOT index/count-based). Alerts
# come from a JSON file that backend.py trims to the last 50 entries, so
# an index/count offset would silently go stale once trimming kicks in.
# Seeded with whatever's already in alerts_log.json at page-load time so
# a fresh session doesn't toast-spam every pre-existing alert on open —
# only alerts that arrive AFTER this session starts get toasted.
# One-time initialization for all disk-derived session state. Reading the
# alert log once here (instead of separately per variable) avoids extra
# file I/O and keeps all three "what already existed at page-load time"
# seeds consistent with each other.
if "toasted_alert_timestamps" not in st.session_state:
    _alerts_at_load = load_recent_alerts_from_disk()

    # Don't toast-spam every alert already in alerts_log.json from earlier
    # testing/tracker runs — only toast alerts that arrive AFTER this
    # session starts.
    st.session_state.toasted_alert_timestamps = {
        a.get("timestamp") for a in _alerts_at_load
    }

    # Same problem for the notification bell badge and the Dispatch page's
    # "ACTIVE ALERT" banner: without seeding these, every pre-existing
    # alert counts as unread/active the instant the page loads. Seed both
    # cutoffs to the newest pre-existing timestamp so a fresh session
    # starts clean — a timestamp cutoff (not a list index/count) survives
    # backend.py trimming the log to its last 50 entries.
    _latest_existing_ts = max(
        (a.get("timestamp", 0.0) for a in _alerts_at_load), default=0.0
    )
    st.session_state.cleared_alerts_cutoff = _latest_existing_ts
    st.session_state.resolved_alerts_cutoff = _latest_existing_ts

if "cleared_alerts_cutoff" not in st.session_state:
    st.session_state.cleared_alerts_cutoff = 0.0

if "resolved_alerts_cutoff" not in st.session_state:
    st.session_state.resolved_alerts_cutoff = 0.0

if "user_name" not in st.session_state:
    st.session_state.user_name = "Alex Morgan"

if "user_role" not in st.session_state:
    st.session_state.user_role = "Store Manager"

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

    @keyframes liveDotBlink {{
        0% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.15; transform: scale(0.8); }}
        100% {{ opacity: 1; transform: scale(1); }}
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

    h1 {{
        font-size: 2.65rem !important;
    }}

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

    section[data-testid="stSidebar"] svg,
    section[data-testid="stSidebar"] svg path,
    section[data-testid="stSidebar"] svg circle,
    section[data-testid="stSidebar"] svg polygon,
    section[data-testid="stSidebar"] svg g,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg path,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg path {{
        fill: rgba(255,255,255,0.65) !important;
        color: rgba(255,255,255,0.65) !important;
        stroke: rgba(255,255,255,0.65) !important;
    }}

    div[data-testid="stColumn"] button[key="notif_bell_btn"] {{
        background: #FFFFFF !important;
        color: {sidebar_bg} !important;
        border: 1px solid rgba(3, 11, 51, 0.14) !important;
        box-shadow: 0 1px 3px rgba(3, 11, 51, 0.12) !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        text-transform: none !important;
        padding: 14px 10px !important;
    }}

    div[data-testid="stColumn"] button[key="notif_bell_btn"] p {{
        color: {sidebar_bg} !important;
    }}

    div[data-testid="stColumn"] button[key="notif_bell_btn"]:hover {{
        background: #F0F0F0 !important;
        box-shadow: 0 2px 6px rgba(3, 11, 51, 0.18) !important;
        transform: translateY(-1px);
    }}

    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid rgba(255,255,255,0.08);
        padding-top: 24px;
    }}

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: rgba(255,255,255,0.92) !important;
        font-family: 'Poppins', sans-serif;
    }}

    section[data-testid="stSidebar"] [data-testid="stExpander"] {{
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }}

    section[data-testid="stSidebar"] summary p {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: rgba(255,255,255,0.78) !important;
    }}

    section[data-testid="stSidebar"] .stButton > button {{
        box-shadow: none !important;
        font-size: 0.92rem !important;
        padding: 10px 14px !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        background: transparent !important;
        text-transform: none !important;
        font-weight: 500 !important;
        justify-content: flex-start !important;
        text-align: left !important;
        margin-bottom: 4px !important;
    }}

    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.14) !important;
        transform: none !important;
    }}

    .sidebar-brand {{
        font-size: 0.8rem;
        font-weight: 600 !important;
        color: rgba(255,255,255,0.45) !important;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 20px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }}

    .sidebar-divider {{
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 18px 0;
    }}

    .profile-card {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }}

    .profile-avatar {{
        width: 42px;
        height: 42px;
        min-width: 42px;
        border-radius: 50%;
        background: {accent_brand};
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.95rem;
        color: #FFFFFF !important;
        font-family: 'Poppins', sans-serif;
    }}

    .profile-name {{
        font-size: 0.98rem;
        font-weight: 600;
        color: #FFFFFF !important;
        line-height: 1.25;
    }}

    .profile-role {{
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5) !important;
    }}

    .status-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3FB950;
        box-shadow: 0 0 6px #3FB950;
        display: inline-block;
    }}

    div[data-testid="stColumn"] button[key="sign_out_btn"] {{
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        color: rgba(255,255,255,0.75) !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        margin-bottom: 0px !important;
    }}

    div[data-testid="stColumn"] button[key="sign_out_btn"] p {{
        color: rgba(255,255,255,0.75) !important;
    }}

    div[data-testid="stColumn"] button[key="sign_out_btn"]:hover {{
        background: rgba(255,59,48,0.12) !important;
        border-color: #FF6B6B !important;
        color: #FF6B6B !important;
        box-shadow: none !important;
        transform: none !important;
    }}

    div[data-testid="stColumn"] button[key="sign_out_btn"]:hover p {{
        color: #FF6B6B !important;
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

    .app-header-subtitle {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 500;
        color: {main_text} !important;
        opacity: 0.65;
        margin-top: -6px;
    }}

    /* LIVE FEED BUTTON WITH BLINKING DOT */
    .blinking-dot {{
        display: inline-block;
        color: #FF3B30 !important;
        margin-right: 6px;
        animation: liveDotBlink 1.2s infinite ease-in-out !important;
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

    /* SOFT ROUNDED HAMBURGER ICON BUTTON
       NOTE: Streamlit does NOT render `key=` as a literal HTML attribute
       on the button — button[key="..."] never matches anything. The
       actual, documented mechanism Streamlit provides for targeting a
       keyed widget with CSS is the auto-added class `st-key-<key>` on
       an ancestor of the widget. That's the selector that actually works;
       the attribute selector is kept alongside as harmless redundancy in
       case of version differences, but .st-key-open_toolbar_btn is doing
       the real work here. */
    .st-key-open_toolbar_btn button,
    div[data-testid="stColumn"] button[key="open_toolbar_btn"] {{
        background-color: #ECE8E1 !important;
        border: none !important;
        border-radius: 14px !important;
        width: 48px !important;
        height: 48px !important;
        min-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        margin-bottom: 20px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
        padding: 0 !important;
    }}

    .st-key-open_toolbar_btn button *,
    .st-key-open_toolbar_btn button p,
    .st-key-open_toolbar_btn button span,
    .st-key-open_toolbar_btn button div,
    div[data-testid="stColumn"] button[key="open_toolbar_btn"] *,
    div[data-testid="stColumn"] button[key="open_toolbar_btn"] p,
    div[data-testid="stColumn"] button[key="open_toolbar_btn"] span,
    div[data-testid="stColumn"] button[key="open_toolbar_btn"] div {{
        color: #030B33 !important;
        -webkit-text-fill-color: #030B33 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }}

    .st-key-open_toolbar_btn button:hover,
    div[data-testid="stColumn"] button[key="open_toolbar_btn"]:hover {{
        background-color: #DFDAD1 !important;
        border: none !important;
        transform: none !important;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# LIVE ALERT POLLING FRAGMENTS
# All fragments below read alerts_log.json via load_recent_alerts_from_disk(),
# which is the file aisletracker.py writes to. This is the ONLY reliable
# channel between the tracker process and this dashboard process.
# ==========================================
@st.fragment(run_every=3)
def render_live_alert_banner():
    alerts = load_recent_alerts_from_disk()

    new_alerts = [
        a for a in alerts
        if a.get("timestamp") not in st.session_state.toasted_alert_timestamps
    ]
    for alert in new_alerts:
        st.toast(alert.get("message", "New alert"), icon="🔔")
        st.session_state.toasted_alert_timestamps.add(alert.get("timestamp"))

    if alerts:
        st.warning(alerts[-1].get("message", "New alert"))
    else:
        st.info("No active dwell alerts right now.")


@st.fragment(run_every=3)
def render_stalled_shoppers_card():
    alert_count = len(load_recent_alerts_from_disk())
    st.markdown(
        f"""
        <div class="card-green-dark">
            <div class="card-title">Stalled Shoppers Alert</div>
            <div class="card-value">{alert_count} Alert{'s' if alert_count != 1 else ''}</div>
            <div class="card-subtext">Requires Floor Staff</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=3)
def render_dispatch_broadcast():
    all_alerts = load_recent_alerts_from_disk()
    live_alerts = [
        a for a in all_alerts
        if a.get("timestamp", 0) > st.session_state.resolved_alerts_cutoff
    ]
    latest = live_alerts[-1]["message"] if live_alerts else "No active alerts — all clear."
    # handle_dwell_alert() entries carry "aisle"; handle_confusion_alert()
    # entries don't (they carry "classification" instead) — fall back
    # gracefully so this doesn't KeyError on the confusion-alert path.
    if live_alerts:
        aisle = live_alerts[-1].get("aisle") or live_alerts[-1].get("classification") or "the monitored zone"
    else:
        aisle = "the monitored zone"

    if live_alerts:
        st.error(f"**ACTIVE ALERT:** Customer stalled in {aisle} needs floor assistance.")
    else:
        st.success("No active alerts. Floor is clear.")

    st.markdown(
        f"""
        <div class="card-green-dark">
            <h1 style="font-size: 2.2rem; margin-top: 0; font-weight: 600; color: #FFFFFF !important;">Automated Floor Broadcast</h1>
            <p style="margin-bottom: 20px; color: #FFFFFF !important;">Message dispatched to floor assistant handset:</p>
            <div style="background: {btn_bg}; color: #FFFFFF !important; border: 3px solid {border_col}; border-radius: 14px; padding: 24px; font-size: 1.3rem; font-weight: 600; box-shadow: 4px 4px 0px {box_shadow_col};">
                "{latest}"
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=3)
def render_notification_bell_button():
    all_alerts = load_recent_alerts_from_disk()
    visible_alerts = [
        a for a in all_alerts
        if a.get("timestamp", 0) > st.session_state.cleared_alerts_cutoff
    ]
    bell_label = f"🔔 {len(visible_alerts)}" if visible_alerts else "🔔"
    if st.button(bell_label, key="notif_bell_btn", use_container_width=True):
        navigate_to("Notifications")


@st.fragment(run_every=3)
def render_notifications_page():
    all_alerts = load_recent_alerts_from_disk()
    visible_alerts = [
        a for a in all_alerts
        if a.get("timestamp", 0) > st.session_state.cleared_alerts_cutoff
    ]

    if not visible_alerts:
        st.markdown(
            """
            <div class="card-green-light" style="text-align:center;">
                <div class="card-title">All Caught Up</div>
                <div class="card-subtext">You have no notifications right now.</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Clear All Button Header
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("CLEAR ALL MESSAGES", key="clear_all_notifs_btn", use_container_width=True):
            st.session_state.cleared_alerts_cutoff = time.time()
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    for alert in reversed(visible_alerts):
        ts = alert.get("timestamp")
        time_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "—"
        st.markdown(
            f"""
            <div class="card-green-light">
                <div class="card-subtext" style="font-weight: 600;">{time_str} — {alert.get('message', '')}</div>
            </div>
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
            <div class="splash-subtitle">Retail Intelligence Engine</div>
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
        "<div class='sidebar-brand'>Toolbar</div>", unsafe_allow_html=True
    )

    _name = st.session_state.user_name.strip() or "Guest User"
    _initials = "".join(p[0].upper() for p in _name.split()[:2]) or "U"

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-avatar">{_initials}</div>
            <div>
                <div class="profile-name">{_name}</div>
                <div class="profile-role">{st.session_state.user_role}</div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Edit Profile", expanded=False):
        st.text_input("Name", key="user_name")
        st.text_input("Role", key="user_role")

    with st.expander("Theme & Display", expanded=False):
        dark_mode_toggle = st.toggle(
            "Dark Mode", value=st.session_state.dark_mode, key="dark_mode_switch"
        )
        if dark_mode_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode_toggle
            st.rerun()

    with st.expander("Navigation", expanded=False):
        st.selectbox(
            "Jump to",
            options=list(TAB_MAP.values()),
            key="nav_select_dropdown",
            on_change=on_nav_dropdown_change,
        )

    with st.expander("Quick Controls", expanded=False):
        st.toggle("Auto-Refresh Feed", value=True)
        st.selectbox(
            "Floor Zone", ["Zone A (Snacks)", "Zone B (Produce)", "Zone C (Bakery)"]
        )
        st.slider(
            "Alert Sensitivity", 10, 60, 45, help="Dwell duration trigger in seconds"
        )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='status-row'><span class='status-dot'></span>STATUS: ONLINE</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
        for _key in list(st.session_state.keys()):
            del st.session_state[_key]
        st.rerun()

# ==========================================
# HAMBURGER MENU BUTTON
# ==========================================
col_btn, col_empty = st.columns([1, 15])
with col_btn:
    if st.button("≡", key="open_toolbar_btn", help="Toggle Menu"):
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
elif st.session_state.active_tab == "Notifications":
    header_title = "Notifications"
else:
    header_title = "Live Tracking"

if st.session_state.active_tab == "Overview":
    col_title, col_bell, col_live_btn = st.columns([4, 0.7, 1.3])
    with col_title:
        st.markdown(
            f"""<div class="app-header-clean">
                <span class="app-brand-clean">{header_title}</span>
            </div>
            <div class="app-header-subtitle">Welcome back, {st.session_state.user_name.split()[0] if st.session_state.user_name.strip() else 'there'}</div>
            """,
            unsafe_allow_html=True,
        )
    with col_bell:
        st.markdown(
            '<div style="padding-top: 15px;"></div>', unsafe_allow_html=True
        )
        render_notification_bell_button()
    with col_live_btn:
        st.markdown(
            '<div style="padding-top: 15px;"></div>', unsafe_allow_html=True
        )
        if st.button("🔴 LIVE FEED", key="live_feed_redirect_btn"):
            navigate_to("Live Tracking")
else:
    st.markdown(
        f'<div class="app-header-clean"><span class="app-brand-clean">{header_title}</span></div>',
        unsafe_allow_html=True,
    )

# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
if st.session_state.active_tab == "Overview":

    render_live_alert_banner()

    st.markdown("### Select App Module")
    n1, n2 = st.columns(2)
    with n1:
        if st.button(
            "OPEN HEATMAP ANALYTICS",
            key="nav_heatmap_btn",
            use_container_width=True,
        ):
            navigate_to("Heatmap")
    with n2:
        if st.button(
            "OPEN STAFF DISPATCH",
            key="nav_dispatch_btn",
            use_container_width=True,
        ):
            navigate_to("Dispatch")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("# Live Floor Metrics")

    st.markdown(
        """
        <div class="card-green-light">
            <div class="card-title">Active Shoppers In Store</div>
            <div class="card-value">14</div>
            <div class="card-subtext">↑ +2 vs 5m ago</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    render_stalled_shoppers_card()

    st.markdown(
        """
        <div class="card-green-light">
            <div class="card-title">Average Dwell Time</div>
            <div class="card-value">38 Secs</div>
            <div class="card-subtext">↓ -4s vs Target</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card-green-dark">
            <div class="card-title">Top Hotspot Zone</div>
            <div class="card-value">Shelf B</div>
            <div class="card-subtext">82% Focus Share</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("# Executive Insights")

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

        if st.button("Hide Daily AI Summary", key="hide_ai_summary_btn"):
            st.session_state.show_ai_summary = False
            st.rerun()

# ==========================================
# TAB 2: HEATMAP ANALYTICS
# ==========================================
elif st.session_state.active_tab == "Heatmap":
    if st.button("← BACK TO DASHBOARD", key="back_from_heatmap"):
        navigate_to("Overview")

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
    if st.button("← BACK TO DASHBOARD", key="back_from_dispatch"):
        navigate_to("Overview")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    render_dispatch_broadcast()

    if st.button("MARK INCIDENT AS RESOLVED", key="resolve_incident_btn", use_container_width=True):
        st.session_state.resolved_alerts_cutoff = time.time()
        st.rerun()

# ==========================================
# TAB 4: LIVE TRACKING
# ==========================================
elif st.session_state.active_tab == "Live Tracking":
    if st.button("← BACK TO DASHBOARD", key="back_from_livetracking"):
        navigate_to("Overview")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="background: {btn_bg}; border: 3px dashed {border_col}; border-radius: 20px; height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #FFFFFF; text-align: center; padding: 20px; box-shadow: 6px 6px 0px {box_shadow_col};">
            <div style="font-size: 1.8rem; font-weight: 600;">Video Stream Placeholder</div>
            <div style="font-size: 1.15rem; color: #A8D0E6; margin-top: 8px;">Insert your teammate's CCTV video tracking code here later.</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

# ==========================================
# TAB 5: NOTIFICATIONS
# ==========================================
elif st.session_state.active_tab == "Notifications":
    if st.button("← BACK TO DASHBOARD", key="back_from_notifications"):
        navigate_to("Overview")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    render_notifications_page()