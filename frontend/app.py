"""SyntheLoop — Streamlit Frontend Application.

Developer-grade closed-loop synthetic tabular data generation dashboard:
- Dark matte palette: black (#08090A), dark slate-grey (#141619), crisp white typography
- Restrained surgical hints of muted olive green (#5A6E36, #8EA658)
- Standout gradient Start Optimization Loop button with interactive hover glow
- Dedicated 3rd box: LLM Planner Reasoning & Refinement Diagnosis styled as an interactive
  terminal where every line is prefixed with `SyntheLoop> `
- Multi-metric KPI cards, comparative Real vs Synthetic explorer, and one-click artifact downloads
"""

import io
import json
import os
import time
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Endpoints
# -----------------------------------------------------------------------------
API_BASE_URL = os.getenv("SYNTHELOOP_API_URL", "http://127.0.0.1:8000")

DEFAULT_THRESHOLDS = {
    "ks_stat_max": 0.15,
    "corr_diff_max": 0.20,
    "js_divergence_max": 0.10,
    "dcr_min_percentile": 5,
    "utility_auc_drop_max": 0.10,
}

# -----------------------------------------------------------------------------
# Page Config & Custom Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SyntheLoop — Closed-Loop Synthetic Data Refinement",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS implementing the exact approved palette:
# Black, Grey, White base + surgical hints of olive green + hover gradient button
CUSTOM_CSS = """
<style>
    /* Global Canvas Styling */
    .stApp {
        background-color: #08090A;
        color: #E6E8EA;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Hide Streamlit default header, Deploy button, and hamburger menu */
    [data-testid="stHeader"] {
        display: none !important;
    }
    .stDeployButton {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }

    /* Clean top spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Top Navigation Bar */
    .syntheloop-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0E1013;
        border: 1px solid #20242B;
        border-radius: 8px;
        padding: 10px 20px;
        margin-bottom: 14px;
    }
    .syntheloop-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .syntheloop-logo-text {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .syntheloop-version-pill {
        font-size: 0.72rem;
        background: #1A1D23;
        color: #8C939E;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #282D36;
        font-family: monospace;
    }
    .syntheloop-nav-right {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .syntheloop-status-pill {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #111417;
        border: 1px solid #242930;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #C0C5CC;
    }
    .led-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6E8B3D;
        box-shadow: 0 0 6px rgba(110, 139, 61, 0.7);
    }
    .led-dot-pending {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #D4A359;
        box-shadow: 0 0 6px rgba(212, 163, 89, 0.7);
    }
    .led-dot-offline {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #A33A3A;
        box-shadow: 0 0 6px rgba(163, 58, 58, 0.7);
    }
    .dataset-pill {
        background: #14171C;
        border: 1px solid #262B34;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #9CA3AF;
        font-family: monospace;
    }

    /* Native Container Box Framing */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0E1013 !important;
        border: 1px solid #1F232B !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }

    /* Telemetry Ribbon */
    .telemetry-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0E1013;
        border: 1px solid #1E2229;
        border-radius: 8px;
        padding: 8px 18px;
        margin-bottom: 14px;
        font-size: 0.85rem;
    }
    .telemetry-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .iter-badge {
        background: #161A20;
        border: 1px solid #2A303C;
        color: #FFFFFF;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-family: monospace;
    }
    .telemetry-time {
        color: #7A828E;
        font-family: monospace;
    }
    .progress-segment-container {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    .progress-segment {
        width: 48px;
        height: 6px;
        border-radius: 3px;
        background: #1A1E24;
    }
    .progress-segment.active {
        background: #5A6E36;
        box-shadow: 0 0 8px rgba(90, 110, 54, 0.5);
    }
    .progress-segment.complete {
        background: #414D2A;
    }

    /* KPI Scorecards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: #101216;
        border: 1px solid #20252D;
        border-radius: 8px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 86px;
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        color: #8C939E;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .kpi-value-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-top: 6px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: monospace;
    }
    .kpi-thresh {
        font-size: 0.72rem;
        color: #6C727D;
        font-family: monospace;
    }
    .kpi-status-tag {
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        font-family: monospace;
    }
    .kpi-status-passed {
        background: #152215;
        color: #7E9E48;
        border: 1px solid #2B3D1B;
    }
    .kpi-status-failed {
        background: #251616;
        color: #D46B6B;
        border: 1px solid #4D2626;
    }
    .kpi-status-pending {
        background: #171A1F;
        color: #6D7480;
        border: 1px solid #242932;
    }

    /* Panel Containers */
    .panel-box {
        background: #0E1013;
        border: 1px solid #1F232B;
        border-radius: 8px;
        padding: 14px;
        height: 100%;
    }
    .panel-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.2px;
    }

    /* Dedicated Terminal Box (3rd box) */
    .terminal-window {
        background: #07080A;
        border: 1px solid #1E232B;
        border-radius: 8px;
        font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
        padding: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: 520px;
    }
    .terminal-topbar {
        background: #0F1216;
        border-bottom: 1px solid #1C2028;
        padding: 8px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .terminal-dots {
        display: flex;
        gap: 6px;
    }
    .t-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .t-dot-red { background: #4D2525; }
    .t-dot-yellow { background: #4A3E20; }
    .t-dot-green { background: #2A4020; }
    .terminal-title {
        font-size: 0.75rem;
        color: #8C939E;
        font-weight: 500;
    }
    .terminal-body {
        padding: 12px;
        overflow-y: auto;
        font-size: 0.8rem;
        line-height: 1.55;
        flex-grow: 1;
        color: #D1D5DB;
        background: #07080A;
    }
    .t-prompt {
        color: #8EA658;
        font-weight: 600;
    }
    .t-line {
        margin-bottom: 4px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .t-highlight-pass {
        color: #8EA658;
        font-weight: 600;
    }
    .t-highlight-fail {
        color: #E06C75;
        font-weight: 600;
    }
    .t-cursor {
        display: inline-block;
        width: 8px;
        height: 14px;
        background-color: #8EA658;
        vertical-align: middle;
        animation: blink 1s step-end infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* Start Optimization Loop Gradient Button & Hover Effect */
    div.stButton > button.syntheloop-start-btn,
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1C2024 0%, #2A361F 55%, #364426 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #4D5E34 !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.22s ease-in-out !important;
        width: 100% !important;
        cursor: pointer !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #22272D 0%, #354425 55%, #44562E 100%) !important;
        border-color: #6E8B3D !important;
        box-shadow: 0 0 16px rgba(110, 139, 61, 0.45) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
        box-shadow: 0 0 8px rgba(110, 139, 61, 0.3) !important;
    }

    /* Secondary / Download Buttons */
    div.stDownloadButton > button {
        background: #14171C !important;
        color: #D1D5DB !important;
        border: 1px solid #282E38 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        padding: 6px 12px !important;
        transition: background 0.15s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: #1C2028 !important;
        border-color: #3E4654 !important;
        color: #FFFFFF !important;
    }

    /* Form Controls & Sliders */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #9CA3AF !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background-color: #121418 !important;
        color: #FFFFFF !important;
        border: 1px solid #262B34 !important;
        border-radius: 6px !important;
    }

    /* Tables & Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1F242C;
        border-radius: 6px;
        background: #0C0E11;
    }
    
    /* Footer Telemetry */
    .footer-telemetry {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #1A1D24;
        padding-top: 10px;
        margin-top: 24px;
        font-size: 0.75rem;
        color: #6C727D;
        font-family: monospace;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# API Client Helper Functions
# -----------------------------------------------------------------------------
def check_api_health(base_url: str) -> bool:
    """Checks whether the FastAPI backend is running and responding."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


def upload_dataset(base_url: str, file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Uploads a CSV file to /upload and returns the EDA summary and run_id."""
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{base_url}/upload", files=files, timeout=30.0)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
        raise RuntimeError(f"Upload failed ({resp.status_code}): {detail}")
    return resp.json()


def trigger_start_run(
    base_url: str,
    run_id: str,
    target_col: Optional[str],
    thresholds: dict[str, Any],
    max_iterations: int,
    groq_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Sends start payload to /runs/{run_id}/start."""
    payload = {
        "target_col": target_col,
        "thresholds": thresholds,
        "max_iterations": max_iterations,
        "groq_api_key": groq_api_key,
    }
    resp = requests.post(f"{base_url}/runs/{run_id}/start", json=payload, timeout=10.0)
    if resp.status_code not in (200, 202):
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
        raise RuntimeError(f"Start run failed ({resp.status_code}): {detail}")
    return resp.json()


def fetch_run_status(base_url: str, run_id: str) -> dict[str, Any]:
    """Fetches the latest status and metrics from /runs/{run_id}/status."""
    resp = requests.get(f"{base_url}/runs/{run_id}/status", timeout=5.0)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
        raise RuntimeError(f"Status check failed ({resp.status_code}): {detail}")
    return resp.json()


def fetch_artifact_bytes(base_url: str, run_id: str, artifact: str) -> Optional[bytes]:
    """Downloads artifact (dataset, report, audit_trail) from /runs/{run_id}/download/{artifact}."""
    try:
        resp = requests.get(f"{base_url}/runs/{run_id}/download/{artifact}", timeout=10.0)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "eda_summary" not in st.session_state:
    st.session_state.eda_summary = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "run_status" not in st.session_state:
    st.session_state.run_status = "idle"
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "max_iterations" not in st.session_state:
    st.session_state.max_iterations = 5
if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = None
if "terminal_logs" not in st.session_state:
    st.session_state.terminal_logs = [
        "SyntheLoop> system initialized. Waiting for dataset ingestion...",
        "SyntheLoop> ready.",
    ]
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "elapsed_str" not in st.session_state:
    st.session_state.elapsed_str = "00:00:00"


def append_terminal_log(message: str) -> None:
    """Appends a line to the dedicated terminal with the required SyntheLoop> prefix."""
    formatted = f"SyntheLoop> {message}" if not message.startswith("SyntheLoop>") else message
    if formatted not in st.session_state.terminal_logs:
        st.session_state.terminal_logs.append(formatted)


# -----------------------------------------------------------------------------
# Sidebar Configuration (API Key & Pipeline Settings)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    env_groq_key = os.getenv("GROQ_API_KEY", "")
    user_groq_key = st.text_input(
        "Groq API Key",
        value=st.session_state.get("groq_api_key", env_groq_key),
        type="password",
        placeholder="gsk_...",
        help="Groq API Key used by LLM Planner & Evaluator (Groq LLaMA-3.3-70B). Can also be set in .env.",
    )
    st.session_state.groq_api_key = user_groq_key.strip() if user_groq_key else ""

    if st.session_state.groq_api_key:
        st.markdown('<span style="color:#7E9E48; font-size:0.8rem; font-family:monospace;">● Groq API Key Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#D4A359; font-size:0.8rem; font-family:monospace;">○ No Key Set (fallback to .env)</span>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#20252D; margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Pipeline Settings")
    st.caption(f"Backend API URL: `{API_BASE_URL}`")
    st.caption("LLM Model: `llama-3.3-70b-versatile`")
    st.caption("Generator: `CTGAN (SDV)`")


# -----------------------------------------------------------------------------
# Top Navigation Bar
# -----------------------------------------------------------------------------
is_api_online = check_api_health(API_BASE_URL)
has_groq_key = bool(st.session_state.get("groq_api_key"))

backend_status_html = (
    '<span class="syntheloop-status-pill"><span class="led-dot"></span> Backend: Online</span>'
    if is_api_online
    else '<span class="syntheloop-status-pill"><span class="led-dot-offline"></span> Backend: Offline</span>'
)

groq_status_html = (
    '<span class="syntheloop-status-pill"><span class="led-dot"></span> Groq LLM: Ready</span>'
    if has_groq_key
    else '<span class="syntheloop-status-pill"><span class="led-dot-pending"></span> Groq LLM: Key Needed</span>'
)

dataset_info_str = (
    f"{st.session_state.filename} ({st.session_state.eda_summary['n_rows']} rows, {st.session_state.eda_summary['n_cols']} cols)"
    if st.session_state.eda_summary
    else "No dataset loaded"
)

# Render Top Bar
st.markdown(
    f"""
    <div class="syntheloop-navbar">
        <div class="syntheloop-brand">
            <span class="syntheloop-logo-text">SyntheLoop</span>
            <span class="syntheloop-version-pill">v1.0</span>
            {backend_status_html}
            {groq_status_html}
        </div>
        <div class="syntheloop-nav-right">
            <span class="dataset-pill">📁 {dataset_info_str}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_api_online:
    st.warning(
        f"⚠️ Cannot reach FastAPI backend at `{API_BASE_URL}`. "
        "Make sure the backend is running via `uvicorn backend.main:app --reload --port 8000`."
    )

# -----------------------------------------------------------------------------
# Optimization Loop Telemetry Ribbon
# -----------------------------------------------------------------------------
current_iter = st.session_state.iteration
max_iter = st.session_state.max_iterations

# Generate segmented progress bars
segment_htmls = []
for i in range(1, max_iter + 1):
    if i < current_iter:
        segment_htmls.append('<div class="progress-segment complete"></div>')
    elif i == current_iter and st.session_state.is_running:
        segment_htmls.append('<div class="progress-segment active"></div>')
    else:
        segment_htmls.append('<div class="progress-segment"></div>')
segments_rendered = "".join(segment_htmls)

status_badge_color = "#8EA658" if st.session_state.run_status in ("completed_threshold_met", "completed_max_iterations") else "#FFFFFF"
status_display = st.session_state.run_status.upper().replace("_", " ")

st.markdown(
    f"""
    <div class="telemetry-ribbon">
        <div class="telemetry-left">
            <span style="font-weight:700; letter-spacing:0.5px; color:#A0A6AD;">OPTIMIZATION LOOP</span>
            <span class="iter-badge">Iteration {current_iter} of {max_iter}</span>
            <span class="telemetry-time">Elapsed: {st.session_state.elapsed_str}</span>
            <span style="color:{status_badge_color}; font-family:monospace; font-weight:600;">Status: {status_display}</span>
        </div>
        <div class="progress-segment-container">
            {segments_rendered}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Row of 5 Compact KPI Scorecards
# -----------------------------------------------------------------------------
metrics = st.session_state.latest_metrics or {}
passed_dict = metrics.get("passed", {})


def get_kpi_badge(passed: Optional[bool]) -> str:
    if passed is True:
        return '<span class="kpi-status-tag kpi-status-passed">PASSED</span>'
    elif passed is False:
        return '<span class="kpi-status-tag kpi-status-failed">FAIL</span>'
    return '<span class="kpi-status-tag kpi-status-pending">PENDING</span>'


# 1. KS Stat
ks_dict = metrics.get("per_column_ks", {})
ks_val = f"{sum(ks_dict.values()) / len(ks_dict):.2f}" if ks_dict else "—"
ks_tag = get_kpi_badge(passed_dict.get("ks"))

# 2. Correlation
corr_val_num = metrics.get("correlation_diff_frobenius")
corr_val = f"{corr_val_num:.2f}" if corr_val_num is not None else "—"
corr_tag = get_kpi_badge(passed_dict.get("correlation"))

# 3. JS Divergence
js_val_num = metrics.get("class_balance_js_divergence")
js_val = f"{js_val_num:.2f}" if js_val_num is not None else "—"
js_tag = get_kpi_badge(passed_dict.get("balance"))

# 4. Privacy DCR
dcr_val_num = metrics.get("privacy_dcr_5th_percentile")
dcr_val = f"{dcr_val_num:.2f}" if dcr_val_num is not None else "—"
dcr_tag = get_kpi_badge(passed_dict.get("privacy"))

# 5. ML Utility AUC
util_dict = metrics.get("utility", {})
auc_drop = util_dict.get("auc_drop")
util_val = f"{util_dict.get('tstr_auc', 0.0):.2f}" if "tstr_auc" in util_dict else "—"
util_tag = get_kpi_badge(passed_dict.get("utility"))

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-header"><span>KS Statistic</span>{ks_tag}</div>
            <div class="kpi-value-row"><span class="kpi-value">{ks_val}</span><span class="kpi-thresh">thresh ≤ 0.15</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header"><span>Correlation Diff</span>{corr_tag}</div>
            <div class="kpi-value-row"><span class="kpi-value">{corr_val}</span><span class="kpi-thresh">thresh ≤ 0.20</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header"><span>JS Divergence</span>{js_tag}</div>
            <div class="kpi-value-row"><span class="kpi-value">{js_val}</span><span class="kpi-thresh">thresh ≤ 0.10</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header"><span>Privacy DCR</span>{dcr_tag}</div>
            <div class="kpi-value-row"><span class="kpi-value">{dcr_val}</span><span class="kpi-thresh">thresh ≥ 0.25</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header"><span>ML Utility (AUC)</span>{util_tag}</div>
            <div class="kpi-value-row"><span class="kpi-value">{util_val}</span><span class="kpi-thresh">thresh drop ≤ 0.10</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main 3-Column Workstation Layout
# -----------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1.1, 1.5, 1.4], gap="medium")

# =============================================================================
# Left Column: Target & Guardrails Controls
# =============================================================================
with col_left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">⚙️ Target & Guardrails</div>', unsafe_allow_html=True)

        # Groq API Key Input Field (Prominent in main UI)
        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#8C939E; margin-bottom:4px;'>🔑 GROQ API KEY</div>", unsafe_allow_html=True)
        env_groq_key = os.getenv("GROQ_API_KEY", "")
        current_groq_key = st.session_state.get("groq_api_key", env_groq_key)
        user_key = st.text_input(
            "Groq API Key",
            value=current_groq_key,
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
            help="Required for LLM Planner & Evaluator (llama-3.3-70b-versatile).",
        )
        st.session_state.groq_api_key = user_key.strip() if user_key else ""
        if st.session_state.groq_api_key:
            st.markdown('<div style="color:#7E9E48; font-size:0.75rem; font-family:monospace; margin-bottom:10px;">● API Key Configured & Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#D4A359; font-size:0.75rem; font-family:monospace; margin-bottom:10px;">⚠️ Enter your Groq API Key (gsk_...) to run the loop</div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1E2229; margin:4px 0 12px 0;'>", unsafe_allow_html=True)

        # Ingestion Source Tabs
        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#8C939E; margin-bottom:4px;'>📁 DATASET INGESTION</div>", unsafe_allow_html=True)
        ingest_tab1, ingest_tab2 = st.tabs(["Upload CSV", "Demo Sample"])

        with ingest_tab1:
            uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                if st.session_state.filename != uploaded_file.name:
                    with st.spinner("Analyzing dataset via EDA module..."):
                        try:
                            content_bytes = uploaded_file.getvalue()
                            upload_res = upload_dataset(API_BASE_URL, content_bytes, uploaded_file.name)
                            st.session_state.run_id = upload_res["run_id"]
                            st.session_state.eda_summary = upload_res["eda_summary"]
                            st.session_state.filename = uploaded_file.name
                            st.session_state.raw_df = pd.read_csv(io.BytesIO(content_bytes))
                            append_terminal_log(f"uploaded '{uploaded_file.name}' (run_id: {upload_res['run_id']})")
                            append_terminal_log(f"EDA complete: {upload_res['eda_summary']['n_rows']} rows, {upload_res['eda_summary']['n_cols']} columns")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload error: {e}")

        with ingest_tab2:
            st.caption("Quickly test with built-in sample churn dataset:")
            sample_path = "data/samples/sample_churn.csv"
            if st.button("Load Sample Churn Dataset", use_container_width=True):
                if os.path.exists(sample_path):
                    with st.spinner("Loading sample dataset..."):
                        try:
                            with open(sample_path, "rb") as f:
                                sample_bytes = f.read()
                            upload_res = upload_dataset(API_BASE_URL, sample_bytes, "sample_churn.csv")
                            st.session_state.run_id = upload_res["run_id"]
                            st.session_state.eda_summary = upload_res["eda_summary"]
                            st.session_state.filename = "sample_churn.csv"
                            st.session_state.raw_df = pd.read_csv(io.BytesIO(sample_bytes))
                            append_terminal_log(f"loaded sample dataset 'sample_churn.csv' (run_id: {upload_res['run_id']})")
                            append_terminal_log(f"EDA complete: {upload_res['eda_summary']['n_rows']} rows, {upload_res['eda_summary']['n_cols']} columns")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading sample: {e}")
                else:
                    st.warning(f"File '{sample_path}' not found.")

        st.markdown("<hr style='border-color:#1E2229; margin:12px 0;'>", unsafe_allow_html=True)

        # Target Column Selector
        col_names = []
        if st.session_state.eda_summary and "columns" in st.session_state.eda_summary:
            col_names = list(st.session_state.eda_summary["columns"].keys())

        default_target_idx = 0
        if col_names:
            for idx, col_name in enumerate(["None"] + col_names):
                if col_name.lower() in ("churn", "income", "target", "label"):
                    default_target_idx = idx
                    break

        target_col_selection = st.selectbox(
            "Target Column (Optional)",
            options=["None"] + col_names,
            index=default_target_idx,
            help="Column to evaluate class balance and downstream ML utility.",
        )
        chosen_target = None if target_col_selection == "None" else target_col_selection

        # Guardrail Sliders
        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#8C939E; margin-top:10px; margin-bottom:4px;'>OPTIMIZATION GUARDRAILS</div>", unsafe_allow_html=True)

        t_ks = st.slider("KS Statistic Max (Fidelity)", 0.05, 0.40, 0.15, 0.01)
        t_corr = st.slider("Correlation Diff Max (Frobenius)", 0.05, 0.40, 0.20, 0.01)
        t_js = st.slider("JS Divergence Max (Class Balance)", 0.01, 0.30, 0.10, 0.01)
        t_dcr = st.slider("Privacy DCR 5th Percentile Floor", 1, 20, 5, 1)
        t_util = st.slider("Utility AUC Drop Max", 0.01, 0.30, 0.10, 0.01)

        max_iterations_input = st.number_input("Max Iterations Budget", min_value=1, max_value=10, value=3, step=1)
        st.session_state.max_iterations = max_iterations_input

        thresholds_payload = {
            "ks_stat_max": t_ks,
            "corr_diff_max": t_corr,
            "js_divergence_max": t_js,
            "dcr_min_percentile": t_dcr,
            "utility_auc_drop_max": t_util,
        }

        # Start Optimization Loop Button & Status Guards
        has_dataset = bool(st.session_state.run_id)
        has_key = bool(st.session_state.groq_api_key)
        can_start = bool(has_dataset and has_key and not st.session_state.is_running and is_api_online)

        st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
        if st.button("▶  Start Optimization Loop", key="start_btn", type="primary", disabled=not can_start):
            try:
                start_res = trigger_start_run(
                    API_BASE_URL,
                    st.session_state.run_id,
                    chosen_target,
                    thresholds_payload,
                    max_iterations_input,
                    groq_api_key=st.session_state.get("groq_api_key") or None,
                )
                st.session_state.is_running = True
                st.session_state.run_status = "running"
                st.session_state.start_time = time.time()
                st.session_state.iteration = 0
                append_terminal_log(f"starting feedback loop for run_id '{st.session_state.run_id}'...")
                append_terminal_log(f"target_col='{chosen_target}', max_iter={max_iterations_input}")
                append_terminal_log("dispatching background pipeline task...")
                st.rerun()
            except Exception as err:
                st.error(f"Failed to start run: {err}")

        # Help caption guiding the user
        if not has_key:
            st.caption("⚠️ Please enter your Groq API Key above to start")
        elif not has_dataset:
            st.caption("⚠️ Upload a CSV or click 'Load Sample Churn Dataset'")
        elif not is_api_online:
            st.caption("⚠️ FastAPI backend is offline (check port 8000)")
        elif st.session_state.is_running:
            st.caption("⏳ Optimization loop is actively running...")
        else:
            st.caption("🟢 Ready to start optimization loop")

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Center Column: Real vs Synthetic Data Explorer
# =============================================================================
with col_center:
    with st.container(border=True):
        st.markdown('<div class="panel-title">📊 Data Explorer & Distributions</div>', unsafe_allow_html=True)

        tab_data, tab_eda, tab_corrs = st.tabs(["Real vs Synthetic", "Column Profiles", "Correlations"])

        with tab_data:
            if st.session_state.raw_df is not None:
                # Check if synthetic data has been generated
                if st.session_state.synthetic_df is None and st.session_state.run_id:
                    synth_bytes = fetch_artifact_bytes(API_BASE_URL, st.session_state.run_id, "dataset")
                    if synth_bytes:
                        try:
                            st.session_state.synthetic_df = pd.read_csv(io.BytesIO(synth_bytes))
                        except Exception:
                            pass

                if st.session_state.synthetic_df is not None:
                    st.caption(f"Comparing first 5 rows: Real vs Synthetic ({st.session_state.filename})")
                    col_sub1, col_sub2 = st.columns(2)
                    with col_sub1:
                        st.markdown("**Real Dataset Sample**")
                        st.dataframe(st.session_state.raw_df.head(5), height=240, use_container_width=True)
                    with col_sub2:
                        st.markdown("**Synthetic Output Sample**")
                        st.dataframe(st.session_state.synthetic_df.head(5), height=240, use_container_width=True)
                else:
                    st.caption(f"Real Data Preview ({st.session_state.filename}):")
                    st.dataframe(st.session_state.raw_df.head(10), height=380, use_container_width=True)
            else:
                st.info("Upload a dataset or load the sample churn data on the left to inspect records.")

        with tab_eda:
            if st.session_state.eda_summary and "columns" in st.session_state.eda_summary:
                summary_cols = []
                for col, details in st.session_state.eda_summary["columns"].items():
                    summary_cols.append({
                        "Column": col,
                        "Type": details.get("type", "unknown"),
                        "Dtype": details.get("dtype", "unknown"),
                        "Missing %": f"{details.get('missing_pct', 0.0):.1f}%",
                        "Unique": details.get("n_unique", 0),
                        "Mean": round(details["mean"], 2) if "mean" in details else "—",
                        "Std": round(details["std"], 2) if "std" in details else "—",
                    })
                st.dataframe(pd.DataFrame(summary_cols), height=380, use_container_width=True)
            else:
                st.info("EDA column profiles will populate once a CSV is loaded.")

        with tab_corrs:
            if st.session_state.eda_summary and "correlation_matrix" in st.session_state.eda_summary:
                corr_mat = st.session_state.eda_summary["correlation_matrix"]
                if corr_mat:
                    st.dataframe(pd.DataFrame(corr_mat), height=380, use_container_width=True)
                else:
                    st.info("Fewer than 2 continuous columns detected for correlation matrix.")
            else:
                st.info("Correlation matrix available upon dataset upload.")

        # Bottom Export Action Bar (Visible when run completes or artifacts exist)
        st.markdown("<hr style='border-color:#1E2229; margin:12px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#8C939E; margin-bottom:8px;'>EXPORT RUN ARTIFACTS</div>", unsafe_allow_html=True)

        d_col1, d_col2, d_col3 = st.columns(3)

        dataset_bytes = fetch_artifact_bytes(API_BASE_URL, st.session_state.run_id, "dataset") if st.session_state.run_id else None
        report_bytes = fetch_artifact_bytes(API_BASE_URL, st.session_state.run_id, "report") if st.session_state.run_id else None
        trail_bytes = fetch_artifact_bytes(API_BASE_URL, st.session_state.run_id, "audit_trail") if st.session_state.run_id else None

        with d_col1:
            st.download_button(
                label="📄 Synthetic CSV",
                data=dataset_bytes or b"",
                file_name=f"synthetic_{st.session_state.run_id or 'sample'}.csv",
                mime="text/csv",
                disabled=dataset_bytes is None,
                use_container_width=True,
            )
        with d_col2:
            st.download_button(
                label="📊 HTML Report",
                data=report_bytes or b"",
                file_name=f"report_{st.session_state.run_id or 'sample'}.html",
                mime="text/html",
                disabled=report_bytes is None,
                use_container_width=True,
            )
        with d_col3:
            st.download_button(
                label="🔍 Audit Trail",
                data=trail_bytes or b"",
                file_name=f"audit_trail_{st.session_state.run_id or 'sample'}.json",
                mime="application/json",
                disabled=trail_bytes is None,
                use_container_width=True,
            )


# =============================================================================
# Right Column: 3rd Box — Dedicated LLM Terminal Output (`SyntheLoop> `)
# =============================================================================
with col_right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">🧠 LLM Planner & Diagnosis Terminal</div>', unsafe_allow_html=True)

        # Dedicated terminal styling matching exact user specifications
        terminal_lines = []
        for log_line in st.session_state.terminal_logs:
            clean_line = log_line.replace("<", "&lt;").replace(">", "&gt;")
            if "PASS" in clean_line:
                clean_line = clean_line.replace("PASS", '<span class="t-highlight-pass">PASS</span>')
            if "FAIL" in clean_line:
                clean_line = clean_line.replace("FAIL", '<span class="t-highlight-fail">FAIL</span>')
            terminal_lines.append(f'<div class="t-line">{clean_line}</div>')

        terminal_content_html = "".join(terminal_lines)

        st.markdown(
            f"""
            <div class="terminal-window">
                <div class="terminal-topbar">
                    <div class="terminal-dots">
                        <span class="t-dot t-dot-red"></span>
                        <span class="t-dot t-dot-yellow"></span>
                        <span class="t-dot t-dot-green"></span>
                    </div>
                    <span class="terminal-title">LLM Planner Reasoning & Refinement Diagnosis</span>
                    <span style="font-size:0.7rem; color:#5A6E36; font-family:monospace;">bash (syntheloop-env)</span>
                </div>
                <div class="terminal-body" id="terminal-body-container">
                    {terminal_content_html}
                    <div class="t-line"><span class="t-prompt">SyntheLoop&gt;</span> <span class="t-cursor"></span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Active Polling Loop (when run is executing in background)
# -----------------------------------------------------------------------------
if st.session_state.is_running and st.session_state.run_id:
    # Update elapsed timer
    if st.session_state.start_time:
        elapsed_sec = int(time.time() - st.session_state.start_time)
        hrs = elapsed_sec // 3600
        mins = (elapsed_sec % 3600) // 60
        secs = elapsed_sec % 60
        st.session_state.elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

    try:
        status_data = fetch_run_status(API_BASE_URL, st.session_state.run_id)
        current_state_status = status_data.get("status", "running")
        st.session_state.run_status = current_state_status
        st.session_state.iteration = status_data.get("iteration", st.session_state.iteration)
        st.session_state.latest_metrics = status_data.get("latest_metrics", st.session_state.latest_metrics)

        # Inspect audit trail entries to extract LLM reasoning lines for terminal
        trail_content = fetch_artifact_bytes(API_BASE_URL, st.session_state.run_id, "audit_trail")
        if trail_content:
            try:
                entries = json.loads(trail_content.decode("utf-8"))
                for entry in entries:
                    it_num = entry.get("iteration", "?")
                    cfg = entry.get("config_used", {})
                    metrics_it = entry.get("metrics", {})
                    fb = entry.get("feedback", {})

                    line_cfg = f"iteration {it_num}: CTGAN config [epochs={cfg.get('epochs')}, batch={cfg.get('batch_size')}, dims={cfg.get('generator_dim')}]"
                    append_terminal_log(line_cfg)

                    if metrics_it:
                        ks_it = metrics_it.get("per_column_ks", {})
                        mean_ks = sum(ks_it.values()) / len(ks_it) if ks_it else 0.0
                        passed_str = "PASS" if metrics_it.get("overall_passed") else "FAIL"
                        line_m = f"metrics (iter {it_num}): KS={mean_ks:.3f}, CorrDiff={metrics_it.get('correlation_diff_frobenius', 0.0):.3f}, JS={metrics_it.get('class_balance_js_divergence', 0.0):.3f} [{passed_str}]"
                        append_terminal_log(line_m)

                    if fb:
                        diag = fb.get("diagnosis", "")
                        if diag:
                            append_terminal_log(f"diagnosis (iter {it_num}): {diag[:120]}...")
                        adjustments = fb.get("config_adjustments", {})
                        if adjustments:
                            append_terminal_log(f"refinement adjustments (iter {it_num}): {adjustments}")
            except Exception:
                pass

        if not status_data.get("is_running", False):
            st.session_state.is_running = False
            if current_state_status == "completed_threshold_met":
                append_terminal_log("SUCCESS: All statistical and privacy thresholds satisfied!")
            elif current_state_status == "completed_max_iterations":
                append_terminal_log("NOTICE: Optimization stopped: reached maximum iterations budget.")
            elif current_state_status == "failed":
                append_terminal_log(f"ERROR: Execution failed: {status_data.get('error')}")

            # Reload synthetic df if ready
            synth_bytes = fetch_artifact_bytes(API_BASE_URL, st.session_state.run_id, "dataset")
            if synth_bytes:
                try:
                    st.session_state.synthetic_df = pd.read_csv(io.BytesIO(synth_bytes))
                except Exception:
                    pass

            # Trigger a final rerun to render completion or error in terminal and enable downloads
            st.rerun()

        # Brief pause then rerun while running to update UI live
        if st.session_state.is_running:
            time.sleep(1.2)
            st.rerun()

    except Exception as poll_err:
        append_terminal_log(f"polling warning: {poll_err}")
        time.sleep(1.5)
        st.rerun()


# -----------------------------------------------------------------------------
# Bottom Telemetry Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-telemetry">
        <span>Engine: <strong>CTGAN (SDV)</strong> &nbsp;|&nbsp; LLM: <strong>Groq LLaMA-3.3-70B</strong> &nbsp;|&nbsp; Audit Trail: <strong>ACTIVE</strong></span>
        <span>SyntheLoop Autonomous Synthetic Tabular Pipeline &nbsp;•&nbsp; Local Session</span>
    </div>
    """,
    unsafe_allow_html=True,
)
