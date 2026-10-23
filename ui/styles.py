"""
UI Styling and Custom CSS module.
Implements modern dark-glass aesthetics, custom badges, metric cards, and responsive components.
"""

import streamlit as st


def apply_custom_styles():
    """
    Inject custom CSS into the Streamlit app.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
        }

        /* Gradient Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.2) 0%, rgba(147, 51, 234, 0.2) 50%, rgba(59, 130, 246, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a5b4fc, #c084fc, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin: 0;
            line-height: 1.5;
        }

        /* Glassmorphic Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(8px);
            transition: all 0.2s ease-in-out;
        }

        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .badge-easy {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-medium {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .badge-hard {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .badge-rag {
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        /* Source Citation Box */
        .source-box {
            background: rgba(15, 23, 42, 0.6);
            border-left: 3px solid #6366f1;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 8px 0;
            font-size: 0.88rem;
        }

        /* Score Metric Cards */
        .metric-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #38bdf8;
            margin: 4px 0;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Streamlit Button Tuning */
        div.stButton > button:first-child {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* High-contrast Sidebar Radio Navigation */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            color: #f1f5f9 !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            padding: 8px 12px !important;
            border-radius: 8px !important;
            margin-bottom: 4px !important;
            transition: all 0.2s ease !important;
            display: flex !important;
            align-items: center !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: rgba(99, 102, 241, 0.15) !important;
            color: #c7d2fe !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
            color: #f8fafc !important;
            font-size: 0.98rem !important;
            font-weight: 600 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] [aria-checked="true"] + div p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [aria-checked="true"] p {
            color: #818cf8 !important;
            font-weight: 700 !important;
        }

        /* Status Indicator */
        .status-dot-ready {
            height: 10px;
            width: 10px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #10b981;
            margin-right: 6px;
        }

        .status-dot-empty {
            height: 10px;
            width: 10px;
            background-color: #ef4444;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #ef4444;
            margin-right: 6px;
        }

        /* =========================================================
           Hide Streamlit Deploy Button & Developer Options Menu
           (Rerun, Clear cache, Record screen, Print, etc.)
           ========================================================= */
        .stAppDeployButton,
        [data-testid="stAppDeployButton"],
        button[title="Deploy this app"],
        button[kind="header"],
        #MainMenu,
        [data-testid="stMainMenu"],
        [data-testid="stToolbarActions"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
