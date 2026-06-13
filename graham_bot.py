import streamlit as st
import pandas as pd
import sqlite3
import bcrypt
import os
import json
import re
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pypdf import PdfReader, PdfWriter

# --- CONFIGURATION ---
# On Streamlit Cloud secrets are stored in st.secrets, not in a .env file.
# Copy them into os.environ so all os.getenv() calls below work unchanged,
# both locally (where .env is used) and on Streamlit Cloud.
try:
    for _k, _v in st.secrets.items():
        if _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:
    pass  # st.secrets is empty locally — load_dotenv() handles it below

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///graham_bot.db")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL") or SMTP_USER
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="Graham-Bot | Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    [data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="metric-container"] label,
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #1a1a2e !important;
    }
    .recommendation-card {
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-weight: bold; margin-bottom: 20px;
    }
    .buy { background-color: #28a745; }
    .strong-buy { background-color: #1e7e34; border: 3px solid #ffd700; }
    .hold { background-color: #ffc107; color: black; }
    .sell { background-color: #dc3545; }
    .rec-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8em; font-weight: bold; color: white; margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CORE IMPORTS  (pure Python — no Streamlit)
# ============================================================
from core.scoring import calculate_graham_score, calculate_full_analysis
from core.email_utils import send_alert_email, send_test_email
from core.db import (
    init_db, hash_password, check_password,
    add_user, get_user,
    save_analysis, get_history, get_ticker_trend_data, calculate_cagr,
    update_analysis_market_price,
    add_to_watchlist, remove_from_watchlist, is_on_watchlist, get_watchlist_with_scores,
    add_alert, get_alerts, delete_alert,
    check_and_fire_alerts,
    is_report_processed, mark_report_processed,
)
from pipeline.cse_fetcher import (
    fetch_announcements as _cse_fetch_announcements,
    filter_reports as _cse_filter_reports,
    get_pdf_url as _cse_get_pdf_url,
    get_report_metadata as _cse_get_report_metadata,
    download_pdf as _cse_download_pdf,
)
from core.extraction import (
    _parse_json_response, _find_relevant_pages,
    detect_page_offset, _DATA_PROMPT, _TOC_KEYWORDS,
    parse_toc_locally, parse_toc_loosely,
    extract_with_openrouter as _extract_with_openrouter_headless,
)

# ============================================================
# STREAMLIT WRAPPERS FOR EXTRACTION
# (keep st.* feedback; delegate core logic to core.extraction)
# ============================================================

def _st_notify(level, msg):
    """Route core extraction log messages to Streamlit UI feedback."""
    icons = {'info': 'ℹ️', 'warning': '⚠️', 'success': '✅', 'error': '❌'}
    icon = icons.get(level, '')
    if level == 'success':
        st.success(f"{icon} {msg}")
    elif level == 'warning':
        st.warning(f"{icon} {msg}")
    elif level == 'error':
        st.error(f"{icon} {msg}")
    else:
        st.info(f"{icon} {msg}")


def extract_financial_data(uploaded_file):
    """Gemini path — uploads a cropped PDF for native PDF understanding."""
    if not API_KEY:
        st.error("Missing Gemini API Key. Please set GEMINI_API_KEY in .env")
        return None, None

    try:
        reader = PdfReader(uploaded_file)
        page_map, relevant_pages = _find_relevant_pages(
            reader, allow_gemini_toc=True, notify_fn=_st_notify
        )
        total_pages = page_map['total_pages']

        st.info(f"🔍 Uploading {len(relevant_pages)} of {total_pages} targeted pages to Gemini...")

        writer = PdfWriter()
        for p_idx in relevant_pages:
            writer.add_page(reader.pages[p_idx])
        cropped_pdf_path = "temp_report_cropped.pdf"
        with open(cropped_pdf_path, "wb") as f:
            writer.write(f)

        genai.configure(api_key=API_KEY)
        myfile = genai.upload_file(cropped_pdf_path)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([_DATA_PROMPT, myfile])
        return _parse_json_response(response.text), page_map

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")
        return None, None
    finally:
        if os.path.exists("temp_report_cropped.pdf"):
            os.remove("temp_report_cropped.pdf")


def extract_financial_data_openrouter(uploaded_file, model_id="openai/gpt-oss-120b:free"):
    """OpenRouter path — saves upload to a temp file, delegates to headless core."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        st.error("Missing OpenRouter API Key. Please set OPENROUTER_API_KEY in .env")
        return None, None

    tmp_path = "temp_openrouter_upload.pdf"
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        uploaded_file.seek(0)

        with st.spinner("Reading contents page and extracting text..."):
            raw, page_map = _extract_with_openrouter_headless(tmp_path, model_id, openrouter_key)

        if raw is None:
            st.error(
                "❌ Extraction failed. If this was a rate limit, wait a minute and try again "
                "or choose a different model."
            )
        return raw, page_map

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")
        return None, None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# SESSION STATE & DB INIT
# ============================================================
for key, default in [('logged_in', False), ('user', None), ('last_analysis', None),
                     ('extracted_data', None), ('page_map', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

init_db()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🛡️ Graham-Bot")
    st.markdown("---")
    if not st.session_state.logged_in:
        auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"])
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        if auth_mode == "Login":
            if st.button("Login"):
                user = get_user(username_input)
                if user and check_password(password_input, user[1]):
                    st.session_state.logged_in = True
                    st.session_state.user = username_input
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            if st.button("Sign Up"):
                if add_user(username_input, password_input):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already taken")
    else:
        st.success(f"Welcome, {st.session_state.user}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.last_analysis = None
            st.session_state.extracted_data = None
            st.session_state.page_map = None
            st.rerun()
    st.markdown("---")
    st.info("Built on Benjamin Graham's 'The Intelligent Investor' principles.")

# ============================================================
# MAIN APP
# ============================================================
ALERT_TYPE_LABELS = {
    "score_above": "Graham Score rises to or above X",
    "score_below": "Graham Score falls below X",
    "mos_above":   "Margin of Safety rises to or above X%",
    "mos_below":   "Margin of Safety falls below X%",
}

if not st.session_state.logged_in:
    st.title("🚀 Graham-Bot Financial Analysis")
    st.markdown("""
    Welcome to **Graham-Bot**, your AI-powered value investing assistant.

    ### How it works:
    1. **Upload** a company's annual report PDF.
    2. **AI Extraction**: Gemini reads the Table of Contents, finds the financial pages, and extracts key metrics.
    3. **Scoring**: Benjamin Graham's rules generate a 0–15 score.
    4. **Recommendation**: Clear Buy/Hold/Sell signal with Margin of Safety.

    **Please login or sign up to continue.**
    """)
else:
    user = st.session_state.user
    st.title("📈 Market Analysis Dashboard")

    # ── Reusable full analysis renderer (used by Analyze tab + History tab) ──
    def _render_analysis_report(a, ticker_val, company_val, key_prefix=""):
        def _na(v, fmt=".1f", suffix=""):
            if v is None: return "N/A"
            return f"{v:{fmt}}{suffix}"

        if not a.get('market_price'):
            st.warning(
                "⚠️ **Market price not available** — P/E, P/B, P/CF and Margin of Safety "
                "cannot be computed. Enter the current market price below the analysis to recalculate."
            )

        grade = a['graham_grade']
        score = a['graham_score']
        rec   = a['recommendation']
        grade_colors = {"A": "#1e7e34", "B": "#28a745", "C": "#ffc107", "D": "#dc3545"}
        grade_color  = grade_colors.get(grade, "#6c757d")
        st.markdown(
            f"""<div style="background:{grade_color};color:{'black' if grade=='C' else 'white'};
            padding:18px 24px;border-radius:12px;margin-bottom:16px;">
            <span style="font-size:2em;font-weight:bold">Grade {grade}</span>
            &nbsp;&nbsp;
            <span style="font-size:1.3em">{company_val} ({ticker_val})</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="font-size:1.2em">Score: {score}/15</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="font-size:1.2em">{rec}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if ticker_val:
            on_wl = is_on_watchlist(user, ticker_val)
            wl_label = "★ Remove from Watchlist" if on_wl else "☆ Add to Watchlist"
            if st.button(wl_label, key=f"{key_prefix}wl_btn"):
                if on_wl:
                    remove_from_watchlist(user, ticker_val)
                else:
                    add_to_watchlist(user, ticker_val, company_val)
                st.rerun()

        st.markdown("#### 1. Growth Rates")
        g1, g2, g3 = st.columns(3)
        with g1:
            rv = a['revenue_growth_pct']
            st.metric("Revenue Growth", _na(rv, ".1f", "%"), delta=f"{rv:+.1f}%" if rv is not None else None)
        with g2:
            pg = a['profit_growth_pct']
            st.metric("Profit Growth", _na(pg, ".1f", "%"), delta=f"{pg:+.1f}%" if pg is not None else None)
        with g3:
            eg = a['eps_growth_pct']
            st.metric("EPS Growth", _na(eg, ".1f", "%"), delta=f"{eg:+.1f}%" if eg is not None else None)

        st.markdown("#### 2 & 3. Valuation & Profitability Ratios")
        v1, v2, v3, v4, v5, v6, v7 = st.columns(7)
        with v1: st.metric("P/E Ratio",     _na(a['pe'], ".1f"))
        with v2: st.metric("P/B Ratio",     _na(a['pb'], ".2f"))
        with v3: st.metric("P/CF Ratio",    _na(a['pcf'], ".1f"))
        with v4: st.metric("ROE",           _na(a['roe'], ".1f", "%"))
        with v5: st.metric("ROA",           _na(a['roa'], ".1f", "%"))
        with v6: st.metric("Current Ratio", _na(a['current_ratio'], ".2f"))
        with v7: st.metric("Debt/Equity",   _na(a['de'], ".2f"))

        st.markdown("#### 4. Dividend Metrics")
        d1, d2 = st.columns(2)
        with d1: st.metric("Dividend Yield", _na(a['div_yield'], ".1f", "%"))
        with d2: st.metric("Payout Ratio",   _na(a['payout_ratio'], ".1f", "%"))

        st.markdown("#### 5. Graham Score Table (max 15 pts)")
        score_df = pd.DataFrame(a['score_table'])[["Criterion", "Value", "Points", "Score"]]
        st.dataframe(
            score_df, use_container_width=True, hide_index=True,
            column_config={"Score": st.column_config.ProgressColumn("Score", max_value=3, format="%d")},
        )
        st.markdown(f"**Total: {score}/15 — Grade {grade}**")

        st.markdown("#### 6. Margin of Safety (Sri Lanka Formula)")
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Intrinsic Value (LKR)", f"{a['intrinsic_value']:.2f}", help="EPS × 7.4  (g=5%, Y=11%)")
        with m2: st.metric("Market Price (LKR)",    f"{a['market_price']:.2f}" if a['market_price'] > 0 else "N/A")
        with m3: st.metric("Margin of Safety",      f"{a['mos_pct']:.1f}%")
        with m4: st.metric("Max Buy Price (30% MOS)", f"{a['max_buy_price']:.2f}")
        st.info(f"MOS Grade: **{a['mos_grade']}**")

        st.markdown("#### 7. Defensive Investor Checklist")
        def_df = pd.DataFrame(a['defensive_checklist'])[["Criterion", "Condition", "Result"]]
        st.dataframe(def_df, use_container_width=True, hide_index=True)
        passes = a['defensive_passes']
        st.markdown(
            f"**{passes}/5 criteria met — {a['defensive_verdict']}** "
            f"({'Passes' if passes >= 4 else 'Fails'} Defensive Investor test)"
        )

        st.markdown("#### 8. Final Recommendation")
        buy = a['buy_decision']
        buy_color      = "#1e7e34" if "YES" in buy else ("#ffc107" if "HOLD" in buy else "#dc3545")
        buy_text_color = "black" if "HOLD" in buy else "white"
        alloc_lkr = a['allocation_pct'] / 100 * 50000
        st.markdown(
            f"""<div style="background:{buy_color};color:{buy_text_color};
            padding:16px;border-radius:10px;margin-bottom:12px;">
            <b style="font-size:1.3em">{buy}</b><br>
            Graham Grade: <b>{grade}</b> &nbsp;|&nbsp;
            Max Buy Price: <b>LKR {a['max_buy_price']:.2f}</b> &nbsp;|&nbsp;
            Allocation: <b>{a['allocation_pct']}% of LKR 50,000 = LKR {alloc_lkr:,.0f}</b>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("#### 9. Bottom Line")
        st.info(a['bottom_line'])

        with st.expander("Raw extracted data (JSON)", expanded=False):
            st.json(a)

    def _cse_analyze_url(pdf_url: str, ticker: str, company: str,
                         report_type: str, fiscal_year: str, model_id: str):
        """Download a CSE PDF, extract, score, and save. Shows result inline."""
        import tempfile, os as _os
        if is_report_processed(pdf_url):
            st.warning(f"⚠️ This report ({ticker}) is already saved in the database.")
            return
        if not OPENROUTER_API_KEY:
            st.error("OPENROUTER_API_KEY not configured. Cannot run extraction.")
            return
        tmp_path = None
        try:
            with st.spinner(f"Downloading {ticker} PDF…"):
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                _os.close(tmp_fd)
                ok = _cse_download_pdf(pdf_url, tmp_path)
            if not ok:
                st.error("PDF download failed. Check the URL and try again.")
                return
            with st.spinner(f"Extracting financial data from {ticker}… (30–90 s)"):
                raw, page_map = _extract_with_openrouter_headless(tmp_path, model_id, OPENROUTER_API_KEY)
        finally:
            if tmp_path and _os.path.exists(tmp_path):
                _os.remove(tmp_path)

        if raw is None:
            st.error("Extraction failed — PDF may be scanned or unreadable.")
            return

        raw["ticker"] = ticker
        raw["company_name"] = company
        raw["report_type"] = report_type
        if not raw.get("fiscal_year") and fiscal_year:
            raw["fiscal_year"] = fiscal_year

        analysis = calculate_full_analysis(raw)
        merged = {**raw, "_analysis": analysis}
        aid = save_analysis(user, company, ticker, merged,
                            analysis["graham_score"], analysis["recommendation"], source="manual")
        mark_report_processed(ticker, company, report_type,
                               str(raw.get("fiscal_year", fiscal_year)), pdf_url, aid)
        st.success(
            f"✅ **{company} ({ticker})** saved — "
            f"Score: {analysis['graham_score']}/15  |  {analysis['recommendation']}"
        )
        _render_analysis_report(analysis, ticker, company, key_prefix=f"cse_{ticker}_")

    tabs = st.tabs(["Analyze New Report", "5-Year Trends", "Watchlist", "Alerts", "Analysis History", "CSE Reports"])

    # ── TAB 1: ANALYZE ────────────────────────────────────────
    with tabs[0]:
        st.subheader("Upload PDF Financial Report")
        uploaded_file = st.file_uploader("Drop PDF here", type="pdf")

        if uploaded_file:
            _MODELS = {
                "Gemini 2.0 Flash": None,
                "GPT-OSS 120B — OpenRouter (Free)": "openai/gpt-oss-120b:free",
                "Gemma 4 31B — OpenRouter (Free)": "google/gemma-4-31b-it:free",
                "DeepSeek V4 Flash — OpenRouter (Free)": "deepseek/deepseek-v4-flash:free",
            }
            col_model, col_btn = st.columns([3, 1])
            with col_model:
                selected_model = st.selectbox(
                    "AI Model",
                    list(_MODELS.keys()),
                    label_visibility="collapsed",
                )
            with col_btn:
                run_analysis = st.button("Run Analysis", use_container_width=True)

            if run_analysis:
                # Clear any previous state so the edit form always shows fresh
                st.session_state.last_analysis = None
                st.session_state.extracted_data = None
                st.session_state.page_map = None
                with st.spinner("Reading table of contents and extracting financial data..."):
                    model_id = _MODELS[selected_model]
                    if model_id is None:
                        raw, page_map = extract_financial_data(uploaded_file)
                    else:
                        raw, page_map = extract_financial_data_openrouter(uploaded_file, model_id)
                    if raw:
                        st.session_state.extracted_data = raw
                        st.session_state.page_map = page_map

        # ── PHASE 2: EDIT FORM (shown after extraction, before saving) ──
        if st.session_state.extracted_data and not st.session_state.last_analysis:
            raw = st.session_state.extracted_data

            def _clean_ticker(v):
                s = str(v or '').strip()
                return '' if s in ('0', 'N/A', 'NA', 'n/a', 'none', 'None') else s

            def _flt(v):
                try: return float(v or 0)
                except: return 0.0

            ticker_missing = _clean_ticker(raw.get('ticker')) == ''

            st.markdown("---")
            st.subheader("Review & Edit Extracted Data")
            st.caption("AI extraction is not always perfect — check the values below before saving. "
                       "Correct any zeros or wrong numbers, then click **Confirm & Score**.")

            # ── Page map expander ────────────────────────────────
            pm = st.session_state.get('page_map')
            if pm:
                method_label = {
                    'toc':          '📋 Table of Contents',
                    'keyword_scan': '🔍 Keyword Scan (no TOC found)',
                    'fallback':     '⚠️ Fallback — first 15 pages',
                }.get(pm['method'], pm['method'])

                pages_sent_str = ', '.join(str(p) for p in pm['pages_sent'])
                expander_title = (f"📄 Pages sent to Gemini — {len(pm['pages_sent'])} of "
                                  f"{pm['total_pages']} total  |  Method: {method_label}")

                with st.expander(expander_title, expanded=False):
                    if pm['method'] == 'toc' and pm['sections']:
                        rows = []
                        for sec in pm['sections']:
                            rows.append({
                                'Section (from TOC)': sec['name'],
                                'TOC Page': sec['toc_page'],
                                'PDF Pages Read': ', '.join(str(p) for p in sec['pdf_pages']),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    elif pm['method'] == 'keyword_scan' and pm['sections']:
                        rows = []
                        for sec in pm['sections']:
                            rows.append({
                                'Matched Keyword': sec['name'],
                                'PDF Pages Read': ', '.join(str(p) for p in sec['pdf_pages']),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.write(f"Pages: {pages_sent_str}")

                    st.caption(f"All pages sent (1-based): {pages_sent_str}")

            if ticker_missing:
                st.error("⚠️ Ticker symbol was not found in the report. "
                         "Please enter it manually — it is required to track this company's history.")

            with st.form("edit_extracted_data"):
                # ── Row 1: Company identity ───────────────────────────
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    company_name = st.text_input("Company Name *", value=raw.get('company_name', ''))
                with c2:
                    ticker_default = _clean_ticker(raw.get('ticker'))
                    ticker = st.text_input(
                        "Ticker Symbol *" + ("  🔴 required" if ticker_missing else ""),
                        value=ticker_default, placeholder="e.g. SAMP, AAPL",
                        help="Used as the unique company ID in the database."
                    )
                with c3:
                    fiscal_year = st.text_input("Fiscal Year", value=str(raw.get('fiscal_year', '')))

                # ── Current Year P&L ──────────────────────────────────
                st.markdown("##### Current Year — Income Statement")
                c4, c5, c6 = st.columns(3)
                with c4:
                    revenue = st.number_input("Revenue (Millions)", value=_flt(raw.get('revenue')),
                                              min_value=0.0, format="%.2f")
                with c5:
                    net_income = st.number_input("Net Income (Millions)", value=_flt(raw.get('net_income')),
                                                 format="%.2f")
                with c6:
                    eps = st.number_input("EPS (Basic)", value=_flt(raw.get('eps')), format="%.4f")

                # ── Previous Year P&L ─────────────────────────────────
                st.markdown("##### Previous Year — Income Statement (for Growth Rates)")
                c7, c8, c9 = st.columns(3)
                with c7:
                    revenue_prev = st.number_input("Revenue Prev Year (M)", value=_flt(raw.get('revenue_prev')),
                                                   min_value=0.0, format="%.2f")
                with c8:
                    net_income_prev = st.number_input("Net Income Prev Year (M)",
                                                      value=_flt(raw.get('net_income_prev')), format="%.2f")
                with c9:
                    eps_prev = st.number_input("EPS Prev Year", value=_flt(raw.get('eps_prev')), format="%.4f")

                # ── Balance Sheet ─────────────────────────────────────
                st.markdown("##### Balance Sheet")
                c10, c11, c12, c13, c14 = st.columns(5)
                with c10:
                    total_assets = st.number_input("Total Assets (M)", value=_flt(raw.get('total_assets')),
                                                   min_value=0.0, format="%.2f")
                with c11:
                    total_equity = st.number_input("Total Equity (M)", value=_flt(raw.get('total_equity')),
                                                   min_value=0.0, format="%.2f")
                with c12:
                    total_liabilities = st.number_input("Total Liabilities (M)",
                                                        value=_flt(raw.get('total_liabilities')),
                                                        min_value=0.0, format="%.2f")
                with c13:
                    current_assets = st.number_input("Current Assets (M)", value=_flt(raw.get('current_assets')),
                                                     min_value=0.0, format="%.2f")
                with c14:
                    current_liabilities = st.number_input("Current Liabilities (M)",
                                                          value=_flt(raw.get('current_liabilities')),
                                                          min_value=0.0, format="%.2f")

                # ── Market & Dividend Data ────────────────────────────
                st.markdown("##### Market Data & Dividends")
                c15, c16, c17, c18 = st.columns(4)
                with c15:
                    market_price = st.number_input("Market Price (LKR)", value=_flt(raw.get('market_price')),
                                                   min_value=0.0, format="%.2f",
                                                   help="Year-end closing price per share")
                with c16:
                    shares_outstanding = st.number_input("Shares Outstanding (M)",
                                                         value=_flt(raw.get('shares_outstanding')),
                                                         min_value=0.0, format="%.4f")
                with c17:
                    dividend_per_share = st.number_input("Dividend Per Share (LKR)",
                                                         value=_flt(raw.get('dividend_per_share')),
                                                         min_value=0.0, format="%.4f")
                with c18:
                    operating_cf = st.number_input("Operating Cash Flow (M)",
                                                   value=_flt(raw.get('operating_cash_flow')),
                                                   format="%.2f")

                # ── AI-extracted fallback ratios (used when raw inputs are missing) ──
                with st.expander("AI-Extracted Ratios (optional — override only if auto-calc is wrong)"):
                    fa1, fa2, fa3, fa4, fa5, fa6 = st.columns(6)
                    with fa1:
                        pe_ratio = st.number_input("P/E Ratio", value=_flt(raw.get('pe_ratio')),
                                                   min_value=0.0, format="%.2f")
                    with fa2:
                        pb_ratio = st.number_input("P/B Ratio", value=_flt(raw.get('pb_ratio')),
                                                   min_value=0.0, format="%.2f")
                    with fa3:
                        roe = st.number_input("ROE (%)", value=_flt(raw.get('roe')), format="%.2f")
                    with fa4:
                        debt_to_equity = st.number_input("Debt/Equity", value=_flt(raw.get('debt_to_equity')),
                                                         min_value=0.0, format="%.4f")
                    with fa5:
                        earnings_growth = st.number_input("5yr EPS Growth (%)",
                                                          value=_flt(raw.get('earnings_growth_5yr')),
                                                          format="%.2f")
                    with fa6:
                        div_index = 0 if raw.get('dividend_paid') == 'Yes' else 1
                        dividend_paid = st.selectbox("Dividend Paid", ["Yes", "No"], index=div_index)

                submitted = st.form_submit_button("✅ Confirm & Score", use_container_width=True)

            if submitted:
                ticker_clean = ticker.strip().upper()
                if not ticker_clean:
                    st.error("Ticker symbol is required. Please enter it above and click Confirm again.")
                else:
                    edited_data = {
                        'company_name': company_name,
                        'ticker': ticker_clean,
                        'fiscal_year': fiscal_year,
                        # Current year
                        'revenue': revenue,
                        'net_income': net_income,
                        'eps': eps,
                        # Previous year
                        'revenue_prev': revenue_prev,
                        'net_income_prev': net_income_prev,
                        'eps_prev': eps_prev,
                        # Balance sheet
                        'total_assets': total_assets,
                        'total_equity': total_equity,
                        'total_liabilities': total_liabilities,
                        'current_assets': current_assets,
                        'current_liabilities': current_liabilities,
                        # Market & dividends
                        'market_price': market_price,
                        'shares_outstanding': shares_outstanding,
                        'dividend_per_share': dividend_per_share,
                        'operating_cash_flow': operating_cf,
                        # AI fallbacks
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'roe': roe,
                        'debt_to_equity': debt_to_equity,
                        'earnings_growth_5yr': earnings_growth,
                        'dividend_paid': dividend_paid,
                    }
                    analysis = calculate_full_analysis(edited_data)
                    score = analysis['graham_score']
                    rec   = analysis['recommendation']
                    mos   = analysis['mos_pct']
                    save_analysis(user, company_name, ticker_clean, edited_data, score, rec)
                    st.session_state.last_analysis = {
                        'data': edited_data,
                        'full': analysis,
                        'ticker': ticker_clean,
                        'company': company_name,
                    }
                    st.session_state.extracted_data = None
                    fired = check_and_fire_alerts(user, ticker_clean, company_name, score, mos)
                    for f in fired:
                        st.success(f"📧 Alert fired: {f}")
                    st.rerun()

        # ── PHASE 3: RESULTS (shown after Confirm & Score) ──────
        if st.session_state.last_analysis:
            la = st.session_state.last_analysis
            ticker_val  = la['ticker']
            company_val = la['company']
            a = la['full'] if 'full' in la else calculate_full_analysis(la.get('data', {}))
            st.markdown("---")
            _render_analysis_report(a, ticker_val, company_val, key_prefix="analyze_")

    # ── TAB 2: 5-YEAR TRENDS ──────────────────────────────────
    with tabs[1]:
        st.subheader("5-Year Financial Trends")
        history_df = get_history(user, include_auto=True)
        if history_df.empty:
            st.info("No analysis history yet. Upload reports to start tracking trends.")
        else:
            tickers = sorted(history_df['ticker'].unique().tolist())
            selected_ticker = st.selectbox("Select Company", tickers, key="trend_ticker")
            company_label = history_df[history_df['ticker'] == selected_ticker]['company_name'].iloc[-1]
            trend_df = get_ticker_trend_data(user, selected_ticker)

            if trend_df.empty:
                st.info("No parseable data for this ticker.")
            elif len(trend_df) == 1:
                st.info(f"Only one report for {selected_ticker}. Upload more annual reports to see trends.")
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
            else:
                st.caption(f"**{company_label}** ({selected_ticker}) — {len(trend_df)} reports")

                # 2×2 metric chart grid
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Revenue (Millions)", "Net Income (Millions)", "EPS", "ROE (%)"),
                    vertical_spacing=0.18, horizontal_spacing=0.12
                )
                for (col_name, row, col, color) in [
                    ('revenue',    1, 1, '#2196F3'),
                    ('net_income', 1, 2, '#4CAF50'),
                    ('eps',        2, 1, '#FF9800'),
                    ('roe',        2, 2, '#9C27B0'),
                ]:
                    fig.add_trace(go.Scatter(
                        x=trend_df['fiscal_year'], y=trend_df[col_name],
                        mode='lines+markers+text',
                        text=[f"{v:.1f}" for v in trend_df[col_name]],
                        textposition='top center',
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ), row=row, col=col)
                fig.update_layout(height=550,
                                  title_text=f"{company_label} — Financial Metrics Over Time",
                                  title_font_size=16)
                st.plotly_chart(fig, use_container_width=True)

                # Graham Score trend
                score_fig = px.line(
                    trend_df, x='fiscal_year', y='score',
                    title="Graham Score Trend", markers=True, text='score',
                    color_discrete_sequence=['#FFD700']
                )
                score_fig.update_traces(textposition='top center')
                score_fig.update_layout(yaxis=dict(range=[0, 15]), height=300)
                st.plotly_chart(score_fig, use_container_width=True)

                # CAGR summary table
                st.subheader("Trend Analysis Summary")
                start_yr = trend_df['fiscal_year'].iloc[0]
                end_yr = trend_df['fiscal_year'].iloc[-1]
                summary_rows = []
                for col_name, label, unit in [
                    ('revenue', 'Revenue', 'M'), ('net_income', 'Net Income', 'M'),
                    ('eps', 'EPS', ''), ('roe', 'ROE', '%')
                ]:
                    vals = trend_df[col_name].tolist()
                    cagr = calculate_cagr(vals)
                    neg_years = sum(1 for v in vals if v < 0)
                    down_years = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1])
                    summary_rows.append({
                        'Metric': f"{label} ({unit})" if unit else label,
                        start_yr: f"{vals[0]:.2f}",
                        end_yr: f"{vals[-1]:.2f}",
                        'CAGR': f"{cagr:.1f}%" if cagr is not None else "N/A",
                        'Negative Years': neg_years,
                        'Declining Years': down_years,
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── TAB 3: WATCHLIST ──────────────────────────────────────
    with tabs[2]:
        st.subheader("My Watchlist")
        watchlist_items = get_watchlist_with_scores(user)

        if not watchlist_items:
            st.info("Your watchlist is empty. Run an analysis then click 'Add to Watchlist'.")
        else:
            st.caption(f"{len(watchlist_items)} companies tracked")
            for i in range(0, len(watchlist_items), 3):
                cols = st.columns(3)
                for j, item in enumerate(watchlist_items[i:i + 3]):
                    with cols[j]:
                        delta_str = None
                        if item['score_change'] is not None:
                            sign = "+" if item['score_change'] > 0 else ""
                            delta_str = f"{sign}{item['score_change']} pts"

                        with st.container(border=True):
                            st.markdown(f"**{item['company_name']}**")
                            st.caption(f"{item['ticker']}  ·  Added {item['added_date']}")

                            if item['last_score'] is not None:
                                sig_change = item['score_change'] is not None and abs(item['score_change']) >= 2
                                delta_color = "normal"
                                if sig_change and item['score_change'] < 0:
                                    delta_color = "inverse"
                                st.metric("Graham Score", f"{item['last_score']}/15",
                                          delta=delta_str, delta_color=delta_color)
                                if item['last_mos'] is not None:
                                    st.metric("Margin of Safety", f"{item['last_mos']}%")
                                rec = item['last_rec'] or ''
                                rec_class = rec.lower().replace(" ", "-")
                                st.markdown(
                                    f'<div class="recommendation-card {rec_class}" '
                                    f'style="padding:8px;font-size:0.82em;">{rec}</div>',
                                    unsafe_allow_html=True)
                                st.caption(f"Last analyzed: {item['last_analyzed']}")
                                if sig_change:
                                    arrow = "📈" if item['score_change'] > 0 else "📉"
                                    st.warning(f"{arrow} Score changed by {delta_str} since last report")
                            else:
                                st.warning("No analysis data yet")

                            if st.button("Remove", key=f"wl_remove_{item['ticker']}"):
                                remove_from_watchlist(user, item['ticker'])
                                st.rerun()

    # ── TAB 4: ALERTS ─────────────────────────────────────────
    with tabs[3]:
        st.subheader("Price & Score Alerts")

        with st.expander("➕ Add New Alert", expanded=True):
            all_tickers_df = get_history(user, include_auto=True)
            ticker_choices = sorted(all_tickers_df['ticker'].unique().tolist()) if not all_tickers_df.empty else []

            col_a, col_b = st.columns(2)
            with col_a:
                if ticker_choices:
                    alert_ticker = st.selectbox("Company (Ticker)", ticker_choices, key="alert_ticker_sel")
                    h = get_history(user, alert_ticker, include_auto=True)
                    alert_company = h['company_name'].iloc[-1] if not h.empty else alert_ticker
                else:
                    alert_ticker = st.text_input("Ticker Symbol", key="alert_ticker_txt").upper()
                    alert_company = st.text_input("Company Name", key="alert_company_txt")

                alert_type = st.selectbox(
                    "Alert Condition",
                    list(ALERT_TYPE_LABELS.keys()),
                    format_func=lambda k: ALERT_TYPE_LABELS[k],
                    key="alert_type_sel"
                )

            with col_b:
                if "score" in alert_type:
                    threshold = st.number_input("Score Threshold (0–15)", 0, 15, 9, key="alert_thresh")
                else:
                    threshold = st.number_input("MOS Threshold (%)", -200, 200, 20, key="alert_thresh_mos")
                alert_email = st.text_input("Send email to", placeholder="you@example.com", key="alert_email")

            c1, c2 = st.columns([2, 1])
            with c1:
                if st.button("Save Alert"):
                    if alert_ticker and alert_email:
                        add_alert(user, alert_ticker, alert_company, alert_type, threshold, alert_email)
                        st.success(f"Alert saved for {alert_ticker}.")
                        st.rerun()
                    else:
                        st.error("Please enter a ticker and email address.")
            with c2:
                if st.button("Send Test Email"):
                    if alert_email:
                        ok, msg = send_test_email(alert_email)
                        st.success("Test email sent!") if ok else st.error(f"Failed: {msg}")
                    else:
                        st.warning("Enter an email address first.")

        if not SMTP_USER:
            st.warning("⚠️ SMTP not configured. Add SMTP_USER and SMTP_PASSWORD to your .env to enable emails.")

        st.markdown("---")
        st.subheader("Your Alerts")
        alerts_df = get_alerts(user)
        if alerts_df.empty:
            st.info("No alerts configured yet.")
        else:
            for _, alert_row in alerts_df.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        condition = ALERT_TYPE_LABELS.get(alert_row['alert_type'], alert_row['alert_type'])
                        thresh_label = (f"{int(alert_row['threshold'])}"
                                        if "score" in alert_row['alert_type']
                                        else f"{alert_row['threshold']}%")
                        status = "🟢 Active" if alert_row['active'] else "⏸ Paused"
                        last_t = alert_row['last_triggered'] or "Never"
                        st.markdown(f"**{alert_row['ticker']}** — {alert_row['company_name']}")
                        st.caption(f"{condition} **{thresh_label}**  ·  {alert_row['email']}  ·  {status}  ·  Last triggered: {last_t}")
                    with c2:
                        if st.button("Delete", key=f"del_alert_{alert_row['id']}"):
                            delete_alert(alert_row['id'])
                            st.rerun()
            st.caption("💡 Alerts fire automatically every time you run an analysis for that company.")

    # ── TAB 5: HISTORY ────────────────────────────────────────
    with tabs[4]:
        st.subheader("Analysis History")
        show_auto = st.checkbox("Include auto-analyzed reports (pipeline)", value=True, key="hist_show_auto")
        history_df = get_history(user, include_auto=show_auto)

        if history_df.empty:
            st.info("No analysis records found.")
        else:
            history_df = history_df.reset_index(drop=True)
            display_df = history_df[['date', 'company_name', 'ticker', 'score', 'recommendation', 'source']].copy()
            display_df['source'] = display_df['source'].fillna('manual').replace(
                {'manual': 'Manual', 'auto': 'Auto'}
            )
            _rt_map = {"annual": "Annual", "quarterly": "Quarterly", "financial": "Financial"}
            display_df['type'] = history_df['data_json'].apply(
                lambda s: _rt_map.get(
                    (json.loads(s) if isinstance(s, str) else s).get("report_type", ""), "Annual"
                ) if s else "Annual"
            )

            st.caption(f"{len(history_df)} records — click a row to view the full analysis")
            event = st.dataframe(
                display_df[['date', 'company_name', 'ticker', 'type', 'score', 'recommendation', 'source']],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "type": st.column_config.TextColumn("Type"),
                    "source": st.column_config.TextColumn("Source"),
                    "score": st.column_config.NumberColumn("Score", format="%d/15"),
                },
            )

            selected_rows = event.selection.rows if hasattr(event, 'selection') else []
            if selected_rows:
                idx = selected_rows[0]
                row = history_df.iloc[idx]
                st.markdown("---")
                st.caption(f"Analyzed: {row['date']}  ·  Source: {row.get('source', 'manual')}")
                try:
                    data = json.loads(row['data_json']) if isinstance(row['data_json'], str) else row['data_json']
                    analysis_id = int(row['id']) if 'id' in row.index else None

                    # ── Market price input (shown when price is missing) ──────
                    stored_price = float(data.get('market_price') or 0)
                    price_key = f"hist_price_{idx}"
                    if price_key not in st.session_state:
                        st.session_state[price_key] = stored_price

                    if stored_price == 0:
                        st.info(
                            "**Market price not in this report.** "
                            "Enter the current CSE market price to compute P/E, P/B, and Margin of Safety."
                        )
                        mp_col, btn_col = st.columns([3, 1])
                        with mp_col:
                            entered_price = st.number_input(
                                "Current Market Price (LKR)",
                                min_value=0.0, step=0.10, format="%.2f",
                                value=st.session_state[price_key],
                                key=f"hist_price_input_{idx}",
                            )
                        with btn_col:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("Save Price", key=f"hist_save_price_{idx}",
                                         disabled=(entered_price <= 0 or analysis_id is None)):
                                if update_analysis_market_price(analysis_id, entered_price):
                                    st.session_state[price_key] = entered_price
                                    st.success(f"Saved LKR {entered_price:.2f} — analysis updated.")
                                    st.rerun()
                                else:
                                    st.error("Could not update database record.")
                        # Apply entered price to live computation
                        if entered_price > 0:
                            data = {**data, 'market_price': entered_price}

                    # Recompute with latest data (uses saved or entered price)
                    a = calculate_full_analysis(data)
                    _render_analysis_report(
                        a,
                        str(row['ticker']),
                        str(row['company_name']),
                        key_prefix=f"hist_{idx}_",
                    )
                except Exception as e:
                    st.error(f"Could not render analysis: {e}")
                    with st.expander("Raw data_json"):
                        st.json(data if 'data' in dir() else {})

    # ── TAB 6: CSE REPORTS ────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Browse & Add CSE Reports")

        _CSE_MODELS_TAB = {
            "GPT-OSS 120B (Free)": "openai/gpt-oss-120b:free",
            "Gemma 4 31B (Free)": "google/gemma-4-31b-it:free",
            "DeepSeek V4 Flash (Free)": "deepseek/deepseek-v4-flash:free",
        }
        cse_model_id = _CSE_MODELS_TAB[
            st.selectbox("AI Model for extraction", list(_CSE_MODELS_TAB.keys()), key="cse_model_sel")
        ]

        # ── SECTION A: Live CSE feed ──────────────────────────────────
        st.markdown("#### Live CSE Announcements")
        st.caption("Latest financial reports published on the Colombo Stock Exchange")

        if st.button("🔄 Refresh feed", key="cse_refresh_btn"):
            st.rerun()

        try:
            _cse_raw_items = _cse_fetch_announcements()
            _cse_feed = _cse_filter_reports(_cse_raw_items)
        except Exception as _e:
            st.error(f"Could not fetch CSE feed: {_e}")
            _cse_feed = []

        if _cse_feed:
            _feed_rows = []
            for _item in _cse_feed:
                _meta = _cse_get_report_metadata(_item)
                _purl = _cse_get_pdf_url(_item)
                _in_db = is_report_processed(_purl)
                _feed_rows.append({
                    "Company": _meta["company_name"],
                    "Ticker": _meta["ticker"],
                    "Type": _meta["report_type"].title(),
                    "Year": _meta["fiscal_year"],
                    "Published": _item.get("uploadedDate", "")[:11],
                    "Status": "✅ In DB" if _in_db else "🆕 New",
                    "_url": _purl,
                    "_ticker": _meta["ticker"],
                    "_company": _meta["company_name"],
                    "_rtype": _meta["report_type"],
                    "_fyear": _meta["fiscal_year"],
                    "_in_db": _in_db,
                })
            _feed_df = pd.DataFrame(_feed_rows)
            _feed_event = st.dataframe(
                _feed_df[["Company", "Ticker", "Type", "Year", "Published", "Status"]],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={"Status": st.column_config.TextColumn("Status", width="small")},
                key="cse_feed_table",
            )
            _feed_sel = _feed_event.selection.rows if hasattr(_feed_event, "selection") else []
            if _feed_sel:
                _fr = _feed_df.iloc[_feed_sel[0]]
                st.markdown("---")
                if _fr["_in_db"]:
                    st.info(f"✅ **{_fr['_ticker']} — {_fr['Company']}** is already in the database. "
                            f"Open **Analysis History** to view it.")
                else:
                    st.info(f"Selected: **{_fr['_ticker']} — {_fr['Company']}** "
                            f"({_fr['Type']} {_fr['Year']}, published {_fr['Published']})")
                    if st.button("⚡ Analyze & Save to DB", key="cse_feed_analyze_btn"):
                        _cse_analyze_url(
                            pdf_url=_fr["_url"],
                            ticker=_fr["_ticker"],
                            company=_fr["_company"],
                            report_type=_fr["_rtype"],
                            fiscal_year=_fr["_fyear"],
                            model_id=cse_model_id,
                        )
        else:
            st.info("No recent financial reports on the CSE feed right now.")

        # ── SECTION B: Add by PDF URL ─────────────────────────────────
        st.markdown("---")
        st.markdown("#### Add Any Report by URL")
        st.caption(
            "Visit the CSE website, open a company profile, right-click a report link "
            "and copy the PDF URL, then paste it below."
        )
        _lc1, _lc2 = st.columns(2)
        with _lc1:
            st.link_button(
                "📋 CSE Company Directory",
                "https://www.cse.lk/listed-entities/listed-company-directory?page=ALPHABETICAL",
            )
        with _lc2:
            st.link_button(
                "🔍 Example: HAYC Profile",
                "https://www.cse.lk/company-profile?symbol=HAYC.N0000",
            )

        _url_in = st.text_input(
            "PDF URL",
            placeholder="https://cdn.cse.lk/cmt/upload_report_file/...",
            key="cse_url_in",
        )
        if _url_in.strip():
            _uc1, _uc2, _uc3 = st.columns(3)
            with _uc1:
                _url_company = st.text_input("Company Name *", key="cse_url_company")
            with _uc2:
                _url_ticker = st.text_input("Ticker Symbol *",
                                            placeholder="e.g. HAYC", key="cse_url_ticker")
            with _uc3:
                _url_rtype = st.selectbox("Report Type",
                                          ["annual", "quarterly", "financial"], key="cse_url_rtype")
            if st.button("⚡ Download & Analyze", key="cse_url_btn"):
                if not _url_company.strip() or not _url_ticker.strip():
                    st.error("Company Name and Ticker Symbol are required.")
                else:
                    _cse_analyze_url(
                        pdf_url=_url_in.strip(),
                        ticker=_url_ticker.strip().upper(),
                        company=_url_company.strip(),
                        report_type=_url_rtype,
                        fiscal_year="",
                        model_id=cse_model_id,
                    )

# --- FOOTER ---
st.markdown("---")
st.caption("Graham-Bot is for educational purposes only. Always conduct your own research before investing.")
