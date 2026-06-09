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
from core.scoring import calculate_graham_score
from core.email_utils import send_alert_email, send_test_email
from core.db import (
    init_db, hash_password, check_password,
    add_user, get_user,
    save_analysis, get_history, get_ticker_trend_data, calculate_cagr,
    add_to_watchlist, remove_from_watchlist, is_on_watchlist, get_watchlist_with_scores,
    add_alert, get_alerts, delete_alert,
    check_and_fire_alerts,
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
    tabs = st.tabs(["Analyze New Report", "5-Year Trends", "Watchlist", "Alerts", "Analysis History"])

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
                # ── Row 1: Company identity ──────────────────────────
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                with c1:
                    company_name = st.text_input("Company Name *",
                                                 value=raw.get('company_name', ''))
                with c2:
                    ticker_default = _clean_ticker(raw.get('ticker'))
                    ticker = st.text_input(
                        "Ticker Symbol *" + ("  🔴 required" if ticker_missing else ""),
                        value=ticker_default,
                        placeholder="e.g. SAMP, AAPL",
                        help="Used as the unique company ID in the database."
                    )
                with c3:
                    fiscal_year = st.text_input("Fiscal Year", value=str(raw.get('fiscal_year', '')))
                with c4:
                    div_index = 0 if raw.get('dividend_paid') == 'Yes' else 1
                    dividend_paid = st.selectbox("Dividend Paid", ["Yes", "No"], index=div_index)

                st.markdown("##### Income & Returns")
                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    revenue = st.number_input("Revenue (Millions)", value=_flt(raw.get('revenue')),
                                              min_value=0.0, format="%.2f")
                with c6:
                    net_income = st.number_input("Net Income (Millions)", value=_flt(raw.get('net_income')),
                                                 format="%.2f")
                with c7:
                    eps = st.number_input("EPS", value=_flt(raw.get('eps')), format="%.4f")
                with c8:
                    roe = st.number_input("ROE (%)", value=_flt(raw.get('roe')), format="%.2f")

                st.markdown("##### Valuation Ratios")
                c9, c10, c11, c12 = st.columns(4)
                with c9:
                    pe_ratio = st.number_input("P/E Ratio", value=_flt(raw.get('pe_ratio')),
                                               min_value=0.0, format="%.2f")
                with c10:
                    pb_ratio = st.number_input("P/B Ratio", value=_flt(raw.get('pb_ratio')),
                                               min_value=0.0, format="%.2f")
                with c11:
                    debt_to_equity = st.number_input("Debt / Equity", value=_flt(raw.get('debt_to_equity')),
                                                     min_value=0.0, format="%.4f")
                with c12:
                    earnings_growth = st.number_input("5yr Earnings Growth (%)",
                                                      value=_flt(raw.get('earnings_growth_5yr')),
                                                      format="%.2f")

                st.markdown("##### Balance Sheet")
                c13, c14, c15 = st.columns(3)
                with c13:
                    current_assets = st.number_input("Current Assets (Millions)",
                                                     value=_flt(raw.get('current_assets')),
                                                     min_value=0.0, format="%.2f")
                with c14:
                    current_liabilities = st.number_input("Current Liabilities (Millions)",
                                                          value=_flt(raw.get('current_liabilities')),
                                                          min_value=0.0, format="%.2f")
                with c15:
                    intrinsic_value_raw = st.number_input("Intrinsic Value (if stated, else 0)",
                                                          value=_flt(raw.get('intrinsic_value')),
                                                          min_value=0.0, format="%.2f")

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
                        'revenue': revenue,
                        'net_income': net_income,
                        'eps': eps,
                        'roe': roe,
                        'debt_to_equity': debt_to_equity,
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'earnings_growth_5yr': earnings_growth,
                        'current_assets': current_assets,
                        'current_liabilities': current_liabilities,
                        'dividend_paid': dividend_paid,
                        'intrinsic_value': intrinsic_value_raw,
                    }
                    score, rec, checklist, mos, intrinsic_v = calculate_graham_score(edited_data)
                    save_analysis(user, company_name, ticker_clean, edited_data, score, rec)
                    st.session_state.last_analysis = {
                        'data': edited_data, 'score': score, 'rec': rec,
                        'checklist': checklist, 'mos': mos, 'intrinsic_v': intrinsic_v,
                        'ticker': ticker_clean, 'company': company_name,
                    }
                    st.session_state.extracted_data = None
                    fired = check_and_fire_alerts(user, ticker_clean, company_name, score, mos)
                    for f in fired:
                        st.success(f"📧 Alert fired: {f}")
                    st.rerun()

        # ── PHASE 3: RESULTS (shown after Confirm & Score) ──────
        if st.session_state.last_analysis:
            la = st.session_state.last_analysis
            data, score, rec, checklist, mos, intrinsic_v = (
                la['data'], la['score'], la['rec'],
                la['checklist'], la['mos'], la['intrinsic_v']
            )
            ticker_val = la['ticker']
            company_val = la['company']

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Graham Score", f"{score}/15")
                rec_class = rec.lower().replace(" ", "-")
                st.markdown(f'<div class="recommendation-card {rec_class}">{rec}</div>',
                            unsafe_allow_html=True)
                if ticker_val:
                    on_wl = is_on_watchlist(user, ticker_val)
                    label = "★ Remove from Watchlist" if on_wl else "☆ Add to Watchlist"
                    if st.button(label):
                        if on_wl:
                            remove_from_watchlist(user, ticker_val)
                        else:
                            add_to_watchlist(user, ticker_val, company_val)
                        st.rerun()

            with col2:
                st.metric("Margin of Safety", f"{mos:.1f}%")
                st.metric("Intrinsic Value (Est.)", f"${intrinsic_v:.2f}")

            with col3:
                st.subheader("Defensive Checklist")
                for item in checklist:
                    st.write(item)

            st.subheader("Saved Financial Data")
            df_display = pd.DataFrame([data]).T.rename(columns={0: "Value"})
            st.dataframe(df_display, use_container_width=True)

    # ── TAB 2: 5-YEAR TRENDS ──────────────────────────────────
    with tabs[1]:
        st.subheader("5-Year Financial Trends")
        history_df = get_history(user)
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
            all_tickers_df = get_history(user)
            ticker_choices = sorted(all_tickers_df['ticker'].unique().tolist()) if not all_tickers_df.empty else []

            col_a, col_b = st.columns(2)
            with col_a:
                if ticker_choices:
                    alert_ticker = st.selectbox("Company (Ticker)", ticker_choices, key="alert_ticker_sel")
                    h = get_history(user, alert_ticker)
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
        st.subheader("Analysis Logs")
        history_df = get_history(user)
        if not history_df.empty:
            st.dataframe(
                history_df[['date', 'company_name', 'ticker', 'score', 'recommendation']],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Your history will appear here once you've analyzed some reports.")

# --- FOOTER ---
st.markdown("---")
st.caption("Graham-Bot is for educational purposes only. Always conduct your own research before investing.")
