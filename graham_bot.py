import streamlit as st
import pandas as pd
import sqlite3
import bcrypt
import os
import json
import smtplib
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
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///graham_bot.db")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL") or SMTP_USER

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

# --- DB PATH ---
def get_db_path():
    if DATABASE_URL.startswith("sqlite:///"):
        return DATABASE_URL.split("sqlite:///")[1]
    return "graham_bot.db"

DB_NAME = get_db_path()

# ============================================================
# SCORING — defined early because watchlist helpers use it
# ============================================================
def calculate_graham_score(data):
    score = 0
    checklist = []

    pe = data.get('pe_ratio', 99)
    if pe < 15:
        score += 3; checklist.append("✅ P/E Ratio < 15 (+3)")
    else:
        checklist.append("❌ P/E Ratio >= 15")

    pb = data.get('pb_ratio', 99)
    if pb < 1.0:
        score += 3; checklist.append("✅ P/B Ratio < 1.0 (+3)")
    elif pb < 1.5:
        score += 2; checklist.append("✅ P/B Ratio < 1.5 (+2)")
    else:
        checklist.append("❌ P/B Ratio >= 1.5")

    growth = data.get('earnings_growth_5yr', 0)
    if growth > 20:
        score += 2; checklist.append("✅ High Growth > 20% (+2)")
    elif growth > 0:
        score += 1; checklist.append("✅ Positive Growth (+1)")
    else:
        checklist.append("❌ Negative/Zero Growth")

    roe = data.get('roe', 0)
    if roe > 15:
        score += 2; checklist.append("✅ High ROE > 15% (+2)")
    elif roe > 10:
        score += 1; checklist.append("✅ Decent ROE > 10% (+1)")
    else:
        checklist.append("❌ Low ROE")

    de = data.get('debt_to_equity', 99)
    if de < 0.5:
        score += 2; checklist.append("✅ Conservative Debt < 0.5 (+2)")
    elif de < 1.0:
        score += 1; checklist.append("✅ Manageable Debt < 1.0 (+1)")
    else:
        checklist.append("❌ High Debt/Equity")

    liabilities = data.get('current_liabilities', 0)
    current_ratio = data.get('current_assets', 0) / liabilities if liabilities > 0 else 0
    if current_ratio > 2.0:
        score += 1; checklist.append("✅ Strong Current Ratio > 2.0 (+1)")

    if data.get('dividend_paid') == 'Yes':
        score += 1; checklist.append("✅ Dividend Payer (+1)")

    if data.get('revenue', 0) > 2000:
        score += 1; checklist.append("✅ Large-Cap Size (+1)")

    if score >= 12: rec = "Strong Buy"
    elif score >= 9: rec = "Buy"
    elif score >= 6: rec = "Hold"
    else: rec = "Sell"

    g = data.get('earnings_growth_5yr', 0)
    v = data.get('eps', 0) * (8.5 + 2 * min(g, 15))
    price = pe * data.get('eps', 1)
    mos = ((v - price) / v * 100) if v > 0 else 0

    return score, rec, checklist, mos, v

# ============================================================
# DATABASE
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, company_name TEXT, ticker TEXT,
                  date TEXT, data_json TEXT, score REAL, recommendation TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, ticker TEXT, company_name TEXT, added_date TEXT,
                  UNIQUE(username, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, ticker TEXT, company_name TEXT,
                  alert_type TEXT, threshold REAL, email TEXT,
                  active INTEGER DEFAULT 1, last_triggered TEXT)''')
    conn.commit()
    conn.close()

def hash_password(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def check_password(pw, hashed):
    return bcrypt.checkpw(pw.encode(), hashed.encode())

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, hash_password(password)))
        conn.commit(); return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone(); conn.close(); return user

def save_analysis(username, company, ticker, data, score, rec):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO analysis (username, company_name, ticker, date, data_json, score, recommendation) VALUES (?,?,?,?,?,?,?)",
              (username, company, ticker, datetime.now().strftime("%Y-%m-%d %H:%M"), json.dumps(data), score, rec))
    conn.commit(); conn.close()

def get_history(username, ticker=None):
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM analysis WHERE username=?"
    params = [username]
    if ticker:
        query += " AND ticker=?"; params.append(ticker)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close(); return df

def get_ticker_trend_data(username, ticker):
    df = get_history(username, ticker)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        try:
            d = json.loads(row['data_json'])
            rows.append({
                'fiscal_year': str(d.get('fiscal_year', row['date'][:4])),
                'revenue': float(d.get('revenue', 0) or 0),
                'net_income': float(d.get('net_income', 0) or 0),
                'eps': float(d.get('eps', 0) or 0),
                'roe': float(d.get('roe', 0) or 0),
                'score': float(row['score']),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values('fiscal_year')
            .drop_duplicates('fiscal_year')
            .reset_index(drop=True))

def calculate_cagr(values):
    clean = [v for v in values if v is not None and v != 0]
    if len(clean) < 2 or clean[0] <= 0:
        return None
    try:
        return ((clean[-1] / clean[0]) ** (1 / (len(clean) - 1)) - 1) * 100
    except Exception:
        return None

# --- Watchlist ---
def add_to_watchlist(username, ticker, company_name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO watchlist (username, ticker, company_name, added_date) VALUES (?,?,?,?)",
                  (username, ticker.upper(), company_name, datetime.now().strftime("%Y-%m-%d")))
        conn.commit(); return c.rowcount > 0
    finally:
        conn.close()

def remove_from_watchlist(username, ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE username=? AND ticker=?", (username, ticker.upper()))
    conn.commit(); conn.close()

def is_on_watchlist(username, ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM watchlist WHERE username=? AND ticker=?", (username, ticker.upper()))
    result = c.fetchone() is not None; conn.close(); return result

def get_watchlist_with_scores(username):
    conn = sqlite3.connect(DB_NAME)
    wl = pd.read_sql_query(
        "SELECT ticker, company_name, added_date FROM watchlist WHERE username=?",
        conn, params=(username,))
    items = []
    for _, row in wl.iterrows():
        ticker = row['ticker']
        analyses = pd.read_sql_query(
            "SELECT score, recommendation, date, data_json FROM analysis "
            "WHERE username=? AND ticker=? ORDER BY date DESC LIMIT 2",
            conn, params=(username, ticker))
        entry = {
            'ticker': ticker, 'company_name': row['company_name'],
            'added_date': row['added_date'], 'last_score': None,
            'last_rec': None, 'last_mos': None,
            'last_analyzed': None, 'score_change': None,
        }
        if not analyses.empty:
            latest = analyses.iloc[0]
            entry['last_score'] = int(latest['score'])
            entry['last_rec'] = latest['recommendation']
            entry['last_analyzed'] = latest['date']
            try:
                d = json.loads(latest['data_json'])
                _, _, _, mos, _ = calculate_graham_score(d)
                entry['last_mos'] = round(mos, 1)
            except Exception:
                pass
            if len(analyses) >= 2:
                entry['score_change'] = int(latest['score']) - int(analyses.iloc[1]['score'])
        items.append(entry)
    conn.close()
    return items

# --- Alerts ---
def add_alert(username, ticker, company_name, alert_type, threshold, email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (username, ticker, company_name, alert_type, threshold, email, active) VALUES (?,?,?,?,?,?,1)",
              (username, ticker.upper(), company_name, alert_type, threshold, email))
    conn.commit(); conn.close()

def get_alerts(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM alerts WHERE username=? ORDER BY id DESC", conn, params=(username,))
    conn.close(); return df

def delete_alert(alert_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    conn.commit(); conn.close()

def check_and_fire_alerts(username, ticker, company_name, current_score, current_mos):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, alert_type, threshold, email FROM alerts WHERE username=? AND ticker=? AND active=1",
              (username, ticker.upper()))
    alerts = c.fetchall(); conn.close()

    fired = []
    for alert_id, alert_type, threshold, email in alerts:
        triggered = False
        if alert_type == "score_above" and current_score >= threshold:
            triggered = True
            subject = f"Graham-Bot: {ticker} Score {current_score}/15 — At or Above {int(threshold)}"
            body = f"{company_name} ({ticker}) Graham Score is {current_score}/15, at or above your threshold of {int(threshold)}."
        elif alert_type == "score_below" and current_score < threshold:
            triggered = True
            subject = f"Graham-Bot: {ticker} Score {current_score}/15 — Below {int(threshold)}"
            body = f"{company_name} ({ticker}) Graham Score dropped to {current_score}/15, below your threshold of {int(threshold)}."
        elif alert_type == "mos_above" and current_mos >= threshold:
            triggered = True
            subject = f"Graham-Bot: {ticker} MOS {current_mos:.1f}% — At or Above {threshold}%"
            body = f"{company_name} ({ticker}) Margin of Safety is {current_mos:.1f}%, at or above your threshold of {threshold}%."
        elif alert_type == "mos_below" and current_mos < threshold:
            triggered = True
            subject = f"Graham-Bot: {ticker} MOS {current_mos:.1f}% — Below {threshold}%"
            body = f"{company_name} ({ticker}) Margin of Safety dropped to {current_mos:.1f}%, below your threshold of {threshold}%."

        if triggered:
            ok, _ = send_alert_email(email, subject, body + "\n\n---\nGraham-Bot | For educational purposes only.")
            if ok:
                conn2 = sqlite3.connect(DB_NAME)
                conn2.execute("UPDATE alerts SET last_triggered=? WHERE id=?",
                              (datetime.now().strftime("%Y-%m-%d %H:%M"), alert_id))
                conn2.commit(); conn2.close()
                fired.append(f"{alert_type} → {email}")
    return fired

# --- Email ---
def send_alert_email(to_email, subject, body):
    if not SMTP_USER or not SMTP_PASSWORD:
        return False, "SMTP not configured"
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo(); server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        return True, "Sent"
    except Exception as e:
        return False, str(e)

def send_test_email(to_email):
    return send_alert_email(
        to_email,
        "Graham-Bot — Test Email",
        "Your Graham-Bot email alerts are configured correctly.\n\n---\nGraham-Bot | For educational purposes only."
    )

# ============================================================
# AI DATA EXTRACTION
# ============================================================
def detect_page_offset(reader, max_scan=20):
    for i in range(min(max_scan, len(reader.pages))):
        text = reader.pages[i].extract_text() or ""
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        candidates = (lines[:3] + lines[-3:]) if len(lines) > 3 else lines
        for line in candidates:
            if line.isdigit():
                n = int(line)
                if 1 <= n <= 8:
                    return i - (n - 1)
    return 0

def extract_financial_data(uploaded_file):
    if not API_KEY:
        st.error("Missing Gemini API Key. Please set GEMINI_API_KEY in .env")
        return None
    try:
        reader = PdfReader(uploaded_file)
        total_pages = len(reader.pages)
        model = genai.GenerativeModel("gemini-flash-latest")

        st.info(f"📄 Report has {total_pages} pages. Reading table of contents...")
        toc_text = ""
        for i in range(min(10, total_pages)):
            toc_text += f"\n--- PDF Page {i + 1} ---\n{reader.pages[i].extract_text() or ''}\n"

        toc_prompt = f"""Analyze this text from the first pages of an annual financial report.
Find the Table of Contents and identify the printed page numbers for financial statement sections.

Look for: Consolidated Balance Sheet, Income Statement / Statement of Operations,
Cash Flow Statement, Notes to Financial Statements, Financial Highlights / Five-Year Summary.

Return ONLY valid JSON (no markdown):
{{"toc_found": true, "sections": [{{"name": "section name", "printed_page": 85}}]}}

If no TOC found: {{"toc_found": false, "sections": []}}

Document text:
{toc_text[:12000]}"""

        toc_resp = model.generate_content(toc_prompt)
        toc_raw = toc_resp.text.strip()
        if "```json" in toc_raw:
            toc_raw = toc_raw.split("```json")[1].split("```")[0].strip()
        elif "```" in toc_raw:
            toc_raw = toc_raw.split("```")[1].split("```")[0].strip()
        toc_data = json.loads(toc_raw)

        relevant_pdf_indices = set()
        if toc_data.get("toc_found") and toc_data.get("sections"):
            offset = detect_page_offset(reader)
            sections = toc_data["sections"]
            st.success(f"✅ Table of Contents found — {len(sections)} financial sections identified.")
            for sec in sections:
                printed = sec.get("printed_page", 0)
                if printed > 0:
                    pdf_idx = (printed - 1) + offset
                    for j in range(pdf_idx, min(pdf_idx + 3, total_pages)):
                        if 0 <= j < total_pages:
                            relevant_pdf_indices.add(j)
        else:
            st.warning("⚠️ No Table of Contents detected — falling back to keyword scan.")
            keywords = [
                "consolidated balance sheet", "statement of financial position",
                "consolidated statement of income", "statement of operations",
                "consolidated statement of cash flows", "financial highlights",
                "five year summary", "five-year summary"
            ]
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").lower()
                if any(k in text for k in keywords):
                    relevant_pdf_indices.add(i)
                    if i + 1 < total_pages:
                        relevant_pdf_indices.add(i + 1)

        if not relevant_pdf_indices:
            st.warning("⚠️ Could not identify financial pages — using first 15 pages.")
            relevant_pdf_indices = set(range(min(15, total_pages)))

        relevant_pages = sorted(list(relevant_pdf_indices))
        st.info(f"🔍 Sending {len(relevant_pages)} targeted pages (out of {total_pages}) to Gemini.")

        writer = PdfWriter()
        for p_idx in relevant_pages:
            writer.add_page(reader.pages[p_idx])
        cropped_pdf_path = "temp_report_cropped.pdf"
        with open(cropped_pdf_path, "wb") as f:
            writer.write(f)

        myfile = genai.upload_file(cropped_pdf_path)
        data_prompt = """Analyze this financial report excerpt and extract the metrics below.
Use 0 for any value not found. Return ONLY valid JSON, no markdown, no extra text.

{
  "company_name": "string",
  "ticker": "string",
  "fiscal_year": "YYYY",
  "revenue": number_in_millions,
  "net_income": number_in_millions,
  "eps": number,
  "roe": percentage_as_number,
  "debt_to_equity": ratio_number,
  "pe_ratio": number,
  "pb_ratio": number,
  "earnings_growth_5yr": percentage_as_number,
  "current_assets": number_in_millions,
  "current_liabilities": number_in_millions,
  "dividend_paid": "Yes or No",
  "intrinsic_value": number_or_0
}"""
        response = model.generate_content([data_prompt, myfile])
        raw_text = response.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        return json.loads(raw_text)

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")
        return None
    finally:
        if os.path.exists("temp_report_cropped.pdf"):
            os.remove("temp_report_cropped.pdf")

# ============================================================
# SESSION STATE & DB INIT
# ============================================================
for key, default in [('logged_in', False), ('user', None), ('last_analysis', None)]:
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
            if st.button("Run Analysis"):
                with st.spinner("Reading table of contents and extracting financial data..."):
                    data = extract_financial_data(uploaded_file)
                    if data:
                        score, rec, checklist, mos, intrinsic_v = calculate_graham_score(data)
                        ticker_val = str(data.get('ticker', '') or '').strip()
                        if ticker_val in ('0', 'N/A', 'NA', 'n/a'):
                            ticker_val = ''
                        company_val = data.get('company_name', '')
                        save_analysis(user, company_val, ticker_val, data, score, rec)
                        st.session_state.last_analysis = {
                            'data': data, 'score': score, 'rec': rec,
                            'checklist': checklist, 'mos': mos, 'intrinsic_v': intrinsic_v,
                            'ticker': ticker_val, 'company': company_val,
                        }
                        if ticker_val:
                            fired = check_and_fire_alerts(user, ticker_val, company_val, score, mos)
                            for f in fired:
                                st.success(f"📧 Alert fired: {f}")

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

            st.subheader("Extracted Financial Data")
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
