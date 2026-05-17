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

# Shared extraction prompt used by both Gemini and OpenRouter
_DATA_PROMPT = """Analyze this financial report and extract the metrics below.
Search the ENTIRE document — check financial highlights, per share data, investor information,
and balance sheets, not just the income statement.
If a value is not stated directly, DERIVE it using the formula in the description.
Only use 0 if the value genuinely cannot be found or calculated.
Return ONLY valid JSON, no markdown, no extra text.

{
  "company_name": "Full legal company name from cover or header",
  "ticker": "Stock ticker/symbol. Check: cover page, investor info, stock exchange listing, share data table. For Sri Lankan companies check CSE listing.",
  "fiscal_year": "Financial year end year as YYYY",
  "revenue": "For normal companies: total revenue/turnover in millions. For banks: Net Interest Income + Non-Interest Income (total operating income) in millions.",
  "net_income": "Profit after tax / Net profit for the year in millions.",
  "eps": "Earnings Per Share — find in Per Share Data table, Financial Highlights, or Five/Ten-Year Summary. Also labelled Basic EPS or Diluted EPS.",
  "roe": "Return on Equity %. Find in Financial Ratios, KPIs, or Financial Highlights. If not stated, calculate as (Net Income / Average Shareholders Equity) x 100.",
  "debt_to_equity": "Total Liabilities / Total Equity from the balance sheet. For banks this is typically 8-15. Calculate from balance sheet if not stated.",
  "pe_ratio": "Price to Earnings ratio. Find in Investor Information, Share Data, Capital Market Information, or Financial Highlights. Use 0 only if completely absent.",
  "pb_ratio": "Price to Book Value ratio. Find in Investor Information, Share Data, or Financial Highlights. Also labelled Market Price to Book Value or P/BV. Use 0 only if completely absent.",
  "earnings_growth_5yr": "5-year earnings growth %. Find in Five/Ten-Year financial summary. Calculate as ((Latest EPS / EPS 5 years ago)^(1/5) - 1) x 100. Use 0 if only 1 year available.",
  "current_assets": "For normal companies: current assets in millions. For banks: total assets due within 1 year, or total assets if not broken down by maturity.",
  "current_liabilities": "For normal companies: current liabilities in millions. For banks: total liabilities due within 1 year, or total deposits + short-term borrowings.",
  "dividend_paid": "Yes if any dividend was declared or paid this financial year, No otherwise.",
  "intrinsic_value": "Stated intrinsic or fair value per share if mentioned, otherwise 0."
}"""

_TOC_KEYWORDS = [
    "statement of financial position",
    "statement of profit or loss and other comprehensive income",
    "statement of profit or loss",
    "statement of changes in equity",
    "statement of cash flows",
    "income statement",
    "consolidated balance sheet",
    "consolidated statement of income",
    "consolidated statement of operations",
    "consolidated statement of cash flows",
    "consolidated statement of changes in equity",
    "financial highlights",
    "five year summary", "five-year summary",
    "ten year summary", "ten-year summary",
    "per share data", "share information",
    "investor information", "shareholders information",
    "capital market data", "financial ratios",
    "key financial indicators", "key performance indicators",
]


_TOC_HEADING_RE = re.compile(
    r'\b(table\s+of\s+contents?|contents?|index)\b',
    re.IGNORECASE,
)

def _find_toc_page_indices(reader, max_scan=30):
    """
    Scan up to max_scan pages to find which ones carry a TOC heading.
    Matches headings like: TABLE OF CONTENTS, CONTENTS, CONTENT, INDEX.
    The matched line must be short (≤ 40 chars) so it's a title, not a sentence.
    Returns a list of 0-based page indices (the heading page + up to 2 following
    pages to capture multi-page TOCs).  Falls back to the first 10 pages if
    no heading is found.
    """
    total = len(reader.pages)
    found = []
    for i in range(min(max_scan, total)):
        text = reader.pages[i].extract_text() or ""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[:20]:
            if len(line) <= 40 and _TOC_HEADING_RE.search(line):
                for j in range(i, min(i + 3, total)):
                    if j not in found:
                        found.append(j)
                break
    return sorted(found) if found else list(range(min(10, total)))


def parse_toc_locally(toc_text):
    financial_keywords = [
        "income statement", "statement of profit or loss",
        "statement of financial position", "balance sheet",
        "statement of cash flows", "cash flow",
        "statement of changes in equity", "changes in equity",
        "comprehensive income", "financial highlights",
        "five year", "five-year", "ten year", "ten-year",
        "per share data", "investor information",
        "shareholders information", "financial ratios",
        "key financial indicators", "key performance indicators",
        "capital market", "segmental", "segment result",
    ]
    sections = []
    for line in toc_text.split('\n'):
        clean = line.strip()
        if len(clean) < 5:
            continue
        m = (re.search(r'^(.+?)\s*\.{2,}\s*(\d{1,4})\s*$', clean) or
             re.search(r'^(.+?)\s{4,}(\d{1,4})\s*$', clean) or
             re.search(r'^(.+?)\s*[-–—|]\s*(\d{1,4})\s*$', clean))
        if not m:
            continue
        name = m.group(1).strip().rstrip('.')
        page = int(m.group(2))
        if 1 <= page <= 2000 and any(kw in name.lower() for kw in financial_keywords):
            if not any(s['printed_page'] == page for s in sections):
                sections.append({'name': name, 'printed_page': page})
    return sections if len(sections) >= 2 else []


def _find_relevant_pages(reader, allow_gemini_toc=True):
    """
    Shared page-detection logic for both Gemini and OpenRouter.
    Returns (page_map, sorted list of 0-based PDF indices).
    allow_gemini_toc=False skips the Gemini TOC API call (used in OpenRouter mode).
    """
    total_pages = len(reader.pages)
    page_map = {'method': None, 'total_pages': total_pages, 'sections': [], 'pages_sent': []}

    st.info(f"📄 Report has {total_pages} pages. Locating contents page...")
    toc_indices = _find_toc_page_indices(reader)
    toc_label = f"PDF page(s) {[i + 1 for i in toc_indices]}" if len(toc_indices) <= 5 else f"first {len(toc_indices)} pages"
    st.info(f"📑 Contents section found at {toc_label}. Reading entries...")

    toc_text = ""
    for i in toc_indices:
        toc_text += f"\n--- PDF Page {i + 1} ---\n{reader.pages[i].extract_text() or ''}\n"

    local_sections = parse_toc_locally(toc_text)
    if local_sections:
        st.info(f"📋 Contents parsed locally — {len(local_sections)} financial sections found (no API call used).")
        toc_data = {"toc_found": True, "sections": local_sections}
    elif allow_gemini_toc:
        st.info("📋 Local parse inconclusive — using Gemini to read table of contents...")
        model = genai.GenerativeModel("gemini-2.0-flash")
        toc_prompt = f"""Analyze this text extracted from an annual financial report's contents page.
The contents page may be titled: TABLE OF CONTENTS, CONTENTS, CONTENT, or INDEX.
Identify the printed page numbers for ALL financial statement sections listed in it.

Look for ANY of these sections (use the exact name from the document, not these labels):
- Income Statement / Statement of Profit or Loss / Statement of Profit or Loss and Other Comprehensive Income
- Statement of Financial Position / Balance Sheet / Consolidated Balance Sheet
- Statement of Changes in Equity (including Group and Bank variants)
- Statement of Cash Flows / Consolidated Statement of Cash Flows
- Notes to Financial Statements / Accounting Policies / Significant Accounting Policies
- Financial Highlights / Five-Year Summary / Ten-Year Summary / Key Financial Indicators
- Per Share Data / Share Information / Investor Information / Shareholders Information / Capital Market Data
- Financial Ratios / Key Performance Indicators / KPIs / Segmental Information

Return ONLY valid JSON (no markdown):
{{"toc_found": true, "sections": [{{"name": "exact section name from document", "printed_page": 85}}]}}
If no contents page is present: {{"toc_found": false, "sections": []}}

Document text:
{toc_text[:12000]}"""
        toc_resp = model.generate_content(toc_prompt)
        toc_raw = toc_resp.text.strip()
        if "```json" in toc_raw:
            toc_raw = toc_raw.split("```json")[1].split("```")[0].strip()
        elif "```" in toc_raw:
            toc_raw = toc_raw.split("```")[1].split("```")[0].strip()
        toc_data = json.loads(toc_raw)
    else:
        toc_data = {"toc_found": False, "sections": []}

    relevant_pdf_indices = set()

    if toc_data.get("toc_found") and toc_data.get("sections"):
        offset = detect_page_offset(reader)
        page_map['method'] = 'toc'
        st.success(f"✅ Table of Contents — {len(toc_data['sections'])} financial sections identified.")
        for sec in toc_data["sections"]:
            printed = sec.get("printed_page", 0)
            if printed > 0:
                pdf_idx = (printed - 1) + offset
                pages_for_sec = []
                for j in range(pdf_idx, min(pdf_idx + 3, total_pages)):
                    if 0 <= j < total_pages:
                        relevant_pdf_indices.add(j)
                        pages_for_sec.append(j + 1)
                page_map['sections'].append({
                    'name': sec.get('name', ''), 'toc_page': printed, 'pdf_pages': pages_for_sec,
                })
    else:
        page_map['method'] = 'keyword_scan'
        st.warning("⚠️ No TOC detected — scanning all pages for financial statement keywords.")
        page_keyword: dict = {}
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").lower()
            for kw in _TOC_KEYWORDS:
                if kw in text and i not in page_keyword:
                    page_keyword[i] = kw
                    relevant_pdf_indices.add(i)
                    if i + 1 < total_pages:
                        relevant_pdf_indices.add(i + 1)
                    break
        for pg_idx, kw in sorted(page_keyword.items()):
            sent = [pg_idx + 1] + ([pg_idx + 2] if pg_idx + 1 < total_pages else [])
            page_map['sections'].append({'name': kw.title(), 'toc_page': None, 'pdf_pages': sent})

    if not relevant_pdf_indices:
        page_map['method'] = 'fallback'
        st.warning("⚠️ Could not identify financial pages — using first 15 pages.")
        relevant_pdf_indices = set(range(min(15, total_pages)))
        page_map['sections'] = [{'name': 'Fallback — first 15 pages', 'toc_page': None,
                                  'pdf_pages': list(range(1, min(16, total_pages + 1)))}]

    relevant_pages = sorted(list(relevant_pdf_indices))
    page_map['pages_sent'] = [p + 1 for p in relevant_pages]
    return page_map, relevant_pages


def extract_financial_data(uploaded_file):
    """Gemini path — uploads a cropped PDF for native PDF understanding."""
    if not API_KEY:
        st.error("Missing Gemini API Key. Please set GEMINI_API_KEY in .env")
        return None, None

    try:
        reader = PdfReader(uploaded_file)
        page_map, relevant_pages = _find_relevant_pages(reader, allow_gemini_toc=True)
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
        raw_text = response.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        return json.loads(raw_text), page_map

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")
        return None, None
    finally:
        if os.path.exists("temp_report_cropped.pdf"):
            os.remove("temp_report_cropped.pdf")

def extract_financial_data_openrouter(uploaded_file, model_id="openai/gpt-oss-120b:free"):
    """OpenRouter path — extracts PDF text locally, sends as a text prompt."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        st.error("Missing OpenRouter API Key. Please set OPENROUTER_API_KEY in .env")
        return None, None

    try:
        reader = PdfReader(uploaded_file)
        page_map, relevant_pages = _find_relevant_pages(reader, allow_gemini_toc=False)
        total_pages = page_map['total_pages']

        st.info(f"🔍 Extracting text from {len(relevant_pages)} of {total_pages} targeted pages...")

        extracted_text = ""
        for p_idx in relevant_pages:
            extracted_text += f"\n--- Page {p_idx + 1} ---\n{reader.pages[p_idx].extract_text() or ''}\n"

        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": _DATA_PROMPT + "\n\nDocument text:\n" + extracted_text[:60000]},
            ],
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"].strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        return json.loads(raw_text), page_map

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")
        return None, None


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
