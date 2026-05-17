import streamlit as st
import pandas as pd
import sqlite3
import bcrypt
import os
import json
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import time
from pypdf import PdfReader, PdfWriter
import io

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///graham_bot.db")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="Graham-Bot | Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
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
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .buy { background-color: #28a745; }
    .strong-buy { background-color: #1e7e34; border: 3px solid #ffd700; }
    .hold { background-color: #ffc107; color: black; }
    .sell { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- DATABASE SETUP ---
def get_db_path():
    # Extract filename from sqlite:///path/to/db
    if DATABASE_URL.startswith("sqlite:///"):
        return DATABASE_URL.split("sqlite:///")[1]
    return "graham_bot.db"

DB_NAME = get_db_path()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  company_name TEXT, 
                  ticker TEXT, 
                  date TEXT, 
                  data_json TEXT, 
                  score REAL, 
                  recommendation TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def save_analysis(username, company, ticker, data, score, rec):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO analysis (username, company_name, ticker, date, data_json, score, recommendation) VALUES (?,?,?,?,?,?,?)",
              (username, company, ticker, datetime.now().strftime("%Y-%m-%d %H:%M"), json.dumps(data), score, rec))
    conn.commit()
    conn.close()

def get_history(username, ticker=None):
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM analysis WHERE username=?"
    params = [username]
    if ticker:
        query += " AND ticker=?"
        params.append(ticker)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# --- AI DATA EXTRACTION ---
def detect_page_offset(reader, max_scan=20):
    """
    Detect the offset between a report's printed page numbers and PDF 0-based indices.
    Scans page headers/footers for small standalone integers (e.g. "1", "2") that
    indicate where the document's numbered pages begin.
    Returns offset such that: pdf_0based_idx = (printed_page - 1) + offset
    """
    for i in range(min(max_scan, len(reader.pages))):
        text = reader.pages[i].extract_text() or ""
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        # Check header and footer lines only
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

        # --- STEP 1: Read first 10 pages locally to capture the Table of Contents ---
        st.info(f"📄 Report has {total_pages} pages. Reading table of contents...")
        toc_text = ""
        for i in range(min(10, total_pages)):
            page_text = reader.pages[i].extract_text() or ""
            toc_text += f"\n--- PDF Page {i + 1} ---\n{page_text}\n"

        # --- STEP 2: Cheap text-only Gemini call to parse the TOC ---
        toc_prompt = f"""Analyze this text extracted from the first pages of an annual financial report.
Find the Table of Contents and identify the printed page numbers for financial statement sections.

Look for sections such as:
- Consolidated Balance Sheet / Statement of Financial Position
- Consolidated Income Statement / Statement of Operations / Profit & Loss
- Consolidated Statement of Cash Flows
- Notes to Financial Statements / Significant Accounting Policies
- Financial Highlights / Five-Year Financial Summary

Return ONLY valid JSON (no markdown, no explanation):
{{
  "toc_found": true,
  "sections": [
    {{"name": "section name here", "printed_page": 85}}
  ]
}}

If no table of contents is found return: {{"toc_found": false, "sections": []}}

Document text:
{toc_text[:12000]}"""

        toc_resp = model.generate_content(toc_prompt)
        toc_raw = toc_resp.text.strip()
        if "```json" in toc_raw:
            toc_raw = toc_raw.split("```json")[1].split("```")[0].strip()
        elif "```" in toc_raw:
            toc_raw = toc_raw.split("```")[1].split("```")[0].strip()
        toc_data = json.loads(toc_raw)

        # --- STEP 3: Map printed page numbers to PDF 0-based indices ---
        relevant_pdf_indices = set()

        if toc_data.get("toc_found") and toc_data.get("sections"):
            offset = detect_page_offset(reader)
            sections = toc_data["sections"]
            st.success(f"✅ Table of Contents found — {len(sections)} financial sections identified.")

            for sec in sections:
                printed = sec.get("printed_page", 0)
                if printed > 0:
                    pdf_idx = (printed - 1) + offset
                    # Grab the section page plus 2 follow-on pages (tables often span pages)
                    for j in range(pdf_idx, min(pdf_idx + 3, total_pages)):
                        if 0 <= j < total_pages:
                            relevant_pdf_indices.add(j)
        else:
            # Fallback: keyword scan across the full document
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
            st.warning("⚠️ Could not identify financial pages — using first 15 pages as a fallback.")
            relevant_pdf_indices = set(range(min(15, total_pages)))

        relevant_pages = sorted(list(relevant_pdf_indices))
        st.info(f"🔍 Sending {len(relevant_pages)} targeted pages (out of {total_pages}) to Gemini for analysis.")

        # --- STEP 4: Crop PDF to the relevant pages only ---
        writer = PdfWriter()
        for p_idx in relevant_pages:
            writer.add_page(reader.pages[p_idx])

        cropped_pdf_path = "temp_report_cropped.pdf"
        with open(cropped_pdf_path, "wb") as f:
            writer.write(f)

        # --- STEP 5: Upload cropped PDF and extract financial data ---
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

        data = json.loads(raw_text)
        return data

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")
        return None
    finally:
        if os.path.exists("temp_report_cropped.pdf"):
            os.remove("temp_report_cropped.pdf")

# --- SCORING LOGIC ---
def calculate_graham_score(data):
    score = 0
    checklist = []
    
    # 1. P/E Ratio (<15 = +3 points)
    pe = data.get('pe_ratio', 99)
    if pe < 15:
        score += 3
        checklist.append("✅ P/E Ratio < 15 (+3)")
    else:
        checklist.append("❌ P/E Ratio >= 15")

    # 2. P/B Ratio (<1.0 = +3, <1.5 = +2)
    pb = data.get('pb_ratio', 99)
    if pb < 1.0:
        score += 3
        checklist.append("✅ P/B Ratio < 1.0 (+3)")
    elif pb < 1.5:
        score += 2
        checklist.append("✅ P/B Ratio < 1.5 (+2)")
    else:
        checklist.append("❌ P/B Ratio >= 1.5")

    # 3. Earnings growth (>20% = +2, >0% = +1)
    growth = data.get('earnings_growth_5yr', 0)
    if growth > 20:
        score += 2
        checklist.append("✅ High Growth > 20% (+2)")
    elif growth > 0:
        score += 1
        checklist.append("✅ Positive Growth (+1)")
    else:
        checklist.append("❌ Negative/Zero Growth")

    # 4. ROE (>15% = +2, >10% = +1)
    roe = data.get('roe', 0)
    if roe > 15:
        score += 2
        checklist.append("✅ High ROE > 15% (+2)")
    elif roe > 10:
        score += 1
        checklist.append("✅ Decent ROE > 10% (+1)")
    else:
        checklist.append("❌ Low ROE")

    # 5. Debt/Equity (<0.5 = +2, <1.0 = +1)
    de = data.get('debt_to_equity', 99)
    if de < 0.5:
        score += 2
        checklist.append("✅ Conservative Debt < 0.5 (+2)")
    elif de < 1.0:
        score += 1
        checklist.append("✅ Manageable Debt < 1.0 (+1)")
    else:
        checklist.append("❌ High Debt/Equity")

    # 6. Graham's Defensive Extras (Size & Liquidity - Up to 3 points)
    liabilities = data.get('current_liabilities', 0)
    current_ratio = data.get('current_assets', 0) / liabilities if liabilities > 0 else 0
    if current_ratio > 2.0:
        score += 1
        checklist.append("✅ Strong Current Ratio > 2.0 (+1)")
    
    if data.get('dividend_paid') == 'Yes':
        score += 1
        checklist.append("✅ Dividend Payer (+1)")
        
    if data.get('revenue', 0) > 2000: # Over $2B
        score += 1
        checklist.append("✅ Large-Cap Size (+1)")

    # Recommendation
    if score >= 12: rec = "Strong Buy"
    elif score >= 9: rec = "Buy"
    elif score >= 6: rec = "Hold"
    else: rec = "Sell"
    
    # Margin of Safety (Simplified)
    # Graham Formula: V = EPS * (8.5 + 2g)
    g = data.get('earnings_growth_5yr', 0)
    v = data.get('eps', 0) * (8.5 + 2 * min(g, 15)) # Cap growth at 15 for formula safety
    price = pe * data.get('eps', 1)
    mos = ((v - price) / v * 100) if v > 0 else 0
    
    return score, rec, checklist, mos, v

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

init_db()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Graham-Bot")
    st.markdown("---")
    
    if not st.session_state.logged_in:
        auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if auth_mode == "Login":
            if st.button("Login"):
                user = get_user(username)
                if user and check_password(password, user[1]):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            if st.button("Sign Up"):
                if add_user(username, password):
                    st.success("User created! Please login.")
                else:
                    st.error("Username already exists")
    else:
        st.success(f"Welcome, {st.session_state.user}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()
            
    st.markdown("---")
    st.info("Built on Benjamin Graham's 'The Intelligent Investor' principles.")

# --- MAIN APP ---
if not st.session_state.logged_in:
    st.title("🚀 Graham-Bot Financial Analysis")
    st.markdown("""
    Welcome to **Graham-Bot**, your AI-powered value investing assistant.
    
    ### How it works:
    1. **Upload** a company's annual or quarterly PDF report.
    2. **AI Extraction**: Gemini analyzes the report for key financial metrics.
    3. **Scoring**: Benjamin Graham's rules are applied to generate a 0-15 score.
    4. **Recommendation**: Get a clear Buy/Hold/Sell signal with Margin of Safety.
    
    **Please login or sign up to continue.**
    """)
    st.image("https://images.unsplash.com/photo-1611974717483-9b04c8612a45?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80")
else:
    st.title("📈 Market Analysis Dashboard")
    
    tabs = st.tabs(["Analyze New Report", "Historical Trends", "My Analysis History"])
    
    with tabs[0]:
        st.subheader("Upload PDF Financial Report")
        uploaded_file = st.file_uploader("Drop PDF here", type="pdf")

        if uploaded_file:
            if st.button("Run Analysis"):
                with st.spinner("Reading table of contents and extracting financial data..."):
                    data = extract_financial_data(uploaded_file)
                    
                    if data:
                        score, rec, checklist, mos, intrinsic_v = calculate_graham_score(data)
                        
                        # Save to DB
                        save_analysis(st.session_state.user, data['company_name'], data['ticker'], data, score, rec)
                        
                        # Display Results
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.metric("Graham Score", f"{score}/15")
                            rec_class = rec.lower().replace(" ", "-")
                            st.markdown(f'<div class="recommendation-card {rec_class}">{rec}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Margin of Safety", f"{mos:.1f}%")
                            st.metric("Intrinsic Value (Est)", f"${intrinsic_v:.2f}")
                            
                        with col3:
                            st.subheader("Defensive Checklist")
                            for item in checklist:
                                st.write(item)
                                
                        st.subheader("Extracted Financial Data")
                        df_data = pd.DataFrame([data]).T.rename(columns={0: "Value"})
                        st.dataframe(df_data, use_container_width=True)
                        
                        st.balloons()

    with tabs[1]:
        st.subheader("Company Performance Trends")
        history_df = get_history(st.session_state.user)
        if not history_df.empty:
            tickers = history_df['ticker'].unique()
            selected_ticker = st.selectbox("Select Ticker to View Trends", tickers)
            
            ticker_data = history_df[history_df['ticker'] == selected_ticker].sort_values('date')
            
            if len(ticker_data) > 1:
                # Plot Score Trend
                fig = px.line(ticker_data, x='date', y='score', title=f"Graham Score Trend for {selected_ticker}",
                             markers=True, line_shape='spline')
                st.plotly_chart(fig, use_container_width=True)
                
                # Extract specific metrics from JSON for more charts
                # This would require more processing of the data_json column
            else:
                st.info("Not enough historical data points for this ticker yet. Upload more reports!")
        else:
            st.info("No analysis history found.")

    with tabs[2]:
        st.subheader("Analysis Logs")
        history_df = get_history(st.session_state.user)
        if not history_df.empty:
            st.dataframe(history_df[['date', 'company_name', 'ticker', 'score', 'recommendation']], use_container_width=True)
        else:
            st.info("Your history will appear here once you've analyzed some reports.")

# --- FOOTER ---
st.markdown("---")
st.caption("Graham-Bot is for educational purposes only. Always conduct your own research before investing.")
