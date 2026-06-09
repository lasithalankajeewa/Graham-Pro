"""
Dual-backend database module.
Reads DATABASE_URL from the environment:
  sqlite:///path/to/file.db   → SQLite  (local dev, default)
  postgresql://...            → PostgreSQL via psycopg2 (Supabase / production)
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import bcrypt
import pandas as pd

from core.email_utils import send_alert_email

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///graham_bot.db")

# Detect backend once at import time
if DATABASE_URL.startswith("sqlite:///"):
    _BACKEND = "sqlite"
    _DB_PATH = DATABASE_URL.split("sqlite:///")[1] or "graham_bot.db"
else:
    _BACKEND = "postgres"
    _DB_PATH = None  # unused for postgres


@contextmanager
def _conn():
    """Yield (connection, placeholder_char) and close on exit."""
    if _BACKEND == "sqlite":
        con = sqlite3.connect(_DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            yield con, "?"
            con.commit()
        finally:
            con.close()
    else:
        import psycopg2
        import psycopg2.extras
        con = psycopg2.connect(DATABASE_URL)
        try:
            yield con, "%s"
            con.commit()
        finally:
            con.close()


def _sql(query: str, ph: str) -> str:
    """Rewrite ? placeholders to %s for PostgreSQL."""
    if ph == "%s":
        return query.replace("?", "%s")
    return query


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db():
    with _conn() as (con, ph):
        cur = con.cursor()

        cur.execute("""CREATE TABLE IF NOT EXISTS users
                       (username TEXT PRIMARY KEY, password TEXT)""")

        cur.execute("""CREATE TABLE IF NOT EXISTS analysis
                       (id INTEGER PRIMARY KEY """ + ("AUTOINCREMENT" if ph == "?" else "GENERATED ALWAYS AS IDENTITY") + """,
                        username TEXT, company_name TEXT, ticker TEXT,
                        date TEXT, data_json TEXT, score REAL,
                        recommendation TEXT, source TEXT DEFAULT 'manual')""")

        cur.execute("""CREATE TABLE IF NOT EXISTS watchlist
                       (id INTEGER PRIMARY KEY """ + ("AUTOINCREMENT" if ph == "?" else "GENERATED ALWAYS AS IDENTITY") + """,
                        username TEXT, ticker TEXT, company_name TEXT, added_date TEXT,
                        UNIQUE(username, ticker))""")

        cur.execute("""CREATE TABLE IF NOT EXISTS alerts
                       (id INTEGER PRIMARY KEY """ + ("AUTOINCREMENT" if ph == "?" else "GENERATED ALWAYS AS IDENTITY") + """,
                        username TEXT, ticker TEXT, company_name TEXT,
                        alert_type TEXT, threshold REAL, email TEXT,
                        active INTEGER DEFAULT 1, last_triggered TEXT)""")

        # processed_reports — deduplication table for the pipeline
        if ph == "?":
            cur.execute("""CREATE TABLE IF NOT EXISTS processed_reports
                           (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ticker TEXT NOT NULL,
                            company_name TEXT NOT NULL,
                            report_type TEXT,
                            fiscal_year TEXT,
                            pdf_url TEXT NOT NULL UNIQUE,
                            processed_at TEXT,
                            analysis_id INTEGER)""")
        else:
            cur.execute("""CREATE TABLE IF NOT EXISTS processed_reports
                           (id SERIAL PRIMARY KEY,
                            ticker TEXT NOT NULL,
                            company_name TEXT NOT NULL,
                            report_type TEXT,
                            fiscal_year TEXT,
                            pdf_url TEXT NOT NULL UNIQUE,
                            processed_at TIMESTAMPTZ DEFAULT NOW(),
                            analysis_id INTEGER)""")

        # Migrate existing analysis table to add source column if missing
        if ph == "?":
            cols = [row[1] for row in cur.execute("PRAGMA table_info(analysis)").fetchall()]
            if "source" not in cols:
                cur.execute("ALTER TABLE analysis ADD COLUMN source TEXT DEFAULT 'manual'")
        else:
            cur.execute("""
                DO $$ BEGIN
                    ALTER TABLE analysis ADD COLUMN source TEXT DEFAULT 'manual';
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def hash_password(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_password(pw, hashed):
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def add_user(username, password):
    with _conn() as (con, ph):
        cur = con.cursor()
        try:
            cur.execute(_sql("INSERT INTO users VALUES (?,?)", ph), (username, hash_password(password)))
            return True
        except Exception:
            return False


def get_user(username):
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(_sql("SELECT * FROM users WHERE username=?", ph), (username,))
        row = cur.fetchone()
        return tuple(row) if row else None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def save_analysis(username, company, ticker, data, score, rec, source="manual"):
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(
            _sql(
                "INSERT INTO analysis (username, company_name, ticker, date, data_json, score, recommendation, source) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ph,
            ),
            (username, company, ticker,
             datetime.now().strftime("%Y-%m-%d %H:%M"),
             json.dumps(data), score, rec, source),
        )
        if ph == "?":
            return cur.lastrowid
        else:
            cur.execute("SELECT lastval()")
            return cur.fetchone()[0]


def get_history(username, ticker=None, include_auto=False):
    with _conn() as (con, ph):
        if include_auto:
            q = _sql("SELECT * FROM analysis WHERE (username=? OR source='auto')", ph)
            params = [username]
        else:
            q = _sql("SELECT * FROM analysis WHERE username=?", ph)
            params = [username]
        if ticker:
            q += _sql(" AND ticker=?", ph)
            params.append(ticker)
        return pd.read_sql_query(q, con, params=params)


def get_ticker_trend_data(username, ticker):
    df = get_history(username, ticker, include_auto=True)
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
    from core.scoring import calculate_graham_score  # avoid circular at module level
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


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def add_to_watchlist(username, ticker, company_name):
    with _conn() as (con, ph):
        cur = con.cursor()
        if ph == "?":
            cur.execute(
                _sql("INSERT OR IGNORE INTO watchlist (username, ticker, company_name, added_date) VALUES (?,?,?,?)", ph),
                (username, ticker.upper(), company_name, datetime.now().strftime("%Y-%m-%d")),
            )
        else:
            cur.execute(
                _sql("INSERT INTO watchlist (username, ticker, company_name, added_date) VALUES (?,?,?,?) ON CONFLICT DO NOTHING", ph),
                (username, ticker.upper(), company_name, datetime.now().strftime("%Y-%m-%d")),
            )
        return con.cursor().rowcount > 0 if ph == "?" else True


def remove_from_watchlist(username, ticker):
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(_sql("DELETE FROM watchlist WHERE username=? AND ticker=?", ph), (username, ticker.upper()))


def is_on_watchlist(username, ticker):
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(_sql("SELECT 1 FROM watchlist WHERE username=? AND ticker=?", ph), (username, ticker.upper()))
        return cur.fetchone() is not None


def get_watchlist_with_scores(username):
    from core.scoring import calculate_graham_score
    with _conn() as (con, ph):
        wl = pd.read_sql_query(
            _sql("SELECT ticker, company_name, added_date FROM watchlist WHERE username=?", ph),
            con, params=(username,))
        items = []
        for _, row in wl.iterrows():
            ticker = row['ticker']
            # Fetch latest 2 analyses for this ticker regardless of who ran them
            analyses = pd.read_sql_query(
                _sql("SELECT score, recommendation, date, data_json, source FROM analysis "
                     "WHERE ticker=? ORDER BY date DESC LIMIT 2", ph),
                con, params=(ticker,))
            entry = {
                'ticker': ticker, 'company_name': row['company_name'],
                'added_date': row['added_date'], 'last_score': None,
                'last_rec': None, 'last_mos': None,
                'last_analyzed': None, 'score_change': None, 'source': None,
            }
            if not analyses.empty:
                latest = analyses.iloc[0]
                entry['last_score'] = int(latest['score'])
                entry['last_rec'] = latest['recommendation']
                entry['last_analyzed'] = latest['date']
                entry['source'] = latest.get('source', 'manual')
                try:
                    d = json.loads(latest['data_json'])
                    _, _, _, mos, _ = calculate_graham_score(d)
                    entry['last_mos'] = round(mos, 1)
                except Exception:
                    pass
                if len(analyses) >= 2:
                    entry['score_change'] = int(latest['score']) - int(analyses.iloc[1]['score'])
            items.append(entry)
        return items


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def add_alert(username, ticker, company_name, alert_type, threshold, email):
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(
            _sql("INSERT INTO alerts (username, ticker, company_name, alert_type, threshold, email, active) "
                 "VALUES (?,?,?,?,?,?,1)", ph),
            (username, ticker.upper(), company_name, alert_type, threshold, email),
        )


def get_alerts(username):
    with _conn() as (con, ph):
        return pd.read_sql_query(
            _sql("SELECT * FROM alerts WHERE username=? ORDER BY id DESC", ph),
            con, params=(username,))


def delete_alert(alert_id):
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(_sql("DELETE FROM alerts WHERE id=?", ph), (alert_id,))


def check_and_fire_alerts(username, ticker, company_name, current_score, current_mos):
    """Fire alerts for a single user + ticker (used after manual analysis)."""
    return _fire_alerts_for_rows(
        ticker, company_name, current_score, current_mos, username_filter=username
    )


def check_and_fire_alerts_for_ticker(ticker, company_name, current_score, current_mos):
    """Fire alerts for ALL users who have rules on this ticker (used by pipeline)."""
    return _fire_alerts_for_rows(
        ticker, company_name, current_score, current_mos, username_filter=None
    )


def _fire_alerts_for_rows(ticker, company_name, current_score, current_mos, username_filter=None):
    with _conn() as (con, ph):
        cur = con.cursor()
        if username_filter:
            cur.execute(
                _sql("SELECT id, alert_type, threshold, email FROM alerts WHERE username=? AND ticker=? AND active=1", ph),
                (username_filter, ticker.upper()),
            )
        else:
            cur.execute(
                _sql("SELECT id, alert_type, threshold, email FROM alerts WHERE ticker=? AND active=1", ph),
                (ticker.upper(),),
            )
        alert_rows = cur.fetchall()

    fired = []
    for row in alert_rows:
        alert_id, alert_type, threshold, email = row[0], row[1], row[2], row[3]
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
            ok, _ = send_alert_email(
                email,
                subject,
                body + "\n\n---\nGraham-Bot | For educational purposes only.",
            )
            if ok:
                with _conn() as (con, ph):
                    con.cursor().execute(
                        _sql("UPDATE alerts SET last_triggered=? WHERE id=?", ph),
                        (datetime.now().strftime("%Y-%m-%d %H:%M"), alert_id),
                    )
                fired.append(f"{alert_type} → {email}")
    return fired


# ---------------------------------------------------------------------------
# Pipeline deduplication
# ---------------------------------------------------------------------------

def is_report_processed(pdf_url: str) -> bool:
    with _conn() as (con, ph):
        cur = con.cursor()
        cur.execute(_sql("SELECT 1 FROM processed_reports WHERE pdf_url=?", ph), (pdf_url,))
        return cur.fetchone() is not None


def mark_report_processed(ticker, company_name, report_type, fiscal_year, pdf_url, analysis_id=None):
    with _conn() as (con, ph):
        cur = con.cursor()
        if ph == "?":
            cur.execute(
                "INSERT OR IGNORE INTO processed_reports "
                "(ticker, company_name, report_type, fiscal_year, pdf_url, processed_at, analysis_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (ticker, company_name, report_type, fiscal_year, pdf_url,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), analysis_id),
            )
        else:
            cur.execute(
                "INSERT INTO processed_reports "
                "(ticker, company_name, report_type, fiscal_year, pdf_url, analysis_id) "
                "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (pdf_url) DO NOTHING",
                (ticker, company_name, report_type, fiscal_year, pdf_url, analysis_id),
            )
