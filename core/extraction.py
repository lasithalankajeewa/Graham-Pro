"""
Pure-Python extraction helpers — no Streamlit imports.
Used by both graham_bot.py (via thin Streamlit wrappers) and the headless pipeline.
"""
import json
import logging
import os
import re
import time

import requests
from pypdf import PdfReader, PdfWriter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared prompt
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def _parse_json_response(text):
    """Robustly extract a JSON object from a model response."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    text = text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        chunk = text[start:end + 1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
            return json.loads(chunk)

    raise ValueError(f"No JSON object found in model response: {text[:200]}")

# ---------------------------------------------------------------------------
# TOC detection helpers
# ---------------------------------------------------------------------------

def detect_page_offset(reader, max_scan=20):
    # Fix 4: try pypdf page labels first (most reliable)
    try:
        labels = list(reader.page_labels)
        for i, label in enumerate(labels):
            if label and str(label).isdigit():
                n = int(label)
                if 1 <= n <= 10:
                    return i - (n - 1)
    except Exception:
        pass

    # Fall back to heuristic: look for small digit in top/bottom lines
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


def _find_toc_page_indices(reader, max_scan=30):
    total = len(reader.pages)
    found = []
    for i in range(min(max_scan, total)):
        text = reader.pages[i].extract_text() or ""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[:20]:
            if len(line) <= 60 and _TOC_HEADING_RE.search(line):
                for j in range(i, min(i + 3, total)):
                    if j not in found:
                        found.append(j)
                break
    if found:
        return sorted(found), True
    return list(range(min(10, total))), False


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


def parse_toc_loosely(toc_text):
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
        if not any(kw in clean.lower() for kw in financial_keywords):
            continue
        nums = re.findall(r'\b(\d{1,4})\b', clean)
        if not nums:
            continue
        page = int(nums[-1])
        if 1 <= page <= 2000:
            name = re.sub(r'[\s\d]+$', '', clean).strip().rstrip('.-–—|')
            if name and not any(s['printed_page'] == page for s in sections):
                sections.append({'name': name, 'printed_page': page})
    return sections if sections else []


# ---------------------------------------------------------------------------
# Fix 1: smarter keyword scan — dedup by keyword type, heading-only lines
# ---------------------------------------------------------------------------

def _keyword_scan(reader, total_pages):
    """
    Scan the full document for financial statement section headings.
    Returns (set of 0-based PDF indices, page_sections list).

    Two constraints vs the old scan:
    1. Dedup by keyword — each statement type matched at most once, taking the
       first (earliest) occurrence which is the actual section heading page.
    2. Heading-only — keyword must appear on a short line (≤ 80 chars) so it's
       a title, not buried in paragraph text or notes.
    """
    found_keywords = set()
    relevant_indices = set()
    page_sections = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            if len(line) > 80:
                continue  # skip paragraph text
            low = line.lower()
            for kw in _TOC_KEYWORDS:
                if kw in low and kw not in found_keywords:
                    found_keywords.add(kw)
                    relevant_indices.add(i)
                    if i + 1 < total_pages:
                        relevant_indices.add(i + 1)
                    sent = [i + 1] + ([i + 2] if i + 1 < total_pages else [])
                    page_sections.append({'name': kw.title(), 'toc_page': None, 'pdf_pages': sent})
                    break  # one keyword per line is enough

    return relevant_indices, page_sections


# ---------------------------------------------------------------------------
# Fix 3: OpenRouter TOC parser — used in pipeline mode when local parsers fail
# ---------------------------------------------------------------------------

_TOC_AI_PROMPT = """This is text extracted from a financial report's table of contents / index page.
Identify the printed page numbers for ALL financial statement sections.

Look for ANY of these (use exact name from the document):
- Income Statement / Statement of Profit or Loss / Comprehensive Income
- Statement of Financial Position / Balance Sheet
- Statement of Changes in Equity
- Statement of Cash Flows
- Notes to Financial Statements / Accounting Policies
- Financial Highlights / Five-Year Summary / Ten-Year Summary
- Per Share Data / Investor Information / Shareholders Information
- Financial Ratios / Key Performance Indicators / Capital Market Data

Return ONLY valid JSON (no markdown):
{"toc_found": true, "sections": [{"name": "exact name", "printed_page": 85}]}
If nothing readable: {"toc_found": false, "sections": []}

Document text:
"""


def _parse_toc_with_openrouter(toc_text, model_id, openrouter_key):
    """Call OpenRouter with only the TOC page text to extract section page numbers."""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": _TOC_AI_PROMPT + toc_text[:8000]}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        result = _parse_json_response(raw)
        sections = result.get("sections", []) if result.get("toc_found") else []
        log.info("OpenRouter TOC parse: %d sections found", len(sections))
        return sections
    except Exception as e:
        log.warning("OpenRouter TOC parse failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Page relevance detection
# notify_fn receives (level, message) where level is 'info'|'warning'|'success'
# ---------------------------------------------------------------------------

def _find_relevant_pages(reader, allow_gemini_toc=True, notify_fn=None,
                          openrouter_key=None, openrouter_model=None):
    """
    Shared page-detection logic for both Gemini and OpenRouter paths.
    Returns (page_map, sorted list of 0-based PDF indices).

    allow_gemini_toc=False  — skip Gemini API call (pipeline mode)
    openrouter_key/model    — when provided, use OpenRouter for AI TOC parsing
                              instead of (or as fallback after) Gemini
    notify_fn(level, msg)   — progress callbacks; defaults to logging
    """
    def _notify(level, msg):
        if notify_fn:
            notify_fn(level, msg)
        else:
            getattr(log, level if level != 'success' else 'info')(msg)

    total_pages = len(reader.pages)
    page_map = {'method': None, 'total_pages': total_pages, 'sections': [], 'pages_sent': []}

    _notify('info', f"Report has {total_pages} pages. Locating contents page...")
    toc_indices, heading_found = _find_toc_page_indices(reader)

    toc_text = ""
    for i in toc_indices:
        toc_text += f"\n--- PDF Page {i + 1} ---\n{reader.pages[i].extract_text() or ''}\n"

    toc_data = {"toc_found": False, "sections": []}

    if heading_found:
        toc_label = f"PDF page(s) {[i + 1 for i in toc_indices]}"
        _notify('info', f"Contents page found at {toc_label}. Reading entries...")

        # Attempt 1: strict local parse (requires separator chars)
        sections = parse_toc_locally(toc_text)
        if sections:
            _notify('info', f"Parsed locally — {len(sections)} financial sections found (no API call).")
            toc_data = {"toc_found": True, "sections": sections}
        else:
            # Attempt 2: loose local parse (keyword + any trailing number)
            sections = parse_toc_loosely(toc_text)
            if sections:
                _notify('info', f"Parsed (loose match) — {len(sections)} financial sections found (no API call).")
                toc_data = {"toc_found": True, "sections": sections}
            else:
                # Attempt 3a: OpenRouter AI parse (pipeline mode)
                if openrouter_key and openrouter_model:
                    _notify('info', "Local parse inconclusive — using OpenRouter to read contents page...")
                    sections = _parse_toc_with_openrouter(toc_text, openrouter_model, openrouter_key)
                    if sections:
                        toc_data = {"toc_found": True, "sections": sections}
                # Attempt 3b: Gemini AI parse (Streamlit mode)
                elif allow_gemini_toc:
                    import google.generativeai as genai
                    _notify('info', "Local parse inconclusive — using Gemini to read contents page...")
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    toc_prompt = (
                        _TOC_AI_PROMPT.replace(
                            "Return ONLY valid JSON (no markdown):",
                            "Return ONLY valid JSON (no markdown, no extra text):"
                        ) + toc_text[:12000]
                    )
                    toc_resp = model.generate_content(toc_prompt)
                    toc_data = _parse_json_response(toc_resp.text)
    else:
        # Fix 1: smarter keyword scan when no TOC heading found
        _notify('warning', "No contents page found — scanning all pages for financial statement keywords.")
        page_map['method'] = 'keyword_scan'
        relevant_pdf_indices, page_sections = _keyword_scan(reader, total_pages)
        page_map['sections'] = page_sections
        relevant_pages = sorted(relevant_pdf_indices)
        if not relevant_pages:
            page_map['method'] = 'fallback'
            _notify('warning', "No financial pages found — using last quarter of document.")
            start = max(0, int(total_pages * 0.75))
            relevant_pages = list(range(start, min(start + 20, total_pages)))
            page_map['sections'] = [{'name': 'Fallback — last quarter', 'toc_page': None,
                                      'pdf_pages': [p + 1 for p in relevant_pages]}]
        page_map['pages_sent'] = [p + 1 for p in relevant_pages]
        return page_map, relevant_pages

    # Map TOC section printed page numbers → PDF indices
    relevant_pdf_indices = set()
    if toc_data.get("toc_found") and toc_data.get("sections"):
        offset = detect_page_offset(reader)
        page_map['method'] = 'toc'
        _notify('success', f"Contents page — {len(toc_data['sections'])} financial sections identified.")
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

    # Fix 2: when TOC found but not parsed, use keyword scan (not "first 15 pages")
    if not relevant_pdf_indices:
        page_map['method'] = 'keyword_scan_fallback'
        _notify('warning', "Contents page found but entries could not be read — falling back to keyword scan.")
        relevant_pdf_indices, page_sections = _keyword_scan(reader, total_pages)
        page_map['sections'] = page_sections
        if not relevant_pdf_indices:
            # Absolute last resort: financial statements are always in the latter part
            page_map['method'] = 'fallback'
            start = max(0, int(total_pages * 0.75))
            relevant_pdf_indices = set(range(start, min(start + 20, total_pages)))
            page_map['sections'] = [{'name': 'Fallback — last quarter', 'toc_page': None,
                                      'pdf_pages': [p + 1 for p in sorted(relevant_pdf_indices)]}]
            _notify('warning', f"Keyword scan also empty — using last quarter of document (pages {start+1}+).")

    relevant_pages = sorted(relevant_pdf_indices)
    page_map['pages_sent'] = [p + 1 for p in relevant_pages]
    return page_map, relevant_pages

# ---------------------------------------------------------------------------
# Headless OpenRouter extraction  (used by pipeline; no Streamlit)
# ---------------------------------------------------------------------------

def extract_with_openrouter(pdf_path: str, model_id: str, openrouter_key: str):
    """
    Headless extraction via OpenRouter.
    pdf_path: filesystem path to the PDF (not st.UploadedFile).
    Returns (raw_dict, page_map) or (None, None) on failure.
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            # Fix 3: forward key/model so OpenRouter TOC parsing is available in pipeline mode
            page_map, relevant_pages = _find_relevant_pages(
                reader,
                allow_gemini_toc=False,
                openrouter_key=openrouter_key,
                openrouter_model=model_id,
            )
            total_pages = page_map['total_pages']

            log.info(f"Extracting text from {len(relevant_pages)} of {total_pages} targeted pages...")
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

        for attempt in range(3):
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 20))
                wait = max(retry_after, 20) * (attempt + 1)
                log.warning(f"Rate limit — waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            raw_text = msg.get("content") or msg.get("reasoning_content") or ""
            return _parse_json_response(raw_text), page_map

        log.error("Rate limit: all 3 retries exhausted.")
        return None, None

    except Exception as e:
        log.error(f"Extraction error: {e}")
        return None, None
