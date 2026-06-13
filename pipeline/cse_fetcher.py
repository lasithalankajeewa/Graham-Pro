"""
CSE announcement fetcher.
Uses the public CSE API (no auth required).

Actual API response shape (field names differ from docs):
  { "reqFinancialAnnouncemnets": [   ← note typo in key
      {
        "id": 51873,
        "path": "cmt/upload_report_file/2352_1781262197673.pdf",
        "manualDate": 1767119400000,
        "uploadedDate": "12 Jun 2026 04:33:17 PM",
        "fileText": "Annual Report as at 31st March 2026",
        "name": "HAYCARB PLC",
        "symbol": "HAYC",
        "logoUrl": "...",
        "authorizedDate": "..."
      }, ...
  ]}
PDF URL = https://cdn.cse.lk/{path}
"""
import logging
import os
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

CSE_API_URL = "https://www.cse.lk/api/getFinancialAnnouncement"
PDF_CDN_BASE = "https://cdn.cse.lk/"

_REPORT_KEYWORDS = [
    "annual report",
    "financial statements",
    "interim financial",
    "quarterly",
    "audited financial",
    "unaudited financial",
]


def fetch_announcements(page: int = 1, page_size: int = 100) -> list[dict]:
    """POST to CSE financial-announcement API and return the raw list."""
    try:
        resp = requests.post(
            CSE_API_URL,
            json={"page": page, "pageSize": page_size},
            timeout=30,
        )
        resp.raise_for_status()
        # API key has a typo: "reqFinancialAnnouncemnets"
        items = resp.json().get("reqFinancialAnnouncemnets") or []
        if not isinstance(items, list):
            log.warning("Unexpected CSE API response shape: %s", type(items))
            return []
        log.info("Fetched %d announcements from CSE (page %d)", len(items), page)
        return items
    except Exception as e:
        log.error("CSE API fetch failed: %s", e)
        return []


def filter_reports(announcements: list[dict]) -> list[dict]:
    """Keep only financial-statement announcements that carry a PDF path."""
    results = []
    for item in announcements:
        title = (item.get("fileText") or "").lower()
        path = item.get("path") or ""
        if not path or not path.endswith(".pdf"):
            continue
        if any(kw in title for kw in _REPORT_KEYWORDS):
            results.append(item)
    log.info("%d / %d announcements are financial reports", len(results), len(announcements))
    return results


def get_pdf_url(announcement: dict) -> str:
    path = announcement.get("path", "")
    return PDF_CDN_BASE + path


def get_report_metadata(announcement: dict) -> dict:
    """Extract ticker, company name, report type, and fiscal year from raw announcement."""
    title = (announcement.get("fileText") or "").lower()
    company = announcement.get("name") or ""
    ticker = announcement.get("symbol") or ""

    if "annual" in title:
        report_type = "annual"
    elif "interim" in title or "quarterly" in title or "quarter" in title:
        report_type = "quarterly"
    else:
        report_type = "financial"

    # Extract year from title e.g. "Annual Report as at 31st March 2026" → "2026"
    import re
    years = re.findall(r'\b(20\d{2})\b', announcement.get("fileText") or "")
    fiscal_year = years[-1] if years else ""

    return {
        "ticker": ticker.upper(),
        "company_name": company,
        "report_type": report_type,
        "fiscal_year": fiscal_year,
    }


def parse_announcement_date(item: dict) -> datetime | None:
    """Parse uploadedDate string → UTC datetime.
    manualDate is the fiscal-year reference date, NOT the upload date — don't use it.
    """
    uploaded = item.get("uploadedDate") or ""
    for fmt in ("%d %b %Y %I:%M:%S %p", "%d %b %Y %H:%M:%S", "%d %b %Y"):
        try:
            return datetime.strptime(uploaded, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_all_since(cutoff_date: datetime, page_size: int = 100) -> list[dict]:
    """Paginate CSE API returning all announcements with date >= cutoff_date.
    API returns newest-first. Stops when an entire page falls before cutoff. Safety cap: 50 pages.
    """
    _min_dt = datetime.min.replace(tzinfo=timezone.utc)
    all_items: list[dict] = []
    for page in range(1, 51):
        items = fetch_announcements(page=page, page_size=page_size)
        if not items:
            break
        in_range = [i for i in items if (parse_announcement_date(i) or _min_dt) >= cutoff_date]
        all_items.extend(in_range)
        if not in_range:
            break
    log.info("fetch_all_since: %d reports found since %s", len(all_items), cutoff_date.date())
    return all_items


def download_pdf(pdf_url: str, dest_path: str) -> bool:
    """Stream-download a PDF to dest_path. Returns True on success."""
    try:
        resp = requests.get(pdf_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size = os.path.getsize(dest_path)
        if size < 1024:
            log.warning("Downloaded file suspiciously small (%d bytes): %s", size, pdf_url)
            return False
        log.info("Downloaded %d KB: %s", size // 1024, pdf_url)
        return True
    except Exception as e:
        log.error("PDF download failed (%s): %s", pdf_url, e)
        return False
