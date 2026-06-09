"""
CSE announcement fetcher.
Uses the public CSE API (no auth required).
"""
import logging
import os
import tempfile

import requests

log = logging.getLogger(__name__)

CSE_API_URL = "https://www.cse.lk/api/getFinancialAnnouncement"
PDF_CDN = "https://cdn.cse.lk/cmt/upload_report_file/{fileName}"

_REPORT_KEYWORDS = [
    "annual report",
    "financial statements",
    "interim",
    "quarterly",
    "audited financial",
    "unaudited financial",
]


def fetch_announcements(page: int = 1, page_size: int = 100) -> list[dict]:
    """POST to CSE financial-announcement API and return the raw list."""
    try:
        resp = requests.post(
            CSE_API_URL,
            data={"page": page, "pageSize": page_size},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # API returns {"reqStatus": "SUCCESS", "responseObject": [...]}
        items = data.get("responseObject") or []
        if not isinstance(items, list):
            log.warning("Unexpected CSE API response shape: %s", type(items))
            return []
        log.info("Fetched %d announcements from CSE (page %d)", len(items), page)
        return items
    except Exception as e:
        log.error("CSE API fetch failed: %s", e)
        return []


def filter_reports(announcements: list[dict]) -> list[dict]:
    """Keep only financial-statement announcements that carry a PDF."""
    results = []
    for item in announcements:
        title = (item.get("announcementTitle") or "").lower()
        category = (item.get("categoryType") or "").lower()
        file_name = item.get("fileName") or ""
        if not file_name:
            continue
        combined = title + " " + category
        if any(kw in combined for kw in _REPORT_KEYWORDS):
            results.append(item)
    log.info("%d / %d announcements are financial reports", len(results), len(announcements))
    return results


def get_pdf_url(announcement: dict) -> str:
    file_name = announcement.get("fileName", "")
    return PDF_CDN.format(fileName=file_name)


def get_report_metadata(announcement: dict) -> dict:
    """Extract ticker, company name, report type, and fiscal year from raw announcement."""
    title = announcement.get("announcementTitle") or ""
    company = announcement.get("companyName") or announcement.get("company") or ""
    ticker = announcement.get("symbol") or announcement.get("ticker") or ""
    category = (announcement.get("categoryType") or "").lower()

    if "annual" in category or "annual" in title.lower():
        report_type = "annual"
    elif "interim" in category or "quarterly" in category:
        report_type = "quarterly"
    else:
        report_type = "financial"

    # Fiscal year: prefer explicit field, fall back to announcement date year
    fiscal_year = str(announcement.get("fiscalYear") or "")
    if not fiscal_year:
        date_str = announcement.get("announcementDate") or announcement.get("date") or ""
        fiscal_year = date_str[:4] if date_str else ""

    return {
        "ticker": ticker.upper(),
        "company_name": company,
        "report_type": report_type,
        "fiscal_year": fiscal_year,
    }


def download_pdf(pdf_url: str, dest_path: str) -> bool:
    """Stream-download a PDF to dest_path. Returns True on success."""
    try:
        resp = requests.get(pdf_url, stream=True, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not pdf_url.endswith(".pdf"):
            log.warning("Unexpected Content-Type '%s' for %s", content_type, pdf_url)
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
