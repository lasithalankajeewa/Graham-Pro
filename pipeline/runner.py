"""
Pipeline orchestrator.
For each new CSE financial report:
  1. Check deduplication
  2. Download PDF
  3. Extract financials via OpenRouter
  4. Score and save to DB
  5. Fire email alerts for any matching user rules
"""
import logging
import os
import tempfile

from core.db import (
    is_report_processed,
    mark_report_processed,
    save_analysis,
    check_and_fire_alerts_for_ticker,
)
from core.extraction import extract_with_openrouter
from core.scoring import calculate_full_analysis, calculate_graham_score
from pipeline.cse_fetcher import (
    download_pdf,
    fetch_all_since,
    fetch_announcements,
    filter_reports,
    get_pdf_url,
    get_report_metadata,
)

log = logging.getLogger(__name__)

PIPELINE_USER = "_pipeline_"


def run_pipeline(
    model_id: str,
    openrouter_key: str,
    tmp_dir: str | None = None,
    max_reports: int | None = None,
    backfill_since: str | None = None,
) -> dict:
    """
    Run a full pipeline cycle. Returns summary counts.
    max_reports: cap for testing (None = no cap).
    backfill_since: ISO date string "YYYY-MM-DD" — fetches all pages since that date.
    """
    stats = {"processed": 0, "skipped": 0, "failed": 0, "alerts_fired": 0}

    if backfill_since:
        from datetime import datetime, timezone
        cutoff = datetime.fromisoformat(backfill_since).replace(tzinfo=timezone.utc)
        log.info("Backfill mode: fetching all reports since %s", cutoff.date())
        announcements = fetch_all_since(cutoff)
    else:
        announcements = fetch_announcements()
    reports = filter_reports(announcements)

    if max_reports:
        reports = reports[:max_reports]

    log.info("Starting pipeline: %d reports to evaluate", len(reports))

    work_dir = tmp_dir or tempfile.mkdtemp(prefix="graham_pipeline_")

    for i, item in enumerate(reports, 1):
        pdf_url = get_pdf_url(item)
        meta = get_report_metadata(item)
        ticker = meta["ticker"]
        company = meta["company_name"]
        report_type = meta["report_type"]
        fiscal_year = meta["fiscal_year"]

        log.info("[%d/%d] %s — %s (%s %s)", i, len(reports), ticker, company, report_type, fiscal_year)

        if is_report_processed(pdf_url):
            log.info("  → already processed, skipping")
            stats["skipped"] += 1
            continue

        pdf_path = os.path.join(work_dir, f"{ticker}_{i}.pdf")
        try:
            ok = download_pdf(pdf_url, pdf_path)
            if not ok:
                log.warning("  → download failed, skipping")
                stats["failed"] += 1
                continue

            raw, page_map = extract_with_openrouter(pdf_path, model_id, openrouter_key)
            if raw is None:
                log.warning("  → extraction returned None, skipping")
                stats["failed"] += 1
                continue

            # CSE API symbol/name are authoritative — always override model extraction.
            # The model sometimes returns wrong tickers, placeholders, or empty strings.
            raw["ticker"] = ticker  # always use CSE symbol
            raw["company_name"] = company  # always use CSE registered name
            if not raw.get("fiscal_year"):
                raw["fiscal_year"] = fiscal_year

            analysis = calculate_full_analysis(raw)
            score = analysis["graham_score"]
            rec   = analysis["recommendation"]
            mos   = analysis["mos_pct"]

            raw["report_type"] = report_type
            # Merge full analysis into stored data so it's available in the UI
            merged = {**raw, "_analysis": analysis}

            analysis_id = save_analysis(
                username=PIPELINE_USER,
                company=company,
                ticker=ticker,
                data=merged,
                score=score,
                rec=rec,
                source="auto",
            )

            mark_report_processed(
                ticker=raw.get("ticker", ticker).upper(),
                company_name=raw.get("company_name", company),
                report_type=report_type,
                fiscal_year=str(raw.get("fiscal_year", fiscal_year)),
                pdf_url=pdf_url,
                analysis_id=analysis_id,
            )

            fired = check_and_fire_alerts_for_ticker(
                ticker=raw.get("ticker", ticker).upper(),
                company_name=raw.get("company_name", company),
                current_score=score,
                current_mos=mos,
            )
            stats["alerts_fired"] += len(fired)
            if fired:
                log.info("  → %d alert(s) fired: %s", len(fired), fired)

            log.info("  → score=%d rec=%s mos=%.1f%% analysis_id=%s", score, rec, mos, analysis_id)
            stats["processed"] += 1

        except Exception as e:
            log.error("  → unexpected error for %s: %s", pdf_url, e, exc_info=True)
            stats["failed"] += 1
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    log.info(
        "Pipeline complete — processed=%d skipped=%d failed=%d alerts=%d",
        stats["processed"], stats["skipped"], stats["failed"], stats["alerts_fired"],
    )
    return stats


def run_price_check() -> dict:
    """Daily/intraday price-only check: fetch live CSE prices, fire any due price_below alerts.
    Runs independently of report extraction — no OpenRouter key needed.
    """
    from core.cse_market import get_live_prices
    from core.db import fire_price_alerts

    stats = {"alerts_fired": 0}
    live_prices = get_live_prices()
    if not live_prices:
        log.warning("Price check: no live prices fetched — aborting")
        return stats

    fired = fire_price_alerts(live_prices)
    stats["alerts_fired"] = len(fired)
    if fired:
        log.info("Price check fired %d alert(s): %s", len(fired), fired)
    else:
        log.info("Price check complete — no thresholds crossed")
    return stats
