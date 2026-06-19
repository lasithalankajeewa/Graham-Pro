"""
Entry point for the daily GitHub Actions pipeline.
Run as: python -m pipeline.main
"""
import logging
import os
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("pipeline.main")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "reports"

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        log.error("DATABASE_URL is not set — aborting")
        sys.exit(1)

    log.info("Initializing database schema...")
    from core.db import init_db
    init_db()

    if mode == "price-check":
        log.info("Running price-only alert check...")
        from pipeline.runner import run_price_check
        stats = run_price_check()
        log.info("Done. alerts_fired=%d", stats["alerts_fired"])
        return

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        log.error("OPENROUTER_API_KEY is not set — aborting")
        sys.exit(1)

    # model used for ALL pipeline analyses
    model_id = os.getenv("PIPELINE_MODEL") or "openai/gpt-oss-120b:free"

    backfill_since = os.getenv("BACKFILL_SINCE") or None

    log.info("Starting CSE pipeline with model: %s", model_id)
    log.info("Database backend: %s", "PostgreSQL" if "postgresql" in database_url else "SQLite")
    if backfill_since:
        log.info("BACKFILL MODE: fetching all reports since %s", backfill_since)

    with tempfile.TemporaryDirectory(prefix="graham_pipeline_") as tmp_dir:
        from pipeline.runner import run_pipeline
        stats = run_pipeline(
            model_id=model_id,
            openrouter_key=openrouter_key,
            tmp_dir=tmp_dir,
            backfill_since=backfill_since,
        )

    log.info("Done. processed=%d skipped=%d failed=%d alerts=%d",
             stats["processed"], stats["skipped"], stats["failed"], stats["alerts_fired"])

    if stats["processed"] == 0 and stats["failed"] > 0:
        log.warning("All attempts failed — check logs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
