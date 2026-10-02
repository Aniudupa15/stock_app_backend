"""Trains (or retrains) the ML momentum model on all price history to date.

    python scripts/train_ml_model.py

Run periodically (weekly is the scheduled cadence - see the TrainMLModel
task) - an expanding-window model only ever gains data, so more-frequent
retraining is harmless. compute_ml_ranking() (used by scripts/ml_picks_email.py)
reads whatever model this last saved; it doesn't retrain itself.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from services.trading_service.momentum.ml_ranking import train_and_save_model  # noqa: E402


async def main() -> None:
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with sf() as s:
            meta = await train_and_save_model(s)
        print(
            f"{datetime.now().isoformat()} | Trained on {meta['n_rows']:,} rows, "
            f"{meta['n_symbols']} symbols, {meta['date_range'][0]} to {meta['date_range'][1]}"
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
