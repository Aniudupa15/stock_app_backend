"""Manually trigger the NSE equity universe sync, without waiting for the
scheduled job. Useful for local development/demo.

    python scripts/run_universe_sync.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.scheduler.jobs import run_universe_sync


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    await run_universe_sync(settings)


if __name__ == "__main__":
    asyncio.run(main())
