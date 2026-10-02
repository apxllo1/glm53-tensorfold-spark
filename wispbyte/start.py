[Reading 34 lines from start (total: 34 lines, 0 remaining)]

from __future__ import annotations

import asyncio
import os

import uvicorn

from .app import app
from .discord_bot import create_bot


async def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    tasks = [asyncio.create_task(server.serve())]

    token = os.getenv("DISCORD_TOKEN", "").strip()
    bot = None
    if token:
        bot = create_bot()
        tasks.append(asyncio.create_task(bot.start(token)))
    else:
        print("[discord] DISCORD_TOKEN not set; HTTP model API only")

    try:
        await asyncio.gather(*tasks)
    finally:
        if bot is not None:
            await bot.close()


if __name__ == "__main__":
    asyncio.run(main())

[executed on device: Alexanders-MacBook-Pro.local (bd265cb2-62d5-47e5-937a-511d5a03b340)]