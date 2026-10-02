from __future__ import annotations

import asyncio
import os
import subprocess

import uvicorn

from .app import app
from .bootstrap import ensure_runtime
from .discord_bot import create_bot


async def wait_for_llama(timeout: float = 90.0) -> None:
    import httpx
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"http://127.0.0.1:{os.getenv('LLAMA_PORT', '8090')}/health")
            if response.status_code in (200, 503):
                print("[llama] server is reachable", flush=True)
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("llama-server did not start")


async def main() -> None:
    llama_bin, model = await asyncio.to_thread(ensure_runtime)
    llama_port = int(os.getenv("LLAMA_PORT", "8090"))
    context = int(os.getenv("LLAMA_CONTEXT", "2048"))
    max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "256"))
    threads = int(os.getenv("LLAMA_THREADS", "1"))
    command = [str(llama_bin), "-m", str(model), "--host", "127.0.0.1", "--port", str(llama_port), "-c", str(context), "-n", str(max_tokens), "-ngl", "0", "-t", str(threads), "-tb", str(threads), "--parallel", "1"]
    print("[llama] starting:", " ".join(command), flush=True)
    process = subprocess.Popen(command)
    try:
        os.environ["LLAMA_BASE"] = f"http://127.0.0.1:{llama_port}"
        await wait_for_llama()
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
            print("[discord] DISCORD_TOKEN not set; HTTP API only", flush=True)
        try:
            await asyncio.gather(*tasks)
        finally:
            if bot is not None:
                await bot.close()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    asyncio.run(main())
