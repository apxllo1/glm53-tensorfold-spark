[Reading 117 lines from start (total: 117 lines, 0 remaining)]

from __future__ import annotations

import json
import os
from pathlib import Path

import discord
import httpx
from discord.ext import commands

DATA_DIR = Path(os.getenv("BOT_DATA_DIR", "./data"))
CHANNEL_FILE = DATA_DIR / "channels.json"
API_BASE = os.getenv("MODEL_API_BASE", "http://127.0.0.1:8000")
API_KEY = os.getenv("APP_API_KEY", "")
PREFIX = os.getenv("BOT_PREFIX", "!")
MAX_REPLY = 1900


def _headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _load_channels() -> set[int]:
    try:
        return {int(x) for x in json.loads(CHANNEL_FILE.read_text())}
    except Exception:
        return set()


def _save_channels(channels: set[int]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHANNEL_FILE.write_text(json.dumps(sorted(channels), indent=2))


class GLMBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=PREFIX, intents=intents, help_command=None)
        self.allowed_channels = _load_channels()
        self._lock = False

    async def setup_hook(self):
        print(f"[discord] ready; prefix={PREFIX}; allowed_channels={sorted(self.allowed_channels)}")

    async def ask_model(self, user_text: str, channel_id: int) -> str:
        async with httpx.AsyncClient(timeout=float(os.getenv("MODEL_TIMEOUT", "900"))) as client:
            tools_resp = await client.get(f"{API_BASE}/v1/tools", headers=_headers())
            tools = tools_resp.json().get("tools", []) if tools_resp.status_code == 200 else []
            body = {
                "model": os.getenv("MODEL_NAME", "local-model"),
                "max_tokens": int(os.getenv("DISCORD_MAX_TOKENS", "800")),
                "messages": [{"role": "user", "content": user_text}],
                "tools": tools,
            }
            r = await client.post(f"{API_BASE}/v1/messages", headers=_headers(), json=body)
            r.raise_for_status()
            data = r.json()
        parts = []
        for part in data.get("content", []):
            if part.get("type") == "text" and part.get("text"):
                parts.append(part["text"])
            elif part.get("type") == "tool_use":
                parts.append(f"Tool requested: `{part.get('name')}`\n```json\n{json.dumps(part.get('input', {}), indent=2)}\n```")
        return "\n\n".join(parts) or "(The model returned no text.)"

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not self.allowed_channels:
            return
        if message.channel.id not in self.allowed_channels:
            return
        try:
            text = await self.ask_model(message.content, message.channel.id)
        except Exception as exc:
            text = f"Model error: `{type(exc).__name__}: {exc}`"
        for i in range(0, len(text), MAX_REPLY):
            await message.channel.send(text[i:i + MAX_REPLY])


def create_bot() -> GLMBot:
    bot = GLMBot()

    @bot.command(name="channel")
    @commands.has_guild_permissions(manage_guild=True)
    async def channel(ctx: commands.Context, action: str = "list"):
        action = action.lower()
        if action == "set":
            bot.allowed_channels = {ctx.channel.id}
            _save_channels(bot.allowed_channels)
            await ctx.send(f"AI messages enabled here: <#{ctx.channel.id}>")
        elif action == "add":
            bot.allowed_channels.add(ctx.channel.id)
            _save_channels(bot.allowed_channels)
            await ctx.send(f"Added <#{ctx.channel.id}>")
        elif action == "remove":
            bot.allowed_channels.discard(ctx.channel.id)
            _save_channels(bot.allowed_channels)
            await ctx.send(f"Removed <#{ctx.channel.id}>")
        elif action == "list":
            await ctx.send("Enabled channels: " + (", ".join(f"<#{x}>" for x in sorted(bot.allowed_channels)) or "none"))
        else:
            await ctx.send("Use `!channel set`, `!channel add`, `!channel remove`, or `!channel list`.")

    @bot.command(name="ask")
    async def ask(ctx: commands.Context, *, prompt: str):
        if bot.allowed_channels and ctx.channel.id not in bot.allowed_channels:
            return
        try:
            answer = await bot.ask_model(prompt, ctx.channel.id)
        except Exception as exc:
            answer = f"Model error: `{type(exc).__name__}: {exc}`"
        for i in range(0, len(answer), MAX_REPLY):
            await ctx.send(answer[i:i + MAX_REPLY])

    return bot

[executed on device: Alexanders-MacBook-Pro.local (bd265cb2-62d5-47e5-937a-511d5a03b340)]