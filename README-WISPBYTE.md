[Reading 66 lines from start (total: 66 lines, 0 remaining)]

# Wispbyte server deployment

This folder adds a **server-hosted model + Discord interface** without changing the original DGX-Spark/TensorFold serving stack.

## Architecture

```text
Discord
   -> Wispbyte Python process
      -> /v1/messages (Anthropic-compatible)
         -> Hugging Face Transformers model on Wispbyte CPU
      -> optional MCP tool registry/relay
```

The original repo is tuned for two NVIDIA DGX Sparks. This layer intentionally does not import TensorFold, CUDA, MLX, or DGX-specific code.

## Wispbyte

Choose the **Python** runtime. Wispbyte's Startup page provides the startup command, packages, environment variables and runtime image. citehttps://wispbyte.com/kb/startup-settings

Add the packages from `wispbyte/requirements.txt` in Additional Python Packages, or upload the file and install it from your startup command.

Recommended startup command:

```bash
python -m pip install -r wispbyte/requirements.txt && python -m wispbyte.start
```

Set these environment variables in Wispbyte's Startup page:

```text
HOST=0.0.0.0
PORT=<the public/app port Wispbyte gives the server>
MODEL_ID=<your CPU-compatible model>
MODEL_NAME=<name shown to clients>
APP_API_KEY=<random secret>
DISCORD_TOKEN=<your bot token>
```

Keep secrets in Wispbyte environment variables rather than committing them. citehttps://wispbyte.com/kb/startup-settings

## Discord channel selector

The bot persists channel selections in `data/channels.json`.

- `!channel set` — only this channel
- `!channel add` — add this channel
- `!channel remove` — remove this channel
- `!channel list` — show enabled channels
- `!ask <message>` — explicitly ask the model

Normal messages are answered only in enabled channels.

## Anthropic-compatible endpoint

`POST /v1/messages` accepts the basic Anthropic Messages request shape, including `messages`, `max_tokens`, `temperature`, `tools`, and `stream`.

`GET /v1/models` reports the configured local model name.

This is the same architectural trick that makes Claude Code Local useful: the outer client speaks Anthropic's format while the local model runtime is replaceable. The upstream project documents its own Python server as an Anthropic Messages API server and its tool-call translation layer. citehttps://github.com/nicedreamzapp/claude-code-local

## MCP

The API already has a small tool registry (`/mcp/register`, `/mcp/register/{name}`, `/v1/tools`) so an eventual PC-side MCP connector can register its tool schemas without changing the model API.

Do **not** expose arbitrary shell/file tools until you have a separate authenticated MCP agent and an allowlist. The server should only receive the tools you intentionally register.

[executed on device: Alexanders-MacBook-Pro.local (bd265cb2-62d5-47e5-937a-511d5a03b340)]