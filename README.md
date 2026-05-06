# Kaggle Host LLM

OpenAI-compatible control plane for best-effort Kaggle notebook LLM workers.

The gateway runs on an always-on host and is exposed through Cloudflare Tunnel. Kaggle notebooks do not need inbound networking: each notebook opens an outbound WebSocket to `/workers/connect`, registers itself, receives jobs, and streams responses back to the gateway.

This project is intentionally best-effort. Kaggle notebooks have session, accelerator, quota, and policy limits, so this is not a 24/24 production cluster design and it must not be used to bypass account/resource restrictions.

## Install

```bash
uv sync --extra test
```

Run the gateway:

```bash
uv run uvicorn kaggle_host_llm.app:app --host 0.0.0.0 --port 8000
```

Useful environment variables:

```bash
HOST=0.0.0.0
PORT=8000
DATABASE_PATH=data/gateway.sqlite3
LIVE_NODES_PATH=data/live_workers.json
GATEWAY_API_KEY=change-me
WORKER_SHARED_TOKEN=change-me-worker-token
HEARTBEAT_TIMEOUT_SECONDS=45
ALIVE_CHECK_INTERVAL_SECONDS=300
JOB_TIMEOUT_SECONDS=600
GATEWAY_PUBLIC_HOSTNAME=your-domain.example
GATEWAY_PUBLIC_URL=https://your-domain.example
GATEWAY_WS_URL=wss://your-domain.example/workers/connect
GATEWAY_HEALTH_URL=https://your-domain.example/health
SERVED_MODEL=qwen2.5-9b-quantized
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
HF_TOKEN=
MAX_WORKER_JOBS=auto
KEEPALIVE_LOG_SECONDS=60
KAGGLE_ACCELERATOR=NvidiaTeslaT4
```

The app automatically loads `.secrets/.env` and `.env` when present. Keep real secrets in `.secrets/.env`; the `.secrets/` folder is ignored by git.

If `GATEWAY_API_KEY` is set, clients must send:

```text
Authorization: Bearer <GATEWAY_API_KEY>
```

If `WORKER_SHARED_TOKEN` is set, workers must connect to:

```text
wss://your-domain.example/workers/connect?token=<WORKER_SHARED_TOKEN>
```

## Cloudflare Tunnel

For stable use, configure a named Cloudflare Tunnel and route a hostname to `http://127.0.0.1:8000`. Store the tunnel token in `.secrets/.env`:

```bash
TOKEN_CUA_TUNNEL=<cloudflare-named-tunnel-token>
GATEWAY_PUBLIC_HOSTNAME=hostllm.ccat.io.vn
GATEWAY_PUBLIC_URL=https://hostllm.ccat.io.vn
GATEWAY_WS_URL=wss://hostllm.ccat.io.vn/workers/connect
GATEWAY_HEALTH_URL=https://hostllm.ccat.io.vn/health
```

Start the named tunnel:

```bash
uv run python scripts/start_named_tunnel.py
```

The wrapper reads `TOKEN_CUA_TUNNEL` from `.secrets/.env` and passes it to `cloudflared` through the `TUNNEL_TOKEN` environment variable.

For quick experiments, create a temporary tunnel from the always-on gateway host to the local Uvicorn port:

```bash
uv run python scripts/start_quick_tunnel.py
```

This wrapper runs `cloudflared tunnel --edge-ip-version 4 --protocol http2 --url http://127.0.0.1:8000`, retries transient Quick Tunnel failures, and writes the generated tunnel values into `.secrets/.env`:

```bash
GATEWAY_PUBLIC_HOSTNAME=<generated>.trycloudflare.com
GATEWAY_PUBLIC_URL=https://<generated>.trycloudflare.com
GATEWAY_WS_URL=wss://<generated>.trycloudflare.com/workers/connect
GATEWAY_HEALTH_URL=https://<generated>.trycloudflare.com/health
```

When a Quick Tunnel is recreated, rerun `scripts/start_quick_tunnel.py` first, then push or repush Kaggle workers so their embedded `GATEWAY_WS_URL` points at the new hostname. Already-running notebooks that only know the old dead tunnel cannot be contacted unless they were built with a separate stable bootstrap URL.

Quick Tunnels can fail or change URL when restarted. If you expose the gateway directly through a public IP and plain HTTP, set `GATEWAY_WS_URL=ws://PUBLIC_IP:8000/workers/connect` instead of `wss://...`.

## API

Health:

```bash
curl http://localhost:8000/health
```

Basic browser chat UI:

```text
http://localhost:8000/chat
https://hostllm.ccat.io.vn/chat
```

If `GATEWAY_API_KEY` is configured, paste that value into the Gateway API key field in the chat sidebar. The browser stores this setting in local storage only.

The chat UI keeps conversation context in browser local storage and sends the full visible message history with each `/v1/chat/completions` request. Use **Clear** to start a fresh conversation.

Live worker file and root control:

```bash
curl http://localhost:8000/workers/live \
  -H "Authorization: Bearer $GATEWAY_API_KEY"

curl http://localhost:8000/workers/uptime \
  -H "Authorization: Bearer $GATEWAY_API_KEY"

curl -X POST http://localhost:8000/workers/<node_id>/terminate \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -d '{"reason":"manual maintenance"}'
```

The root writes currently alive workers to `data/live_workers.json` by default. It refreshes this file on `/health`, `/workers/live`, terminate requests, and every 300 seconds via the background alive check loop. `/workers/uptime` summarizes uptime, active capacity, current jobs, and total jobs by Kaggle account.

OpenAI-compatible chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -d '{
    "model": "qwen2.5-9b-quantized",
    "messages": [{"role": "user", "content": "Xin chao"}],
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 128
  }'
```

Quick pipeline test:

```bash
uv run python scripts/test_chat_pipeline.py --question "Say OK only."
```

`usage.prompt_tokens`, `usage.completion_tokens`, and `usage.total_tokens` are gateway-side estimates. They are useful for rough monitoring, but they are not exact tokenizer counts from the Kaggle worker.

Streaming uses standard server-sent events:

```json
{"model":"qwen2.5-9b-quantized","messages":[{"role":"user","content":"hello"}],"stream":true}
```

## Kaggle Worker

The notebook template is [notebooks/kaggle_qwen_worker.ipynb](notebooks/kaggle_qwen_worker.ipynb). Set these variables in the Kaggle notebook environment or edit the setup cell:

```bash
GATEWAY_WS_URL=wss://your-domain.example/workers/connect
WORKER_TOKEN=<WORKER_SHARED_TOKEN>
SERVED_MODEL=qwen2.5-9b-quantized
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
HF_TOKEN=
MAX_WORKER_JOBS=auto
KEEPALIVE_LOG_SECONDS=60
KAGGLE_ACCELERATOR=NvidiaTeslaT4
```

`SERVED_MODEL` is the name clients request through `/v1/chat/completions`. `MODEL_ID` is the actual Hugging Face or Kaggle-accessible model path loaded by `transformers`. By default the worker loads Qwen2.5 Instruct and applies 4-bit `bitsandbytes` quantization; AWQ checkpoints may require extra packages such as `gptqmodel`. `HF_TOKEN` is optional but helps with Hugging Face rate limits. `MAX_WORKER_JOBS=auto` advertises one concurrent slot per detected GPU, so a T4x2 notebook registers capacity 2. The last notebook cell runs forever with a WebSocket reconnect loop and periodic keepalive logs so the Kaggle run remains active.
The wrapper pushes with `--accelerator NvidiaTeslaT4` by default. P100 can fail with current Kaggle PyTorch images because that build does not support CUDA `sm_60`.

Create a git-ignored staging folder for Kaggle:

```bash
uv run python scripts/stage_kaggle_worker.py --owner <kaggle-username>
```

For one-command staging + push with credentials selected from `.secrets/all-kaggle.json`:

```bash
uv run python scripts/push_kaggle_worker.py <kaggle-username>
```

The wrapper creates a temporary `KAGGLE_CONFIG_DIR`, writes that account's `kaggle.json` outside the repo, runs `kaggle kernels push`, checks `kaggle kernels status`, and removes the temporary credential directory when done. The Kaggle notebook itself contains a long-running reconnect loop; the local wrapper does not need to stay attached for the notebook to keep running.

To keep the local command open until the gateway sees the worker:

```bash
uv run python scripts/push_kaggle_worker.py <kaggle-username> --wait-active
```

If `GATEWAY_WS_URL` uses a public IP, set `GATEWAY_HEALTH_URL=http://127.0.0.1:8000/health` so local polling does not depend on router NAT loopback.

To inspect or delete old worker kernels with the same `.secrets/all-kaggle.json` credentials:

```bash
uv run python scripts/manage_kaggle_kernel.py <kaggle-username> list
uv run python scripts/manage_kaggle_kernel.py <kaggle-username> logs <owner/slug>
uv run python scripts/manage_kaggle_kernel.py <kaggle-username> delete <owner/slug> -y
```

To clean up all staged worker kernels after rotating a temporary tunnel:

```bash
uv run python scripts/manage_kaggle_kernel.py all delete-workers
uv run python scripts/manage_kaggle_kernel.py all delete-workers --yes
```

The first command is a dry run and lists every kernel whose slug starts with `kaggle-qwen-worker` for all Kaggle accounts in `.secrets/all-kaggle.json`. The second command deletes those matched kernels. To clean one account only:

```bash
uv run python scripts/manage_kaggle_kernel.py <kaggle-username> delete-workers --yes
```

Delete failed/running old worker kernels if Kaggle reports the maximum GPU session count has been reached.

For direct public-IP exposure, verify local state with:

```bash
uv run python scripts/check_public_gateway.py
```

The router must forward `WAN TCP 8000` to the Windows LAN IP, not the WSL IP. If the router WAN IP does not equal the public IP shown by the script, the connection is behind double NAT/CGNAT and direct IP exposure will not work.

Dry-run without calling Kaggle:

```bash
uv run python scripts/push_kaggle_worker.py <kaggle-username> --dry-run
```

The staging folder contains the notebook, `kernel-metadata.json`, and optional `worker_config.json` for runtime worker settings. Keep Kaggle API tokens outside the repo; `.secrets/all-kaggle.json` is ignored by git.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra test pytest
```

The tests use fake WebSocket workers and do not contact Kaggle.

## Repo Maintenance Notes

- PDF docs: N/A, no `pdf/` tree exists in this repo yet.
- Mindmap: N/A, no mindmap files exist.
- `milestones.md`: N/A, file does not exist.
- `plan.md` / `plan_next...`: N/A, no planning files exist.
