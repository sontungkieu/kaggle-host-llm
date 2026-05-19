# Kaggle Host LLM

OpenAI-compatible control plane for best-effort Kaggle notebook LLM workers.

The gateway runs on an always-on host and is exposed through Cloudflare Tunnel. Kaggle notebooks do not need inbound networking: each notebook opens an outbound WebSocket to `/workers/connect`, registers itself, receives jobs, and streams responses back to the gateway.

This project is intentionally best-effort. Kaggle notebooks have session, accelerator, quota, and policy limits, so this is not a 24/24 production cluster design and it must not be used to bypass account/resource restrictions.

Operational notes, troubleshooting history, and current known-good commands are collected in [docs/kaggle_worker_runbook.md](docs/kaggle_worker_runbook.md). For the full sequence from root setup, Kaggle worker push, and calling the API from another machine, see [docs/setup_to_external_client.md](docs/setup_to_external_client.md).

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
GROQ_KEY_FILE=.secrets/groq_key.env
GROQ_BASE_URL=https://api.groq.com/openai/v1
GATEWAY_PUBLIC_HOSTNAME=your-domain.example
GATEWAY_PUBLIC_URL=https://your-domain.example
GATEWAY_WS_URL=wss://your-domain.example/workers/connect
GATEWAY_HEALTH_URL=https://your-domain.example/health
SERVED_MODEL=qwen2.5-9b-quantized
WORKER_BACKEND=vllm
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
HF_TOKEN=
MAX_WORKER_JOBS=auto
KEEPALIVE_LOG_SECONDS=60
KAGGLE_ACCELERATOR=NvidiaTeslaT4
VLLM_MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ
VLLM_SERVED_MODEL=qwen2.5-9b-quantized
VLLM_QUANTIZATION=awq
VLLM_TENSOR_PARALLEL_SIZE=auto
VLLM_MAX_MODEL_LEN=4096
VLLM_GPU_MEMORY_UTILIZATION=0.88
VLLM_DTYPE=auto
OCR_MODEL=paddleocr-ppstructurev3
OCR_CAPACITY=1
PADDLEOCR_DEVICE=auto
PADDLEOCR_LANG=
DEEPSEEK_OCR_MODEL_ID=deepseek-ai/DeepSeek-OCR-2
DEEPSEEK_OCR_PROMPT=<image>\n<|grounding|>Convert the document to markdown.
DEEPSEEK_OCR_BASE_SIZE=1024
DEEPSEEK_OCR_IMAGE_SIZE=768
DEEPSEEK_OCR_CROP_MODE=true
DEEPSEEK_OCR_DTYPE=float16
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
Assistant replies stream into the browser and include a small local timing line with response seconds and approximate completion `tok/s` based on gateway usage when available, or a local estimate otherwise. Image URL and local image attachments are supported for Groq vision routes.

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

`usage.prompt_tokens`, `usage.completion_tokens`, and `usage.total_tokens` use worker-provided counts when available, for example from vLLM. Otherwise the gateway returns rough estimates.

Streaming uses standard server-sent events:

```json
{"model":"qwen2.5-9b-quantized","messages":[{"role":"user","content":"hello"}],"stream":true}
```

### Groq routes

Store Groq API keys in `.secrets/groq_key.env`; the file is ignored by git. The gateway reads this file directly and round-robins requests across all keys it finds:

```bash
GROQ_API_KEYS=gsk_first,gsk_second

# or labeled keys
kieusontung8=gsk_third

# or one key per line
gsk_fourth
```

Any user message that starts with `groq:<model>` bypasses Kaggle workers and is sent to Groq's OpenAI-compatible chat completions API. The prefix is stripped before forwarding:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -d '{
    "model": "qwen2.5-9b-quantized",
    "messages": [
      {"role": "user", "content": "groq:llama-3.3-70b-versatile Say OK only."}
    ],
    "stream": true
  }'
```

For images, use a Groq vision model and OpenAI content parts:

```json
{
  "model": "qwen2.5-9b-quantized",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "groq:meta-llama/llama-4-scout-17b-16e-instruct Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
      ]
    }
  ],
  "stream": true
}
```

Image requests without a `groq:<vision-model>` prefix are rejected with HTTP 400 because the current Kaggle worker path is text-only.

### OCR

OCR uses the same root gateway and outbound Kaggle worker WebSocket model as chat. Start one or more OCR notebooks, then call:

```bash
curl http://localhost:8000/v1/ocr \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -d '{
    "model": "paddleocr-ppstructurev3",
    "image_url": "https://example.com/document.png",
    "return_format": "markdown"
  }'
```

You can also send inline base64:

```json
{
  "model": "paddleocr-ppstructurev3",
  "image_base64": "iVBORw0KGgo...",
  "filename": "document.png",
  "mime_type": "image/png",
  "return_format": "all"
}
```

Supported inputs are exactly one of `image_url`, `image_base64`, `document_url`, or `document_base64`. URLs can be `http(s)` URLs or `data:` URLs. The response shape is:

```json
{
  "object": "ocr.result",
  "model": "paddleocr-ppstructurev3",
  "text": "...",
  "markdown": "...",
  "pages": [],
  "data": {},
  "metadata": {}
}
```

Use `return_format=text` or `return_format=markdown` for compact API responses.
Use `return_format=all` only when you need raw OCR geometry, recognition scores,
and backend-specific JSON because those payloads can be large over Cloudflare.

Initial worker models:

- `paddleocr-ppstructurev3`: PaddleOCR PP-StructureV3 document parsing; returns Markdown plus page JSON when available.
- `deepseek-ocr2`: DeepSeek-OCR-2 template; uses the model's `infer()` path to convert an image to Markdown.

Quick OCR pipeline test:

```bash
uv run python scripts/test_ocr_pipeline.py \
  --model paddleocr-ppstructurev3 \
  --image-file ./document.png
```

When testing through Cloudflare, point the client at the public hostname:

```bash
uv run python scripts/test_ocr_pipeline.py \
  --base-url https://hostllm.ccat.io.vn \
  --model deepseek-ocr2 \
  --image-file ./document.png \
  --timeout 900
```

The test clients send a `User-Agent` header because Cloudflare Browser Integrity
Check/Bot protections can return `403` with `error code: 1010` for bare Python
HTTP clients. If other API clients hit `1010`, add a normal `User-Agent` header
or create a Cloudflare WAF skip rule for `hostllm.ccat.io.vn` paths such as
`/v1/*`.

## Kaggle Worker

The notebook template is [notebooks/kaggle_qwen_worker.ipynb](notebooks/kaggle_qwen_worker.ipynb). Set these variables in the Kaggle notebook environment or edit the setup cell:

```bash
GATEWAY_WS_URL=wss://your-domain.example/workers/connect
WORKER_TOKEN=<WORKER_SHARED_TOKEN>
SERVED_MODEL=qwen2.5-9b-quantized
WORKER_BACKEND=vllm
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
HF_TOKEN=
MAX_WORKER_JOBS=auto
KEEPALIVE_LOG_SECONDS=60
KAGGLE_ACCELERATOR=NvidiaTeslaT4
VLLM_MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ
VLLM_SERVED_MODEL=qwen2.5-9b-quantized
VLLM_QUANTIZATION=awq
VLLM_TENSOR_PARALLEL_SIZE=auto
VLLM_MAX_MODEL_LEN=4096
VLLM_GPU_MEMORY_UTILIZATION=0.88
```

`SERVED_MODEL` is the name clients request through `/v1/chat/completions`. `WORKER_BACKEND=vllm` starts a local vLLM OpenAI-compatible server inside the notebook and forwards gateway jobs to `http://127.0.0.1:8001/v1/chat/completions`; this keeps vLLM private inside Kaggle and preserves the outbound WebSocket architecture. For vLLM on T4x2, prefer AWQ/GPTQ checkpoints and keep `VLLM_MAX_MODEL_LEN` conservative, for example `4096`, to avoid OOM. `WORKER_BACKEND=transformers` remains available only as an explicit legacy backend; it is no longer the default.

`MODEL_ID` is used only by the legacy transformers backend. `VLLM_MODEL_ID` is the model used by the vLLM backend. `HF_TOKEN` is optional but helps with Hugging Face rate limits. `MAX_WORKER_JOBS=auto` registers one concurrent slot per detected GPU for vLLM, so a T4x2 vLLM notebook registers capacity 2. The worker loop accepts multiple jobs concurrently up to that capacity and streams responses back over the single WebSocket.
The wrapper pushes with `--accelerator NvidiaTeslaT4` by default. P100 can fail with current Kaggle PyTorch images because that build does not support CUDA `sm_60`.

Create a git-ignored staging folder for Kaggle:

```bash
uv run python scripts/stage_kaggle_worker.py --owner <kaggle-username>
```

For one-command staging + push with credentials selected from `.secrets/all-kaggle.json`:

```bash
uv run python scripts/push_kaggle_worker.py <kaggle-username>
```

To push a vLLM worker using Qwen2.5 7B AWQ on T4x2:

```bash
uv run python scripts/push_kaggle_worker.py <kaggle-username> \
  --worker-backend vllm \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --vllm-model-id Qwen/Qwen2.5-7B-Instruct-AWQ \
  --vllm-quantization awq \
  --vllm-max-model-len 4096 \
  --wait-active
```

The wrapper creates a temporary `KAGGLE_CONFIG_DIR`, writes that account's `kaggle.json` outside the repo, runs `kaggle kernels push`, checks `kaggle kernels status`, and removes the temporary credential directory when done. The Kaggle notebook itself contains a long-running reconnect loop; the local wrapper does not need to stay attached for the notebook to keep running.

The gateway and Kaggle wrapper commands append structured local events to `kaggle.log`. The file is JSONL, ignored by git, and contains timestamp, account, backend/model, node id or kernel id, command/runtime phase, error type, message, and `occurrence` count for repeated warnings/errors. Runtime events include worker registration, disconnects, heartbeat timeouts, root termination requests, send failures, and worker job errors. To change the path:

```bash
KAGGLE_LOG_PATH=data/logs/kaggle.log uv run python scripts/push_kaggle_worker.py <kaggle-username>
```

To inspect recent events or grouped counts:

```bash
uv run python scripts/kaggle_event_log.py --tail 20
uv run python scripts/kaggle_event_log.py
```

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

## Kaggle OCR Workers

PaddleOCR PP-StructureV3 worker:

```bash
uv run python scripts/push_kaggle_ocr_worker.py <kaggle-username> \
  --backend paddleocr-ppstructurev3 \
  --wait-active
```

DeepSeek-OCR-2 worker:

```bash
uv run python scripts/push_kaggle_ocr_worker.py <kaggle-username> \
  --backend deepseek-ocr2 \
  --served-model deepseek-ocr2 \
  --wait-active
```

Dry-run staging only:

```bash
uv run python scripts/push_kaggle_ocr_worker.py <kaggle-username> \
  --backend paddleocr-ppstructurev3 \
  --dry-run
```

The OCR notebooks are [notebooks/kaggle_paddleocr_worker.ipynb](notebooks/kaggle_paddleocr_worker.ipynb) and [notebooks/kaggle_deepseek_ocr2_worker.ipynb](notebooks/kaggle_deepseek_ocr2_worker.ipynb). Both register via `/workers/connect` and listen for `ocr_job` envelopes. Use `OCR_MODEL` to override the registered model name, `OCR_CAPACITY` for concurrency, and `KAGGLE_ACCELERATOR=NvidiaTeslaT4` for T4-backed Kaggle sessions.

For PaddleOCR, `PADDLEOCR_DEVICE=auto` maps T4x2 to `gpu:0,1`; set it explicitly if Paddle reports device issues. For DeepSeek-OCR-2 on T4, the template defaults to `DEEPSEEK_OCR_DTYPE=float16` and applies a small dtype-safe scatter patch because the model's remote inference path can produce mixed float32/float16 image embeddings.

The PaddleOCR worker normalizes `PPStructureV3.concatenate_markdown_pages(...)`
output because Paddle may return a dict containing `markdown_texts` rather than
a plain string. Re-push the PaddleOCR notebook after updating the template to
get cleaner `text`/`markdown` output on Kaggle.

If Kaggle reports `Maximum batch GPU session count of 2 reached`, that account already has two GPU sessions. Keep the current workers if they are healthy, push from another account, or delete old kernels first:

```bash
uv run python scripts/manage_kaggle_kernel.py <kaggle-username> delete-workers \
  --prefix kaggle-paddleocr-worker \
  --page-size 50

uv run python scripts/manage_kaggle_kernel.py <kaggle-username> delete-workers \
  --prefix kaggle-paddleocr-worker \
  --page-size 50 \
  --yes
```

Use `--prefix kaggle-deepseek-ocr2-worker` for DeepSeek OCR2 cleanup.

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
