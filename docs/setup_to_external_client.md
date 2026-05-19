# Setup To External Client Guide

This guide is the shortest end-to-end path from a fresh root machine to a
public API that can be called from another machine. It assumes the public
hostname is `https://hostllm.ccat.io.vn`; replace it if the Cloudflare route
changes.

## 1. Root Machine Setup

Run these commands on the always-on root machine, not on Kaggle:

```bash
cd /home/tung/kaggle-host-llm
uv sync --extra test
mkdir -p .secrets data
```

Create or update `.secrets/.env`:

```bash
GATEWAY_API_KEY=<random-client-api-key>
WORKER_SHARED_TOKEN=<random-worker-token>
TOKEN_CUA_TUNNEL=<cloudflare-named-tunnel-token>

GATEWAY_PUBLIC_HOSTNAME=hostllm.ccat.io.vn
GATEWAY_PUBLIC_URL=https://hostllm.ccat.io.vn
GATEWAY_WS_URL=wss://hostllm.ccat.io.vn/workers/connect
GATEWAY_HEALTH_URL=https://hostllm.ccat.io.vn/health

KAGGLE_ACCELERATOR=NvidiaTeslaT4
OCR_CAPACITY=1
PADDLEOCR_DEVICE=auto
DEEPSEEK_OCR_DTYPE=float16
```

Do not commit `.secrets/.env`. `GATEWAY_API_KEY` is the key external clients
use. `WORKER_SHARED_TOKEN` is only for Kaggle workers connecting to
`/workers/connect`.

## 2. Cloudflare Named Tunnel

In Cloudflare Zero Trust / Tunnels, create a named tunnel and add a Published
Application route:

```text
Hostname: hostllm.ccat.io.vn
Service URL: http://127.0.0.1:8000
```

The DNS record should point at the named tunnel, for example:

```text
hostllm.ccat.io.vn CNAME <tunnel-id>.cfargotunnel.com
```

Start the gateway and tunnel on the root machine:

```bash
setsid -f bash -c 'cd /home/tung/kaggle-host-llm && uv run uvicorn kaggle_host_llm.app:app --host 0.0.0.0 --port 8000 >> data/gateway.log 2>&1'
setsid -f bash -c 'cd /home/tung/kaggle-host-llm && uv run python scripts/start_named_tunnel.py >> data/cloudflared.log 2>&1'
```

Check both local and public health:

```bash
curl http://127.0.0.1:8000/health
curl https://hostllm.ccat.io.vn/health
```

If the public endpoint returns Cloudflare `1033`, the named tunnel connector is
not running or is connected to the wrong tunnel token. Check:

```bash
tail -n 80 data/cloudflared.log
pgrep -af 'cloudflared|start_named_tunnel'
```

The log should show the expected tunnel ID, the hostname
`hostllm.ccat.io.vn`, and `Registered tunnel connection`.

## 3. Kaggle Credential Setup

The push scripts use Kaggle legacy `kaggle.json` credentials from a local
account bundle. The sensitive source path is:

```text
/home/tung/all-kaggle.json
```

The scripts materialize a temporary `KAGGLE_CONFIG_DIR` per account and do not
require manually copying `kaggle.json` into the repo. Do not print, copy, or
commit Kaggle API keys.

## 4. Push Kaggle Workers

Push a PaddleOCR worker:

```bash
uv run python scripts/push_kaggle_ocr_worker.py kieutung \
  --backend paddleocr-ppstructurev3 \
  --wait-active \
  --gateway-health-url http://127.0.0.1:8000/health
```

Push a DeepSeek OCR2 worker:

```bash
uv run python scripts/push_kaggle_ocr_worker.py kieutung \
  --backend deepseek-ocr2 \
  --served-model deepseek-ocr2 \
  --wait-active \
  --gateway-health-url http://127.0.0.1:8000/health
```

Push a vLLM chat worker:

```bash
uv run python scripts/push_kaggle_worker.py kieutung \
  --worker-backend vllm \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --vllm-model-id Qwen/Qwen2.5-7B-Instruct-AWQ \
  --vllm-quantization awq \
  --vllm-max-model-len 4096 \
  --wait-active \
  --gateway-health-url http://127.0.0.1:8000/health
```

Use another Kaggle owner instead of `kieutung` when that account already has
the maximum GPU sessions running.

## 5. Check Worker Status

Gateway worker registry:

```bash
curl -sS https://hostllm.ccat.io.vn/health
```

Human-readable local view:

```bash
uv run python - <<'PY'
import json, urllib.request
health = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8))
print('active_workers', health.get('active_workers'))
for w in health.get('workers', []):
    if w.get('status') == 'active':
        print(w.get('node_id'), w.get('owner'), w.get('model'), w.get('accelerator'), 'jobs=', w.get('current_jobs'))
PY
```

Kaggle kernel status:

```bash
uv run python scripts/manage_kaggle_kernel.py kieutung status kieutung/<kernel-slug>
uv run python scripts/manage_kaggle_kernel.py kieutung logs kieutung/<kernel-slug>
```

List likely worker kernels:

```bash
uv run python scripts/manage_kaggle_kernel.py kieutung list --search kaggle-paddleocr-worker
uv run python scripts/manage_kaggle_kernel.py kieutung list --search kaggle-deepseek-ocr2-worker
uv run python scripts/manage_kaggle_kernel.py kieutung list --search kaggle-qwen-worker
```

Delete old worker kernels after reviewing the dry run:

```bash
uv run python scripts/manage_kaggle_kernel.py kieutung delete-workers --prefix kaggle-paddleocr-worker
uv run python scripts/manage_kaggle_kernel.py kieutung delete-workers --prefix kaggle-paddleocr-worker --yes
```

Use `owner=all` only for intentional bulk cleanup:

```bash
uv run python scripts/manage_kaggle_kernel.py all delete-workers --prefix kaggle-qwen-worker
```

## 6. Test From The Root Machine

Chat:

```bash
uv run python scripts/test_chat_pipeline.py \
  --base-url https://hostllm.ccat.io.vn \
  --question "Say OK only." \
  --timeout 650
```

PaddleOCR:

```bash
uv run python scripts/test_ocr_pipeline.py \
  --base-url https://hostllm.ccat.io.vn \
  --model paddleocr-ppstructurev3 \
  --image-file ./bill.png \
  --return-format text \
  --timeout 900
```

DeepSeek OCR2:

```bash
uv run python scripts/test_ocr_pipeline.py \
  --base-url https://hostllm.ccat.io.vn \
  --model deepseek-ocr2 \
  --image-file ./bill.png \
  --return-format text \
  --timeout 900
```

The test scripts send `User-Agent: kaggle-host-llm-test-client/1.0`. This is
important because Cloudflare Browser Integrity Check/Bot protections can return
`403` with `error code: 1010` for bare Python clients.

## 7. Call From Another Machine Without This Repo

External machines do not need the repo. They only need:

```text
Base URL: https://hostllm.ccat.io.vn
API key: GATEWAY_API_KEY from .secrets/.env
Required headers: Authorization, Content-Type, User-Agent
```

Chat with `curl`:

```bash
export GATEWAY_API_KEY='your-api-key'

curl https://hostllm.ccat.io.vn/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: kaggle-host-llm-client/1.0" \
  -d '{
    "model": "groq:llama-3.1-8b-instant",
    "messages": [
      {"role": "user", "content": "Say hello in Vietnamese"}
    ],
    "temperature": 0.2,
    "max_tokens": 200
  }'
```

PaddleOCR with a local image:

```bash
export GATEWAY_API_KEY='your-api-key'
IMG_B64=$(base64 -w 0 bill.png)

curl https://hostllm.ccat.io.vn/v1/ocr \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: kaggle-host-llm-client/1.0" \
  -d "{
    \"model\": \"paddleocr-ppstructurev3\",
    \"image_base64\": \"$IMG_B64\",
    \"filename\": \"bill.png\",
    \"mime_type\": \"image/png\",
    \"return_format\": \"text\"
  }"
```

DeepSeek OCR2 with a local image:

```bash
export GATEWAY_API_KEY='your-api-key'
IMG_B64=$(base64 -w 0 bill.png)

curl https://hostllm.ccat.io.vn/v1/ocr \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: kaggle-host-llm-client/1.0" \
  -d "{
    \"model\": \"deepseek-ocr2\",
    \"image_base64\": \"$IMG_B64\",
    \"filename\": \"bill.png\",
    \"mime_type\": \"image/png\",
    \"return_format\": \"text\"
  }"
```

On macOS, use this base64 command instead:

```bash
IMG_B64=$(base64 < bill.png | tr -d '\n')
```

Python client without repo:

```python
import base64
import requests

api_key = "your-api-key"

with open("bill.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://hostllm.ccat.io.vn/v1/ocr",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-host-llm-client/1.0",
    },
    json={
        "model": "paddleocr-ppstructurev3",
        "image_base64": image_base64,
        "filename": "bill.png",
        "mime_type": "image/png",
        "return_format": "text",
    },
    timeout=900,
)

print(response.status_code)
print(response.text)
```

Prefer `image_url` over `image_base64` when the image is already hosted. Base64
adds upload overhead and can dominate total latency through Cloudflare and the
Kaggle WebSocket path.

## 8. Troubleshooting Checklist

- `403 error code: 1010`: add a normal `User-Agent` header, or create a
  Cloudflare WAF skip rule for `hostllm.ccat.io.vn` and `/v1/*`.
- `530 error code: 1033`: Cloudflare route has no active tunnel connector.
  Restart `scripts/start_named_tunnel.py` and check `data/cloudflared.log`.
- `active_workers=0`: Kaggle notebooks are not connected. Check notebook logs,
  `GATEWAY_WS_URL`, and `WORKER_SHARED_TOKEN`.
- `Maximum batch GPU session count of 2 reached`: that Kaggle account already
  has two GPU sessions. Stop old sessions or use another account.
- `worker disconnected`: the Kaggle notebook restarted or lost WebSocket during
  the job. Retry after it reconnects.
- Huge OCR response: use `return_format=text` or `return_format=markdown`; use
  `return_format=all` only for raw backend debugging.

## 9. Performance Notes

Observed bill-image timings through `https://hostllm.ccat.io.vn` on T4x2:

```text
paddleocr-ppstructurev3: worker inference about 1.6s, public request about 6.8s
deepseek-ocr2: worker inference about 24s, public request about 33.5s
```

The gap is mostly image upload/base64 overhead, gateway dispatch, Cloudflare
round-trip time, and root-to-Kaggle WebSocket transit.

Speed priorities:

- Use `image_url` or compressed/resized images when possible.
- Use compact `return_format`.
- For receipts, add a future lightweight `paddleocr-text` worker if
  PP-StructureV3 layout/table parsing is unnecessary.
- For DeepSeek OCR2, try lower `DEEPSEEK_OCR_IMAGE_SIZE` such as `640` or
  `512`, and try `DEEPSEEK_OCR_CROP_MODE=false` for simple receipt images.

