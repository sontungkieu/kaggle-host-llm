# Kaggle Worker Runbook

This file records the practical setup notes, failures, fixes, and operating commands learned while bringing the gateway, Kaggle LLM workers, PaddleOCR workers, and DeepSeek OCR2 workers online.

## Operating Model

- The always-on root node runs FastAPI/Uvicorn locally on port `8000`.
- Cloudflare Tunnel exposes the root node through a public HTTPS hostname.
- Kaggle notebooks are outbound-only workers. They connect to `GATEWAY_WS_URL` over WebSocket and register metadata with `/workers/connect`.
- Workers are best-effort. Kaggle may reclaim sessions, hit GPU limits, lose network, or stop kernels. Treat workers as disposable.
- Do not use this as a quota-bypass cluster. Keep old kernels cleaned up and respect Kaggle runtime limits.

## Required Local Files

Keep secrets out of git:

- `.secrets/.env`: gateway tokens, tunnel token, model config.
- `.secrets/all-kaggle.json`: local copy of Kaggle account credentials, or `/home/tung/all-kaggle.json` as the sensitive source.
- `.secrets/groq_key.env`: optional Groq keys.
- `kaggle.log`: local JSONL event log, ignored by git.
- `data/gateway.log`: local Uvicorn log, ignored by git.

Minimum `.secrets/.env` values:

```bash
GATEWAY_API_KEY=<random-api-key>
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

For local wrapper waits, using the public hostname can fail with Cloudflare Access/security rules or NAT loopback. Prefer this flag when pushing from the root machine:

```bash
--gateway-health-url http://127.0.0.1:8000/health
```

The notebook still receives the public `GATEWAY_WS_URL`; only the local wait loop uses the local health URL.

## Start Root Gateway

Foreground:

```bash
uv run uvicorn kaggle_host_llm.app:app --host 0.0.0.0 --port 8000
```

Detached shell:

```bash
mkdir -p data
setsid -f bash -c 'cd /home/tung/kaggle-host-llm && uv run uvicorn kaggle_host_llm.app:app --host 0.0.0.0 --port 8000 >> data/gateway.log 2>&1'
```

Check:

```bash
curl http://127.0.0.1:8000/health
```

If restarting the gateway, workers will disconnect and then reconnect through their notebook retry loops. Wait until `/health` shows active workers again.

## Start Cloudflare Tunnel

Named tunnel:

```bash
uv run python scripts/start_named_tunnel.py
```

Quick tunnel:

```bash
uv run python scripts/start_quick_tunnel.py
```

Lessons:

- Quick tunnels have no stable uptime and can return transient `500` errors during creation.
- Named tunnel is the right path for a stable root hostname.
- Only the root node needs a public route. Kaggle workers do not need public IPs or ngrok.
- If the tunnel hostname changes, existing notebooks that were staged with the old `GATEWAY_WS_URL` cannot discover the new hostname. Re-push workers or delete old kernels and push fresh workers.

## Push OCR Workers

PaddleOCR PP-StructureV3:

```bash
uv run python scripts/push_kaggle_ocr_worker.py kieutung \
  --backend paddleocr-ppstructurev3 \
  --wait-active \
  --gateway-health-url http://127.0.0.1:8000/health
```

DeepSeek OCR2:

```bash
uv run python scripts/push_kaggle_ocr_worker.py kieutung \
  --backend deepseek-ocr2 \
  --served-model deepseek-ocr2 \
  --wait-active \
  --gateway-health-url http://127.0.0.1:8000/health
```

Current working DeepSeek OCR2 fixes:

- Default `DEEPSEEK_OCR_DTYPE=float16` to fit T4 memory.
- Patch `torch.Tensor.masked_scatter_` in the notebook to cast image embeddings to the destination dtype/device before scatter. This avoids the `Half` vs `Float` runtime error from the model's remote inference path.
- Read `*.mmd` output files because DeepSeek writes OCR output to `result.mmd`; reading only `*.md`/`*.txt` returns `"None"`.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before model load to reduce fragmentation risk.

DeepSeek OCR2 failed paths:

- `float16` without scatter patch: `masked_scatter_: expected self and source to have same dtypes but got Half and Float`.
- `float32`: avoided dtype mismatch but exceeded T4 memory.
- Reading only `*.md`, `*.txt`, `*.html`, `*.json`: inference finished, but gateway response was `"None"` because the useful file was `result.mmd`.

PaddleOCR lessons:

- Install `paddlex[ocr]`; plain PaddleOCR dependencies are insufficient for PP-StructureV3.
- Avoid mixing incompatible torch/Paddle GPU libraries. A previous install path produced `ncclCommShrink`/`libtorch_cuda.so` symbol errors.
- Use Paddle CUDA APIs for GPU detection after torch is removed from that notebook path.
- `PADDLEOCR_DEVICE=auto` maps T4x2 to `gpu:0,1`.
- Kaggle notebooks already run inside an event loop, so the final worker cell should use `await worker_loop()`, not `asyncio.run(worker_loop())`.

## Push LLM Worker

vLLM worker:

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

LLM worker lessons:

- Prefer vLLM on T4x2 with AWQ/GPTQ checkpoints.
- Keep `VLLM_MAX_MODEL_LEN` conservative, for example `4096`, to reduce OOM risk.
- `MAX_WORKER_JOBS=auto` registers capacity based on detected GPU count. For T4x2, that is normally capacity `2`.
- P100 sessions can fail with current Kaggle PyTorch images because the wheel may not include `sm_60`.

## Test Pipelines

Chat:

```bash
uv run python scripts/test_chat_pipeline.py \
  --question "Say OK only." \
  --timeout 650
```

PaddleOCR:

```bash
uv run python scripts/test_ocr_pipeline.py \
  --model paddleocr-ppstructurev3 \
  --image-file /tmp/kaggle_host_ocr_test.png \
  --timeout 900
```

DeepSeek OCR2:

```bash
uv run python scripts/test_ocr_pipeline.py \
  --model deepseek-ocr2 \
  --image-file /tmp/kaggle_host_ocr_test.png \
  --timeout 900
```

Known good DeepSeek sample output:

```text
## Kaggle OCR test

Invoice total: 123.45 USD
```

## Worker Health

Human-readable active worker view:

```bash
uv run python - <<'PY'
import json, urllib.request
health = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8))
print('active_workers', health.get('active_workers'))
for w in health.get('workers', []):
    if w.get('status') == 'active':
        print(w.get('node_id'), w.get('owner'), w.get('model'), w.get('accelerator'), 'jobs=', w.get('current_jobs'), 'job_count=', w.get('job_count'))
PY
```

Terminate a connected worker from root:

```bash
curl -X POST http://127.0.0.1:8000/workers/<node_id>/terminate \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -d '{"reason":"manual maintenance"}'
```

Important: root termination stops the WebSocket worker loop, but it may not immediately free Kaggle GPU session quota. Delete old Kaggle kernels when Kaggle reports maximum GPU session count.

## Cleanup Old Kaggle Kernels

Dry-run list:

```bash
uv run python scripts/manage_kaggle_kernel.py kieutung delete-workers \
  --prefix kaggle-deepseek-ocr2-worker \
  --page-size 50
```

Delete matched kernels:

```bash
uv run python scripts/manage_kaggle_kernel.py kieutung delete-workers \
  --prefix kaggle-deepseek-ocr2-worker \
  --page-size 50 \
  --yes
```

Other useful prefixes:

```bash
--prefix kaggle-paddleocr-worker
--prefix kaggle-qwen-worker
```

Bulk cleanup all accounts from `.secrets/all-kaggle.json`:

```bash
uv run python scripts/manage_kaggle_kernel.py all delete-workers --prefix kaggle-qwen-worker
uv run python scripts/manage_kaggle_kernel.py all delete-workers --prefix kaggle-qwen-worker --yes
```

## Logs

Structured Kaggle log summary:

```bash
uv run python scripts/kaggle_event_log.py
```

Recent raw JSONL events:

```bash
uv run python scripts/kaggle_event_log.py --tail 20
```

Gateway server logs:

```bash
tail -n 120 data/gateway.log
```

Common `kaggle.log` error types:

- `kaggle_save_kernel_400`: Kaggle rejected `SaveKernel`, usually account/session or notebook metadata issue.
- `kaggle_gpu_session_limit`: maximum GPU sessions reached.
- `kaggle_kernel_permission_denied`: wrong slug/private notebook/status lookup mismatch.
- `kaggle_auth_required`: CLI not using the expected temporary `KAGGLE_CONFIG_DIR`.
- `deepseek_ocr_dtype_mismatch`: DeepSeek image embedding scatter dtype mismatch.
- `kaggle_cuda_out_of_memory`: model path exceeded GPU memory.
- `paddleocr_missing_paddlex_ocr_extra`: missing `paddlex[ocr]`.
- `paddleocr_torch_nccl_conflict`: torch/Paddle GPU shared library conflict.
- `kaggle_notebook_asyncio_event_loop`: notebook called `asyncio.run()` inside Kaggle's existing loop.
- `gateway_health_check_error`: local wrapper could not poll health URL.

## Cloudflare And Public IP Lessons

- `whoami`/external IP only tells the public address. It does not prove inbound access works.
- If exposing by public IP, the router must forward WAN TCP `8000` to the Windows LAN IP, and Windows must portproxy to the WSL IP.
- If router WAN IP differs from the public IP, the network is behind double NAT/CGNAT and direct public IP exposure will not work.
- A named Cloudflare Tunnel avoids router forwarding and CGNAT issues.
- Cloudflare Access/security settings can cause `403` on `/health` for local wait loops. Use local health URL for wrapper wait, while keeping public `GATEWAY_WS_URL` for notebooks.

## Current Good State Checklist

After setup, this is the target:

```text
active_workers 2
ocr-deepseek2-... deepseek-ocr2 NvidiaTeslaT4x2 jobs=0
ocr-paddle-... paddleocr-ppstructurev3 NvidiaTeslaT4x2 jobs=0
```

Both commands should return HTTP 200:

```bash
uv run python scripts/test_ocr_pipeline.py --model deepseek-ocr2 --image-file /tmp/kaggle_host_ocr_test.png --timeout 900
uv run python scripts/test_ocr_pipeline.py --model paddleocr-ppstructurev3 --image-file /tmp/kaggle_host_ocr_test.png --timeout 900
```

## Source References

- DeepSeek-OCR remote model code writes `result.mmd` and uses `masked_scatter_` in its image embedding path: <https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/modeling_deepseekocr.py>
- DeepSeek-OCR repository examples use `model.infer(...)` with `base_size`, `image_size`, `crop_mode`, and `save_results`: <https://github.com/deepseek-ai/DeepSeek-OCR>
