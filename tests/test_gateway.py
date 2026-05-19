from __future__ import annotations

import asyncio
import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from kaggle_host_llm.app import create_app
from kaggle_host_llm.groq_client import GroqKeyPool, load_groq_keys
from kaggle_host_llm.kaggle_log import log_kaggle_event, summarize
from kaggle_host_llm.openai_models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from kaggle_host_llm.registry import WorkerRegistry
from kaggle_host_llm.settings import Settings


MODEL = "qwen2.5-9b-quantized"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
OCR_MODEL = "paddleocr-ppstructurev3"


@pytest.fixture(autouse=True)
def isolate_kaggle_log(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_LOG_PATH", str(tmp_path / "kaggle.log"))


class FakeGroqClient:
    def __init__(self) -> None:
        self.routes = []

    def has_keys(self) -> bool:
        return True

    async def complete(self, route):
        self.routes.append(route)
        return ChatCompletionResponse(
            id="chatcmpl-groq-test",
            created=1,
            model=route.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="groq answer")
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )

    async def stream(self, route):
        self.routes.append(route)
        yield (
            'data: {"id":"chatcmpl-groq-test","object":"chat.completion.chunk",'
            '"created":1,"model":"groq-test","choices":[{"index":0,'
            '"delta":{"content":"groq "},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl-groq-test","object":"chat.completion.chunk",'
            '"created":1,"model":"groq-test","choices":[{"index":0,'
            '"delta":{"content":"stream"},"finish_reason":null}]}\n\n'
        )
        yield "data: [DONE]\n\n"


def make_client(tmp_path, **overrides):
    settings = Settings(
        database_path=str(tmp_path / "gateway.sqlite3"),
        live_nodes_path=str(tmp_path / "live_workers.json"),
        heartbeat_timeout_seconds=overrides.get("heartbeat_timeout_seconds", 45.0),
        alive_check_interval_seconds=overrides.get("alive_check_interval_seconds", 300.0),
        job_timeout_seconds=overrides.get("job_timeout_seconds", 5.0),
        api_key=overrides.get("api_key", ""),
        worker_token=overrides.get("worker_token", ""),
    )
    return TestClient(create_app(settings))


def chat_payload(**overrides):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 16,
        "stream": False,
    }
    payload.update(overrides)
    return payload


def register_worker(client: TestClient, *, token: str = ""):
    suffix = f"?token={token}" if token else ""
    websocket = client.websocket_connect(f"/workers/connect{suffix}")
    ws = websocket.__enter__()
    ws.send_json(
        {
            "type": "register",
            "node_id": "node-1",
            "owner": "tester",
            "model": MODEL,
            "accelerator": "NvidiaTeslaT4x2",
            "capacity": 1,
        }
    )
    assert ws.receive_json()["type"] == "registered"
    return websocket, ws


def register_ocr_worker(client: TestClient, *, token: str = ""):
    suffix = f"?token={token}" if token else ""
    websocket = client.websocket_connect(f"/workers/connect{suffix}")
    ws = websocket.__enter__()
    ws.send_json(
        {
            "type": "register",
            "node_id": "ocr-node-1",
            "owner": "tester",
            "model": OCR_MODEL,
            "accelerator": "NvidiaTeslaT4x2",
            "capacity": 1,
        }
    )
    assert ws.receive_json()["type"] == "registered"
    return websocket, ws


def test_openai_request_validation_requires_user_message(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "system", "content": "x"}]},
        )

    assert response.status_code == 422


def test_registry_marks_stale_workers_disconnected(tmp_path):
    registry = WorkerRegistry(str(tmp_path / "registry.sqlite3"), heartbeat_timeout_seconds=0)
    registry.init_db()
    with sqlite3.connect(str(tmp_path / "registry.sqlite3")) as conn:
        conn.execute(
            """
            INSERT INTO worker_nodes (
                node_id, owner, model, accelerator, status, capacity,
                current_jobs, last_heartbeat, connected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("stale-node", "tester", MODEL, "NvidiaTeslaT4x2", "active", 1, 0, 0.0, 0.0),
        )
        conn.commit()

    asyncio.run(registry.cleanup_stale())
    nodes = registry.list_nodes()

    assert nodes[0]["node_id"] == "stale-node"
    assert nodes[0]["status"] == "disconnected"


def test_dispatcher_returns_503_without_worker(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/chat/completions", json=chat_payload())

    assert response.status_code == 503
    assert "no healthy worker" in response.json()["detail"]


def test_streaming_dispatcher_returns_503_without_worker(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/chat/completions", json=chat_payload(stream=True))

    assert response.status_code == 503
    assert "no healthy worker" in response.json()["detail"]


def test_ocr_request_validation_requires_one_input(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/ocr", json={"model": OCR_MODEL})

    assert response.status_code == 422


def test_ocr_returns_503_without_worker(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/ocr",
            json={"model": OCR_MODEL, "image_url": "https://example.com/doc.png"},
        )

    assert response.status_code == 503
    assert "no healthy OCR worker" in response.json()["detail"]


def test_ocr_routes_to_worker_and_returns_markdown(tmp_path):
    with make_client(tmp_path, worker_token="secret") as client:
        websocket_ctx, ws = register_ocr_worker(client, token="secret")
        result = {}

        def call_gateway():
            result["response"] = client.post(
                "/v1/ocr",
                json={
                    "model": OCR_MODEL,
                    "image_base64": "aGVsbG8=",
                    "filename": "doc.png",
                    "return_format": "markdown",
                },
            )

        thread = threading.Thread(target=call_gateway)
        thread.start()
        job = ws.receive_json()
        assert job["type"] == "ocr_job"
        assert job["model"] == OCR_MODEL
        assert job["request"]["image_base64"] == "aGVsbG8="
        ws.send_json(
            {
                "type": "ocr_done",
                "job_id": job["job_id"],
                "result": {
                    "text": "hello",
                    "markdown": "# hello",
                    "pages": [{"index": 0, "markdown": "# hello"}],
                    "data": {"pages": []},
                    "metadata": {"backend": "fake"},
                },
            }
        )
        thread.join(timeout=5)
        websocket_ctx.__exit__(None, None, None)

    assert not thread.is_alive()
    response = result["response"]
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == OCR_MODEL
    assert body["text"] == "hello"
    assert body["markdown"] == "# hello"
    assert body["pages"][0]["index"] == 0
    assert body["metadata"]["backend"] == "fake"


def test_ocr_worker_error_returns_controlled_error(tmp_path):
    with make_client(tmp_path) as client:
        websocket_ctx, ws = register_ocr_worker(client)
        result = {}

        def call_gateway():
            result["response"] = client.post(
                "/v1/ocr",
                json={"model": OCR_MODEL, "image_url": "https://example.com/doc.png"},
            )

        thread = threading.Thread(target=call_gateway)
        thread.start()
        job = ws.receive_json()
        ws.send_json(
            {
                "type": "ocr_error",
                "job_id": job["job_id"],
                "error": "OCR failed",
            }
        )
        thread.join(timeout=5)
        websocket_ctx.__exit__(None, None, None)

    assert not thread.is_alive()
    assert result["response"].status_code == 502
    assert "OCR failed" in result["response"].json()["detail"]


def test_kaggle_log_records_owner_backend_and_occurrence(tmp_path):
    log_file = tmp_path / "kaggle.log"
    log_kaggle_event(
        "observed_kaggle_issue",
        level="error",
        owner="kieutung",
        backend="deepseek-ocr2",
        node_id="ocr-node",
        message="masked_scatter_: expected self and source to have same dtypes",
    )
    record = log_kaggle_event(
        "observed_kaggle_issue",
        level="error",
        owner="kieutung",
        backend="deepseek-ocr2",
        node_id="ocr-node",
        message="masked_scatter_: expected self and source to have same dtypes",
    )

    assert record["occurrence"] == 2
    assert record["error_type"] == "deepseek_ocr_dtype_mismatch"
    assert summarize(log_file) == [
        (
            (
                "kieutung",
                "deepseek-ocr2",
                "observed_kaggle_issue",
                "deepseek_ocr_dtype_mismatch",
            ),
            2,
        )
    ]


def test_groq_key_file_loads_multiple_formats_and_round_robins(tmp_path):
    key_file = tmp_path / "groq_key.env"
    key_file.write_text(
        "\n".join(
            [
                "GROQ_API_KEYS=key-a,key-b",
                "GROQ_API_KEY_3=key-c",
                "personal=key-labeled",
                "key-d",
            ]
        ),
        encoding="utf-8",
    )

    assert load_groq_keys(key_file) == [
        "key-a",
        "key-b",
        "key-c",
        "key-labeled",
        "key-d",
    ]

    async def collect_keys():
        pool = GroqKeyPool(key_file)
        return [await pool.next_key() for _ in range(5)]

    assert asyncio.run(collect_keys()) == [
        "key-a",
        "key-b",
        "key-c",
        "key-labeled",
        "key-d",
    ]


def test_groq_prefix_routes_without_worker_and_strips_prefix(tmp_path):
    with make_client(tmp_path) as client:
        fake_groq = FakeGroqClient()
        client.app.state.dispatcher.groq_client = fake_groq
        response = client.post(
            "/v1/chat/completions",
            json=chat_payload(
                messages=[
                    {
                        "role": "user",
                        "content": f"groq:{GROQ_MODEL} describe this",
                    }
                ]
            ),
        )

    assert response.status_code == 200
    assert response.json()["model"] == GROQ_MODEL
    assert response.json()["choices"][0]["message"]["content"] == "groq answer"
    assert len(fake_groq.routes) == 1
    assert fake_groq.routes[0].model == GROQ_MODEL
    assert fake_groq.routes[0].request.messages[-1].content == "describe this"


def test_groq_vision_content_parts_route_to_groq(tmp_path):
    with make_client(tmp_path) as client:
        fake_groq = FakeGroqClient()
        client.app.state.dispatcher.groq_client = fake_groq
        response = client.post(
            "/v1/chat/completions",
            json=chat_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"groq:{GROQ_MODEL} what is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgo="
                                },
                            },
                        ],
                    }
                ]
            ),
        )

    assert response.status_code == 200
    content = fake_groq.routes[0].request.messages[-1].model_dump(mode="json")["content"]
    assert content[0]["text"] == "what is in this image?"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_input_without_groq_prefix_returns_400(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/chat/completions",
            json=chat_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            },
                        ],
                    }
                ]
            ),
        )

    assert response.status_code == 400
    assert "Groq vision route" in response.json()["detail"]


def test_groq_streaming_route_emits_proxied_chunks(tmp_path):
    with make_client(tmp_path) as client:
        fake_groq = FakeGroqClient()
        client.app.state.dispatcher.groq_client = fake_groq
        response = client.post(
            "/v1/chat/completions",
            json=chat_payload(
                stream=True,
                messages=[{"role": "user", "content": f"groq:{GROQ_MODEL} hello"}],
            ),
        )

    assert response.status_code == 200
    assert "groq " in response.text
    assert "stream" in response.text
    assert "[DONE]" in response.text


def test_chat_page_returns_basic_ui(tmp_path):
    with make_client(tmp_path, api_key="api-key") as client:
        response = client.get("/chat")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Kaggle Host LLM" in response.text
    assert "/v1/chat/completions" in response.text
    assert "tok/s" in response.text
    assert "Image URL" in response.text


def test_non_streaming_completion_routes_to_worker(tmp_path):
    with make_client(tmp_path, worker_token="secret") as client:
        websocket_ctx, ws = register_worker(client, token="secret")
        result = {}

        def call_gateway():
            result["response"] = client.post("/v1/chat/completions", json=chat_payload())

        thread = threading.Thread(target=call_gateway)
        thread.start()
        job = ws.receive_json()
        assert job["type"] == "job"
        assert job["model"] == MODEL
        ws.send_json({"type": "job_done", "job_id": job["job_id"], "content": "hi there"})
        thread.join(timeout=5)
        websocket_ctx.__exit__(None, None, None)

    assert not thread.is_alive()
    response = result["response"]
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_non_streaming_completion_uses_worker_token_usage(tmp_path):
    with make_client(tmp_path) as client:
        websocket_ctx, ws = register_worker(client)
        result = {}

        def call_gateway():
            result["response"] = client.post("/v1/chat/completions", json=chat_payload())

        thread = threading.Thread(target=call_gateway)
        thread.start()
        job = ws.receive_json()
        ws.send_json(
            {
                "type": "job_done",
                "job_id": job["job_id"],
                "content": "hi there",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )
        thread.join(timeout=5)
        websocket_ctx.__exit__(None, None, None)

    assert not thread.is_alive()
    body = result["response"].json()
    assert body["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_worker_reconnect_does_not_let_old_socket_unregister_new_socket(tmp_path):
    with make_client(tmp_path) as client:
        first_ctx, _ = register_worker(client)
        second_ctx, _ = register_worker(client)

        first_ctx.__exit__(None, None, None)
        health = client.get("/health").json()
        second_ctx.__exit__(None, None, None)

    assert health["active_workers"] == 1


def test_health_writes_live_worker_file(tmp_path):
    live_nodes_path = tmp_path / "live_workers.json"
    with make_client(tmp_path) as client:
        websocket_ctx, _ = register_worker(client)
        health = client.get("/health").json()
        websocket_ctx.__exit__(None, None, None)

    payload = json.loads(live_nodes_path.read_text(encoding="utf-8"))
    assert health["active_workers"] == 1
    assert payload["active_workers"] == 1
    assert payload["workers"][0]["node_id"] == "node-1"


def test_uptime_summary_groups_by_account_and_capacity(tmp_path):
    with make_client(tmp_path, api_key="api-key") as client:
        websocket_ctx, _ = register_worker(client)
        response = client.get(
            "/workers/uptime",
            headers={"Authorization": "Bearer api-key"},
        )
        websocket_ctx.__exit__(None, None, None)

    assert response.status_code == 200
    body = response.json()
    assert body["active_nodes"] == 1
    assert body["active_capacity"] == 1
    assert body["accounts"][0]["owner"] == "tester"
    assert body["accounts"][0]["active_nodes"] == 1


def test_root_can_terminate_worker(tmp_path):
    with make_client(tmp_path, api_key="api-key") as client:
        websocket_ctx, ws = register_worker(client)
        response = client.post(
            "/workers/node-1/terminate",
            headers={"Authorization": "Bearer api-key"},
            json={"reason": "test shutdown"},
        )
        message = ws.receive_json()
        websocket_ctx.__exit__(None, None, None)

    assert response.status_code == 200
    assert response.json()["terminated"] is True
    assert message["type"] == "terminate"
    assert message["reason"] == "test shutdown"


def test_streaming_completion_emits_sse_chunks(tmp_path):
    with make_client(tmp_path) as client:
        websocket_ctx, ws = register_worker(client)
        result = {}

        def call_gateway():
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json=chat_payload(stream=True),
            ) as response:
                result["status_code"] = response.status_code
                result["body"] = "".join(response.iter_text())

        thread = threading.Thread(target=call_gateway)
        thread.start()
        job = ws.receive_json()
        ws.send_json({"type": "token_delta", "job_id": job["job_id"], "content": "he"})
        ws.send_json({"type": "token_delta", "job_id": job["job_id"], "content": "llo"})
        ws.send_json({"type": "job_done", "job_id": job["job_id"]})
        thread.join(timeout=5)
        websocket_ctx.__exit__(None, None, None)

    assert not thread.is_alive()
    assert result["status_code"] == 200
    assert "data: " in result["body"]
    assert "[DONE]" in result["body"]
    assert "hello" not in result["body"]
    assert "he" in result["body"]
    assert "llo" in result["body"]


def test_worker_disconnect_during_job_returns_controlled_error(tmp_path):
    with make_client(tmp_path) as client:
        websocket_ctx, ws = register_worker(client)
        result = {}

        def call_gateway():
            result["response"] = client.post("/v1/chat/completions", json=chat_payload())

        thread = threading.Thread(target=call_gateway)
        thread.start()
        job = ws.receive_json()
        assert job["type"] == "job"
        websocket_ctx.__exit__(None, None, None)
        thread.join(timeout=5)

    assert not thread.is_alive()
    response = result["response"]
    assert response.status_code == 502
    assert "disconnected" in response.json()["detail"]
