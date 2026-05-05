from __future__ import annotations

import asyncio
import json
import sqlite3
import threading

from fastapi.testclient import TestClient

from kaggle_host_llm.app import create_app
from kaggle_host_llm.registry import WorkerRegistry
from kaggle_host_llm.settings import Settings


MODEL = "qwen2.5-9b-quantized"


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
