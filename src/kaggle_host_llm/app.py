from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Body, Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .dispatcher import Dispatcher
from .openai_models import ChatCompletionRequest
from .registry import WorkerRegistry
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    registry = WorkerRegistry(
        resolved_settings.database_path,
        heartbeat_timeout_seconds=resolved_settings.heartbeat_timeout_seconds,
    )
    dispatcher = Dispatcher(
        registry,
        job_timeout_seconds=resolved_settings.job_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        resolved_settings.ensure_database_parent()
        registry.init_db()
        await registry.cleanup_stale()
        registry.write_live_nodes_file(resolved_settings.live_nodes_path)

        async def monitor_live_nodes() -> None:
            while True:
                await asyncio.sleep(resolved_settings.alive_check_interval_seconds)
                await registry.cleanup_stale()
                registry.write_live_nodes_file(resolved_settings.live_nodes_path)

        monitor_task = asyncio.create_task(monitor_live_nodes())
        try:
            yield
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.registry = registry
    app.state.dispatcher = dispatcher

    def require_api_key(authorization: str = Header(default="")) -> None:
        if not resolved_settings.api_key:
            return
        expected = f"Bearer {resolved_settings.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    async def health() -> dict[str, object]:
        await registry.cleanup_stale()
        live_nodes = registry.write_live_nodes_file(resolved_settings.live_nodes_path)
        nodes = registry.list_nodes()
        return {
            "ok": True,
            "active_workers": len(live_nodes),
            "workers": nodes,
        }

    @app.get("/workers/live")
    async def live_workers(_: None = Depends(require_api_key)) -> dict[str, object]:
        await registry.cleanup_stale()
        live_nodes = registry.write_live_nodes_file(resolved_settings.live_nodes_path)
        return {
            "live_nodes_path": resolved_settings.live_nodes_path,
            "active_workers": len(live_nodes),
            "workers": live_nodes,
        }

    @app.get("/workers/uptime")
    async def worker_uptime(_: None = Depends(require_api_key)) -> dict[str, object]:
        await registry.cleanup_stale()
        registry.write_live_nodes_file(resolved_settings.live_nodes_path)
        return registry.uptime_summary()

    @app.post("/workers/{node_id}/terminate")
    async def terminate_worker(
        node_id: str,
        payload: dict[str, str] | None = Body(default=None),
        _: None = Depends(require_api_key),
    ) -> dict[str, object]:
        reason = (payload or {}).get("reason") or "terminated by root"
        terminated = await registry.terminate_worker(node_id, reason=reason)
        if not terminated:
            raise HTTPException(status_code=404, detail="worker not found or not connected")
        registry.write_live_nodes_file(resolved_settings.live_nodes_path)
        return {
            "terminated": True,
            "node_id": node_id,
            "reason": reason,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        _: None = Depends(require_api_key),
    ):
        if request.stream:
            stream = await dispatcher.stream(request)
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
            )
        return await dispatcher.complete(request)

    @app.websocket("/workers/connect")
    async def connect_worker(websocket: WebSocket) -> None:
        if resolved_settings.worker_token:
            token = websocket.query_params.get("token", "")
            if token != resolved_settings.worker_token:
                await websocket.close(code=1008)
                return

        await websocket.accept()
        worker = None
        try:
            register_message = await websocket.receive_json()
            if register_message.get("type") != "register":
                await websocket.close(code=1003, reason="first message must be register")
                return
            worker = await registry.register(websocket, register_message)
            await websocket.send_json(
                {
                    "type": "registered",
                    "node_id": worker.node_id,
                    "model": worker.model,
                    "capacity": worker.capacity,
                }
            )
            while True:
                message = await websocket.receive_json()
                await registry.deliver_worker_message(worker.node_id, message)
        except WebSocketDisconnect:
            pass
        except ValueError as exc:
            await websocket.close(code=1003, reason=str(exc))
        finally:
            if worker is not None:
                await registry.unregister(worker.node_id, websocket=worker.websocket)

    return app


app = create_app()
