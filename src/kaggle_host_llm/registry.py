from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import WebSocket


@dataclass
class PendingJob:
    job_id: str
    queue: asyncio.Queue[dict[str, Any]]


@dataclass
class WorkerConnection:
    node_id: str
    owner: str
    model: str
    accelerator: str
    capacity: int
    websocket: WebSocket
    current_jobs: int = 0
    pending_jobs: dict[str, PendingJob] = field(default_factory=dict)

    @property
    def has_capacity(self) -> bool:
        return self.current_jobs < self.capacity


class WorkerRegistry:
    def __init__(self, database_path: str, heartbeat_timeout_seconds: float = 45.0) -> None:
        self.database_path = database_path
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._lock = asyncio.Lock()
        self._connections: dict[str, WorkerConnection] = {}

    def init_db(self) -> None:
        path = Path(self.database_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_nodes (
                    node_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    model TEXT NOT NULL,
                    accelerator TEXT NOT NULL,
                    status TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    current_jobs INTEGER NOT NULL,
                    last_heartbeat REAL NOT NULL,
                    connected_at REAL NOT NULL,
                    disconnected_at REAL,
                    total_uptime_seconds REAL NOT NULL DEFAULT 0,
                    job_count INTEGER NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_column(conn, "worker_nodes", "disconnected_at", "REAL")
            self._ensure_column(
                conn, "worker_nodes", "total_uptime_seconds", "REAL NOT NULL DEFAULT 0"
            )
            self._ensure_column(conn, "worker_nodes", "job_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "worker_nodes", "last_seen", "REAL NOT NULL DEFAULT 0")
            conn.execute(
                """
                UPDATE worker_nodes
                SET last_seen = CASE
                    WHEN last_seen = 0 THEN last_heartbeat
                    ELSE last_seen
                END
                """
            )
            conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    async def register(self, websocket: WebSocket, metadata: dict[str, Any]) -> WorkerConnection:
        node_id = str(metadata.get("node_id") or "").strip()
        owner = str(metadata.get("owner") or "unknown").strip()
        model = str(metadata.get("model") or "").strip()
        accelerator = str(metadata.get("accelerator") or "unknown").strip()
        capacity = int(metadata.get("capacity") or 1)
        if not node_id:
            raise ValueError("worker register message requires node_id")
        if not model:
            raise ValueError("worker register message requires model")
        if capacity < 1:
            raise ValueError("worker capacity must be at least 1")

        now = time.time()
        async with self._lock:
            previous = self._connections.get(node_id)
            if previous is not None:
                await self._fail_pending_locked(previous, "worker reconnected")
            worker = WorkerConnection(
                node_id=node_id,
                owner=owner,
                model=model,
                accelerator=accelerator,
                capacity=capacity,
                websocket=websocket,
            )
            self._connections[node_id] = worker
            self._upsert_worker(worker, status="active", now=now)
            return worker

    async def unregister(
        self,
        node_id: str,
        reason: str = "worker disconnected",
        websocket: WebSocket | None = None,
    ) -> None:
        async with self._lock:
            worker = self._connections.get(node_id)
            if websocket is not None and worker is not None and worker.websocket is not websocket:
                return
            worker = self._connections.pop(node_id, None)
            if worker is not None:
                await self._fail_pending_locked(worker, reason)
            now = time.time()
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    """
                    UPDATE worker_nodes
                    SET status = ?,
                        current_jobs = 0,
                        disconnected_at = ?,
                        last_seen = ?,
                        total_uptime_seconds = total_uptime_seconds + MAX(0, ? - connected_at)
                    WHERE node_id = ?
                    """,
                    ("disconnected", now, now, now, node_id),
                )
                conn.commit()

    async def heartbeat(self, node_id: str) -> None:
        async with self._lock:
            now = time.time()
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    """
                    UPDATE worker_nodes
                    SET status = ?, last_heartbeat = ?, last_seen = ?
                    WHERE node_id = ?
                    """,
                    ("active", now, now, node_id),
                )
                conn.commit()

    async def terminate_worker(self, node_id: str, reason: str = "terminated by root") -> bool:
        async with self._lock:
            worker = self._connections.get(node_id)
            if worker is None:
                return False
            await worker.websocket.send_json(
                {
                    "type": "terminate",
                    "node_id": node_id,
                    "reason": reason,
                }
            )
            await self._fail_pending_locked(worker, reason)
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    """
                    UPDATE worker_nodes
                    SET status = ?, current_jobs = 0, last_seen = ?
                    WHERE node_id = ?
                    """,
                    ("terminating", time.time(), node_id),
                )
                conn.commit()
            return True

    async def select_worker(self, model: str) -> WorkerConnection | None:
        await self.cleanup_stale()
        async with self._lock:
            candidates = [
                worker
                for worker in self._connections.values()
                if worker.model == model and worker.has_capacity
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda worker: (worker.current_jobs, worker.node_id))
            return candidates[0]

    async def reserve_job(self, worker: WorkerConnection, job_id: str) -> PendingJob:
        async with self._lock:
            latest = self._connections.get(worker.node_id)
            if latest is None or not latest.has_capacity:
                raise RuntimeError("worker is no longer available")
            pending = PendingJob(job_id=job_id, queue=asyncio.Queue())
            latest.current_jobs += 1
            latest.pending_jobs[job_id] = pending
            self._update_jobs_locked(latest, increment_job_count=True)
            return pending

    async def release_job(self, node_id: str, job_id: str) -> None:
        async with self._lock:
            worker = self._connections.get(node_id)
            if worker is None:
                return
            worker.pending_jobs.pop(job_id, None)
            worker.current_jobs = max(0, worker.current_jobs - 1)
            self._update_jobs_locked(worker)

    async def deliver_worker_message(self, node_id: str, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "heartbeat":
            await self.heartbeat(node_id)
            return
        job_id = str(message.get("job_id") or "")
        if not job_id:
            return
        async with self._lock:
            worker = self._connections.get(node_id)
            pending = worker.pending_jobs.get(job_id) if worker else None
            if pending is not None:
                await pending.queue.put(message)

    async def cleanup_stale(self) -> None:
        cutoff = time.time() - self.heartbeat_timeout_seconds
        stale: list[str] = []
        with sqlite3.connect(self.database_path) as conn:
            rows = conn.execute(
                """
                SELECT node_id FROM worker_nodes
                WHERE status = 'active' AND last_heartbeat < ?
                """,
                (cutoff,),
            ).fetchall()
            stale = [row[0] for row in rows]
        for node_id in stale:
            await self.unregister(node_id, reason="worker heartbeat timed out")

    def list_nodes(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT node_id, owner, model, accelerator, status, capacity,
                       current_jobs, last_heartbeat, connected_at, disconnected_at,
                       total_uptime_seconds, job_count, last_seen
                FROM worker_nodes
                ORDER BY node_id
                """
            ).fetchall()
            now = time.time()
            nodes = []
            for row in rows:
                node = dict(row)
                node["session_uptime_seconds"] = self._session_uptime_seconds(node, now)
                node["effective_uptime_seconds"] = (
                    float(node.get("total_uptime_seconds") or 0)
                    + node["session_uptime_seconds"]
                )
                nodes.append(node)
            return nodes

    def list_active_nodes(self) -> list[dict[str, Any]]:
        cutoff = time.time() - self.heartbeat_timeout_seconds
        return [
            node
            for node in self.list_nodes()
            if node["status"] == "active" and float(node["last_heartbeat"]) >= cutoff
        ]

    def write_live_nodes_file(self, path: str) -> list[dict[str, Any]]:
        live_nodes = self.list_active_nodes()
        output_path = Path(path)
        if output_path.parent and str(output_path.parent) != ".":
            output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.time(),
            "active_workers": len(live_nodes),
            "workers": live_nodes,
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        return live_nodes

    def uptime_summary(self) -> dict[str, Any]:
        nodes = self.list_nodes()
        accounts: dict[str, dict[str, Any]] = {}
        for node in nodes:
            owner = str(node.get("owner") or "unknown")
            account = accounts.setdefault(
                owner,
                {
                    "owner": owner,
                    "active_nodes": 0,
                    "active_capacity": 0,
                    "current_jobs": 0,
                    "total_jobs": 0,
                    "uptime_seconds": 0.0,
                    "node_ids": [],
                },
            )
            if node["status"] == "active":
                account["active_nodes"] += 1
                account["active_capacity"] += int(node.get("capacity") or 0)
                account["current_jobs"] += int(node.get("current_jobs") or 0)
            account["total_jobs"] += int(node.get("job_count") or 0)
            account["uptime_seconds"] += float(node.get("effective_uptime_seconds") or 0)
            account["node_ids"].append(node["node_id"])

        for account in accounts.values():
            account["uptime_minutes"] = round(account["uptime_seconds"] / 60, 2)
            account["uptime_hours"] = round(account["uptime_seconds"] / 3600, 4)

        return {
            "generated_at": time.time(),
            "active_nodes": sum(1 for node in nodes if node["status"] == "active"),
            "active_capacity": sum(
                int(node.get("capacity") or 0) for node in nodes if node["status"] == "active"
            ),
            "current_jobs": sum(
                int(node.get("current_jobs") or 0) for node in nodes if node["status"] == "active"
            ),
            "accounts": sorted(accounts.values(), key=lambda item: item["owner"]),
            "nodes": nodes,
        }

    async def fail_worker(self, node_id: str, job_id: str, reason: str) -> None:
        async with self._lock:
            worker = self._connections.get(node_id)
            pending = worker.pending_jobs.get(job_id) if worker else None
            if pending is not None:
                await pending.queue.put(
                    {"type": "job_error", "job_id": job_id, "error": reason}
                )

    async def _fail_pending_locked(self, worker: WorkerConnection, reason: str) -> None:
        for pending in worker.pending_jobs.values():
            await pending.queue.put(
                {"type": "job_error", "job_id": pending.job_id, "error": reason}
            )
        worker.pending_jobs.clear()
        worker.current_jobs = 0

    def _upsert_worker(self, worker: WorkerConnection, status: str, now: float) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO worker_nodes (
                    node_id, owner, model, accelerator, status, capacity,
                    current_jobs, last_heartbeat, connected_at, disconnected_at,
                    last_seen
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    owner = excluded.owner,
                    model = excluded.model,
                    accelerator = excluded.accelerator,
                    status = excluded.status,
                    capacity = excluded.capacity,
                    current_jobs = excluded.current_jobs,
                    last_heartbeat = excluded.last_heartbeat,
                    connected_at = excluded.connected_at,
                    disconnected_at = NULL,
                    last_seen = excluded.last_seen
                """,
                (
                    worker.node_id,
                    worker.owner,
                    worker.model,
                    worker.accelerator,
                    status,
                    worker.capacity,
                    worker.current_jobs,
                    now,
                    now,
                    None,
                    now,
                ),
            )
            conn.commit()

    def _update_jobs_locked(
        self,
        worker: WorkerConnection,
        increment_job_count: bool = False,
    ) -> None:
        job_count_increment = 1 if increment_job_count else 0
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                UPDATE worker_nodes
                SET current_jobs = ?,
                    last_heartbeat = ?,
                    last_seen = ?,
                    job_count = job_count + ?
                WHERE node_id = ?
                """,
                (worker.current_jobs, time.time(), time.time(), job_count_increment, worker.node_id),
            )
            conn.commit()

    def _session_uptime_seconds(self, node: dict[str, Any], now: float) -> float:
        if node.get("status") != "active":
            return 0.0
        connected_at = float(node.get("connected_at") or now)
        last_heartbeat = float(node.get("last_heartbeat") or connected_at)
        end_time = min(now, last_heartbeat + self.heartbeat_timeout_seconds)
        return max(0.0, end_time - connected_at)
