"""The hosted write path materialises ``references`` edges, and a malformed
trace body is a 422.

Found on dev (2026-10-19, t2-11a proof): two entries written over HTTP with
``pattern_refs`` had no ``references`` edge until ``/pro/graph/backfill`` ran,
because the async write path writes the entry through the adapter directly and
only the sync ``AgentMemory.write`` materialised them. Propagation
(``propagate_failure``) walks those edges, so a hosted writer's children were
never charged for a parent's failure. And ``POST /api/v1/traces`` answered a
missing ``TraceEntry.confidence`` with a 500.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory
from amfs_core.models import GraphEdge
from amfs_filesystem.adapter import FilesystemAdapter


class _AsyncAdapterSpy:
    """The slice of the async adapter the write route touches."""

    def __init__(self, handle: AgentMemory) -> None:
        self._handle = handle
        self._namespace = handle.namespace
        self.edges: list[GraphEdge] = []

    async def write(self, entry):
        return self._handle._adapter.write(entry)

    async def upsert_graph_edge(self, edge: GraphEdge, *, namespace: str, branch: str) -> None:
        self.edges.append(edge)

    async def log_event(self, *a, **k) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
def server_bits(monkeypatch, tmp_path):
    from amfs_http import server

    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    handle = AgentMemory(agent_id="http-server", adapter=adapter)
    spy = _AsyncAdapterSpy(handle)
    monkeypatch.setattr(server, "_memory", handle)
    monkeypatch.setattr(server, "_get_memory", lambda: handle)
    monkeypatch.setattr(server, "_async_adapter", spy)
    monkeypatch.setattr(server, "_get_server_embedder", lambda: None)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(server, "_HAS_PRO_TRACES", False, raising=False)
    return server, spy


def _settle(client) -> None:
    """Let the write route's background task run: TestClient drives the app
    on its own loop, and a no-op request after the write is the barrier."""
    client.get("/health")


def test_async_write_path_materialises_references_edges(server_bits) -> None:
    from fastapi.testclient import TestClient

    server, spy = server_bits
    with TestClient(server.app) as client:
        r = client.post(
            "/api/v1/entries",
            json={
                "entity_path": "research/drill-derived",
                "key": "child-template",
                "value": "derived from the parent rule",
                "agent_id": "research-agent",
                "pattern_refs": [
                    "research/drill/parent-rule",
                    "research/drill/parent-rule",
                    "not-a-spec",
                    "research/drill-derived/child-template",
                ],
            },
        )
        assert r.status_code == 200, r.text
        _settle(client)
    refs = [e for e in spy.edges if e.relation == "references"]
    assert refs, "no references edge was materialised on the async write path"
    # Every edge is child -> parent; a bare word is not an entry spec and a
    # self-reference is not an edge. (The duplicate ref is one edge in the
    # graph: upsert_graph_edge bumps evidence_count rather than adding a row.)
    assert {(e.source_entity, e.target_entity, e.source_type, e.target_type) for e in refs} == {
        ("research/drill-derived/child-template", "research/drill/parent-rule", "entry", "entry"),
    }
    assert refs[0].provenance == {"agent_id": "research-agent", "trigger": "write"}
    wrote = [e for e in spy.edges if e.relation == "wrote"]
    assert len(wrote) == 1 and wrote[0].target_entity == "research/drill-derived/child-template"


def test_malformed_trace_body_is_a_422_not_a_500(server_bits) -> None:
    from fastapi.testclient import TestClient

    server, _ = server_bits
    with TestClient(server.app) as client:
        r = client.post(
            "/api/v1/traces",
            json={
                "id": "t-1",
                "agent_id": "research-agent",
                "session_id": "s-1",
                "causal_entries": [
                    {"entity_path": "research/policies", "key": "data-retention", "version": 1}
                ],
            },
        )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert any(d.get("loc", [])[-1] == "confidence" for d in detail), detail
