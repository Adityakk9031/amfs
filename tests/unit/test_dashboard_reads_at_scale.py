"""The whole-account reads a dashboard page makes, at scale.

One Agents page mount on a 6,000-agent, 430K-entry account fired a cluster
compile (minutes of Python), a consolidation status that loaded every entry
to count ten paths, a 3 MB roster and a 600 KB stats payload — and stalled
every other request on the instance while it did. These pin what replaced
that: paging and search in SQL with a ``total``, a status counted in SQL,
stats without the per-agent and per-entity maps, digests fetched by scope,
and a cluster graph built from candidate pairs rather than every pair.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
import amfs_http.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _client(monkeypatch, adapter, *, vis=None, visible_agents=None):
    mem = types.SimpleNamespace(
        namespace="ns",
        _adapter=adapter,
        list=lambda entity_path=None, branch="main": list(getattr(adapter, "entries", [])),
    )
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    monkeypatch.setattr(server, "_async_adapter", None)
    monkeypatch.setattr(server, "_active_visibility_filter", lambda request: vis)
    monkeypatch.setattr(server, "_get_visibility_filter", lambda request: vis)
    monkeypatch.setattr(
        server, "_visible_agent_ids",
        lambda request: None if visible_agents is None else set(visible_agents),
    )
    return TestClient(server.app)


# ── /agents/enriched pages in SQL ─────────────────────────────────────────


class _PagingAgentsAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    def list_agents_enriched(self, namespace="default", *, agent_ids=None, q=None,
                             platform=None, sort="last_active", limit=None, offset=0):
        self.calls.append(dict(
            agent_ids=agent_ids, q=q, platform=platform, sort=sort, limit=limit, offset=offset,
        ))
        if sort not in ("last_active", "entries_written", "created_at", "agent_id"):
            raise ValueError(f"unknown sort {sort!r}")
        if offset >= 3:
            return []
        rows = [{"agent_id": f"a{i}", "entries_written": 10 - i} for i in range(offset, 3)]
        if limit is not None:
            rows = rows[:limit]
            for r in rows:
                r["total"] = 3
                r["active_7d"] = 2
        return rows

    def get_agent_activity_histogram(self, aid, days=7, namespace="default"):
        return [1] * days


class TestEnrichedAgentsPage:
    def test_a_page_carries_the_total_and_the_parameters_reach_sql(self, monkeypatch):
        adapter = _PagingAgentsAdapter()
        client = _client(monkeypatch, adapter, visible_agents={"a0", "a2"})
        res = client.get(
            "/api/v1/agents/enriched?limit=2&offset=0&q=bil&platform=cursor"
            "&sort=entries_written&histograms=false"
        )
        assert res.status_code == 200
        body = res.json()
        assert [a["agent_id"] for a in body["agents"]] == ["a0", "a1"]
        assert (body["total"], body["active_7d"], body["limit"], body["offset"]) == (3, 2, 2, 0)
        assert all("total" not in a and "active_7d" not in a for a in body["agents"]), (
            "the counts are on the page only"
        )
        assert adapter.calls == [dict(
            agent_ids=["a0", "a2"], q="bil", platform="cursor", sort="entries_written",
            limit=2, offset=0,
        )]

    def test_a_page_past_the_end_still_reports_the_count(self, monkeypatch):
        adapter = _PagingAgentsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/agents/enriched?limit=2&offset=10&histograms=false").json()
        assert body["agents"] == [] and body["total"] == 3 and body["active_7d"] == 2
        assert adapter.calls[-1]["limit"] == 1 and adapter.calls[-1]["offset"] == 0

    def test_unpaged_is_the_whole_roster_as_before(self, monkeypatch):
        adapter = _PagingAgentsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/agents/enriched?histograms=false").json()
        assert len(body["agents"]) == 3 and "total" not in body
        assert adapter.calls == [dict(
            agent_ids=None, q=None, platform=None, sort="last_active", limit=None, offset=0,
        )]

    def test_an_unknown_sort_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _PagingAgentsAdapter())
        assert client.get("/api/v1/agents/enriched?sort=karma").status_code == 400

    def test_an_adapter_without_paging_is_filtered_here(self, monkeypatch):
        class _Old:
            def list_agents_enriched(self, namespace="default"):
                return [{"agent_id": "a0"}, {"agent_id": "hidden"}]

            def get_agent_activity_histogram(self, aid, days=7, namespace="default"):
                return [0] * days

        client = _client(monkeypatch, _Old(), visible_agents={"a0"})
        body = client.get("/api/v1/agents/enriched?limit=1&histograms=false").json()
        assert [a["agent_id"] for a in body["agents"]] == ["a0"]
        assert "total" not in body, "no SQL paging happened, so no page count is claimed"


# ── /entities pages in SQL ────────────────────────────────────────────────


class _PagingEntitiesAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    def entity_summaries(self, *, scope=None, q=None, agent=None, min_confidence=None,
                         sort="last_updated", limit=None, offset=0):
        self.calls.append(dict(
            scope=scope, q=q, agent=agent, min_confidence=min_confidence, sort=sort,
            limit=limit, offset=offset,
        ))
        rows = [{"entity_path": f"acme/{p}", "entry_count": 5 - i} for i, p in enumerate("abcd")]
        if q:
            rows = [r for r in rows if q in r["entity_path"]]
        total = len(rows)
        if limit is not None:
            rows = rows[offset:offset + limit]
            for r in rows:
                r["total"] = total
        return rows


class TestEntitySummariesPage:
    def test_a_page_with_search_and_total(self, monkeypatch):
        adapter = _PagingEntitiesAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get(
            "/api/v1/entities?limit=2&offset=1&q=acme&agent=bob&min_confidence=0.7"
            "&sort=entry_count"
        ).json()
        assert [e["entity_path"] for e in body["entities"]] == ["acme/b", "acme/c"]
        assert (body["total"], body["limit"], body["offset"]) == (4, 2, 1)
        assert all("total" not in e for e in body["entities"])
        assert adapter.calls == [dict(
            scope=None, q="acme", agent="bob", min_confidence=0.7, sort="entry_count",
            limit=2, offset=1,
        )]

    def test_unpaged_unchanged(self, monkeypatch):
        adapter = _PagingEntitiesAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/entities").json()
        assert len(body["entities"]) == 4 and "total" not in body
        assert adapter.calls == [dict(
            scope=None, q=None, agent=None, min_confidence=None, sort="last_updated",
            limit=None, offset=0,
        )]

    def test_an_older_adapter_is_searched_and_paged_here(self, monkeypatch):
        class _Old:
            def entity_summaries(self):
                return [
                    {"entity_path": "acme/a", "entry_count": 1, "agents": ["bob"],
                     "avg_confidence": 0.95},
                    {"entity_path": "beta/b", "entry_count": 9, "agents": ["bob"],
                     "avg_confidence": 0.9},
                    {"entity_path": "acme/c", "entry_count": 4, "agents": ["bob"],
                     "avg_confidence": 0.8},
                    {"entity_path": "acme/d", "entry_count": 7, "agents": ["eve"],
                     "avg_confidence": 0.99},
                    {"entity_path": "acme/e", "entry_count": 8, "agents": ["bob"],
                     "avg_confidence": 0.3},
                ]

        client = _client(monkeypatch, _Old())
        body = client.get(
            "/api/v1/entities?limit=1&offset=1&q=acme&agent=bob&min_confidence=0.7"
            "&sort=entry_count"
        ).json()
        assert [e["entity_path"] for e in body["entities"]] == ["acme/a"]
        assert body["total"] == 2, "eve's path and the low-confidence one are filtered here"
        low_first = client.get("/api/v1/entities?sort=avg_confidence_asc&limit=2").json()
        assert [e["entity_path"] for e in low_first["entities"]] == ["acme/e", "acme/c"]

    def test_an_unknown_sort_is_a_400(self, monkeypatch):
        client = _client(monkeypatch, _PagingEntitiesAdapter())
        assert client.get("/api/v1/entities?sort=vibes").status_code == 400


# ── /stats without the maps ───────────────────────────────────────────────


class _StatsAdapter:
    def __init__(self):
        self.calls: list[dict] = []

    def stats_extended(self, *, scope=None, breakdown=True):
        self.calls.append(dict(scope=scope, breakdown=breakdown))
        out = {"total_entries": 7, "total_agents": 2, "total_entities": 3, "total_recalls": 1}
        if breakdown:
            out["agents"] = {"a": 4, "b": 3}
            out["entities"] = {"x": 7}
        return out


class TestStatsBreakdown:
    def test_breakdown_false_leaves_the_maps_out_and_keeps_the_totals(self, monkeypatch):
        adapter = _StatsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/stats?breakdown=false").json()
        assert "agents" not in body and "entities" not in body
        assert (body["total_agents"], body["total_entities"]) == (2, 3)
        assert adapter.calls == [dict(scope=None, breakdown=False)]

    def test_default_is_the_full_shape(self, monkeypatch):
        adapter = _StatsAdapter()
        client = _client(monkeypatch, adapter)
        body = client.get("/api/v1/stats").json()
        assert body["agents"] == {"a": 4, "b": 3}
        assert adapter.calls == [dict(scope=None, breakdown=True)]

    def test_an_adapter_without_the_option_is_trimmed_here(self, monkeypatch):
        class _Old:
            def stats_extended(self):
                return {"total_entries": 1, "total_agents": 1, "total_entities": 1,
                        "agents": {"a": 1}, "entities": {"x": 1}}

        client = _client(monkeypatch, _Old())
        body = client.get("/api/v1/stats?breakdown=false").json()
        assert "agents" not in body and body["total_agents"] == 1


# ── consolidation status counts in SQL ────────────────────────────────────


class TestConsolidationStatus:
    def test_ready_entities_come_from_one_grouped_query(self, monkeypatch):
        calls: list[dict] = []

        class _Adapter:
            def list_branches(self, namespace="default", status="active"):
                return []

            def consolidation_ready_entities(self, *, min_entries, limit, scope=None):
                calls.append(dict(min_entries=min_entries, limit=limit, scope=scope))
                return ["acme/billing", "acme/auth"]

            def list(self, *a, **kw):
                raise AssertionError("the status must not load the account's entries")

        client = _client(monkeypatch, _Adapter())
        monkeypatch.setattr(server, "_cortex_worker", None)
        body = client.get("/api/v1/cortex/consolidation/status").json()
        assert body["entities_ready"] == ["acme/billing", "acme/auth"]
        assert calls == [dict(min_entries=10, limit=10, scope=None)]

    def test_an_adapter_without_the_query_counts_loaded_entries(self, monkeypatch):
        entries = [types.SimpleNamespace(entity_path="big") for _ in range(10)]
        entries += [types.SimpleNamespace(entity_path="small") for _ in range(3)]

        class _Adapter:
            def list_branches(self, namespace="default", status="active"):
                return []

            def list(self, *a, **kw):
                return list(entries)

        client = _client(monkeypatch, _Adapter())
        monkeypatch.setattr(server, "_cortex_worker", None)
        assert client.get("/api/v1/cortex/consolidation/status").json()["entities_ready"] == ["big"]


# ── /cortex/digests filters in SQL ────────────────────────────────────────


def test_digests_scope_and_limit_reach_the_adapter(monkeypatch):
    from amfs_postgres.adapter import PostgresAdapter

    calls: list[dict] = []
    adapter = object.__new__(PostgresAdapter)
    adapter._namespace = "ns"

    def list_digests(digest_type=None, namespace="default", branch="main", *, scope=None,
                     written_by=None, relevant_to=None, limit=None):
        calls.append(dict(
            digest_type=digest_type, namespace=namespace, scope=scope, written_by=written_by,
            limit=limit,
        ))
        return []

    adapter.list_digests = list_digests
    client = _client(monkeypatch, adapter)
    body = client.get(
        "/api/v1/cortex/digests?digest_type=agent_brief&scope=billing-agent&limit=1"
    ).json()
    assert body == {"digests": [], "total": 0}
    client.get("/api/v1/cortex/digests?digest_type=entity&written_by=billing-agent&limit=50")
    assert len(calls) == 2
    assert (calls[0]["digest_type"].value, calls[0]["scope"], calls[0]["limit"]) == (
        "agent_brief", "billing-agent", 1,
    )
    assert calls[0]["written_by"] is None
    assert (calls[1]["digest_type"].value, calls[1]["written_by"], calls[1]["limit"]) == (
        "entity", "billing-agent", 50,
    )


# ── the cluster graph is built from candidate pairs ───────────────────────


class TestClusterCandidates:
    @pytest.fixture
    def strategy(self):
        pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")
        from amfs_cortex.clustering import AgentClusterStrategy

        return AgentClusterStrategy

    def test_pairs_share_a_path_or_an_edge_and_nothing_else_is_scored(self, strategy):
        ids = ["a", "b", "c", "d"]
        paths = {"a": {"x"}, "b": {"x", "y"}, "c": {"y"}, "d": {"z"}}
        collab = {"a": set(), "b": set(), "c": set(), "d": {"a"}}
        assert strategy.candidate_pairs(ids, paths, collab) == [("a", "b"), ("a", "d"), ("b", "c")]

    def test_a_fleet_with_nothing_in_common_proposes_no_pairs(self, strategy):
        ids = [f"agent-{i}" for i in range(5000)]
        paths = {aid: {f"repo-{aid}"} for aid in ids}
        collab = {aid: set() for aid in ids}
        assert strategy.candidate_pairs(ids, paths, collab) == []

    def test_a_hub_path_proposes_nothing_but_still_counts_in_the_score(self, strategy):
        ids = [f"agent-{i}" for i in range(strategy.HUB_PATH_AGENTS + 1)]
        paths = {aid: {"everyone/writes/here"} for aid in ids}
        paths["agent-0"].add("pair")
        paths["agent-1"].add("pair")
        collab = {aid: set() for aid in ids}
        assert strategy.candidate_pairs(ids, paths, collab) == [("agent-0", "agent-1")]

    def test_edges_to_unknown_agents_and_self_are_ignored_and_order_is_stable(self, strategy):
        ids = ["b", "a"]
        paths = {"a": {"x"}, "b": {"x"}}
        collab = {"a": {"a", "ghost", "b"}, "b": set()}
        assert strategy.candidate_pairs(ids, paths, collab) == [("b", "a")]

    def test_type_similarity_alone_no_longer_makes_an_edge(self, strategy):
        """Two agents that share nothing but a taste for facts are not a cluster."""
        from amfs_core.models import GraphEdge

        agents = [
            {"agent_id": a, "entity_paths": [f"only/{a}"], "type_dist": {"fact": 10}}
            for a in ("a", "b", "c")
        ]
        edge = GraphEdge(
            source_entity="x", source_type="agent", relation="learned_from",
            target_entity="y", target_type="agent",
        )
        digest = strategy(scope="account:ns").compute_clusters(agents, [edge])
        assert digest.summary["clusters"] == []
