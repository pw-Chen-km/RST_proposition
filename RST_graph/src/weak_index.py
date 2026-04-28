"""
Global Weak Adjacency Index for TRACE-RAG.

A_w[i][j] = |entities(i) ∩ entities(j)|  (shared named-entity count)

This index is:
  - Built ONCE after ainsert() completes (corpus-level, not per-query).
  - Serialised to workspace/weak_index.json.
  - Loaded at TraceRetriever.__init__() and reused across all queries.

At query time, EvidenceInducer uses neighbors() to do multi-hop BFS
over the weak graph, subject to the query's bridging budget b_q.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class WeakAdjacencyIndex:
    """
    Sparse shared-entity connectivity layer over proposition nodes.

    Attributes:
        sparse_aw:      {(node_i, node_j): shared_entity_count}
                        Only non-zero pairs are stored (undirected symmetric).
        node_entity_map: {node_id: [entity, ...]}
    """

    sparse_aw: Dict[Tuple[str, str], int] = field(default_factory=dict)
    node_entity_map: Dict[str, List[str]] = field(default_factory=dict)
    _adj: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Build adjacency list if sparse_aw is already populated (e.g. from load or manual init)
        if self.sparse_aw:
            self._build_adj()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, graph: nx.DiGraph, ner=None) -> None:
        """
        Construct A_w from a proposition graph.

        Args:
            graph: The full proposition nx.DiGraph.
            ner:   A PropositionNER instance.  If None, one is created
                   with the default model (en_core_web_sm).
        """
        if ner is None:
            from .ner import PropositionNER
            ner = PropositionNER()

        self.node_entity_map = ner.build_node_entity_map(graph)
        self.sparse_aw = self._compute_sparse(self.node_entity_map)
        self._build_adj()
        logger.info(
            f"WeakAdjacencyIndex built: {len(self.node_entity_map)} nodes, "
            f"{len(self.sparse_aw)} non-zero pairs."
        )

    def _build_adj(self) -> None:
        """Cache adjacency list for O(1) neighbor lookups."""
        self._adj = {}
        for (ni, nj), score in self.sparse_aw.items():
            if ni not in self._adj: self._adj[ni] = []
            if nj not in self._adj: self._adj[nj] = []
            self._adj[ni].append((nj, score))
            self._adj[nj].append((ni, score))

    def _compute_sparse(
        self, node_entity_map: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """
        Compute pairwise shared-entity counts.

        We store only pairs with count > 0, and only in canonical order
        (i < j lexicographically) to halve storage; bridge_score() and
        neighbors() both handle the symmetric lookup.
        """
        # Batch by entity to avoid O(N^2)
        entity_to_nodes: Dict[str, List[str]] = {}
        for nid, ents in node_entity_map.items():
            for e in ents:
                if e not in entity_to_nodes:
                    entity_to_nodes[e] = []
                entity_to_nodes[e].append(nid)

        sparse: Dict[Tuple[str, str], int] = {}
        # Only iterate over pairs that share at least one entity
        for entity, nids in entity_to_nodes.items():
            # If an entity appears in too many nodes (e.g. "maya"), 
            # we might want to skip it to prevent a dense weak graph.
            # For now, let's keep all.
            for i_idx, ni in enumerate(nids):
                for nj in nids[i_idx + 1 :]:
                    key = (ni, nj) if ni < nj else (nj, ni)
                    sparse[key] = sparse.get(key, 0) + 1

        return sparse

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def bridge_score(self, node_i: str, node_j: str) -> int:
        """Direct shared-entity score between two nodes (0 if none)."""
        key = (node_i, node_j) if node_i < node_j else (node_j, node_i)
        return self.sparse_aw.get(key, 0)

    def neighbors(self, node: str) -> List[Tuple[str, int]]:
        """
        Return all nodes weakly connected to ``node`` and their scores.

        Used by EvidenceInducer for multi-hop weak-path BFS.
        Returns: [(neighbor_node_id, shared_entity_count), ...]
        """
        return self._adj.get(node, [])

    def island_bridge_score(
        self,
        island_a: Set[str],
        island_b: Set[str],
        b_q: int,
    ) -> Tuple[float, List[str]]:
        """
        Find the best weak path (≤ b_q intermediate nodes) between two islands.

        Returns:
            (best_score, best_intermediate_path)
            best_score is 0 if no valid path exists within budget.
        """
        all_island_nodes: Set[str] = island_a | island_b

        best_score = 0
        best_intermediates: List[str] = []

        # BFS from every node on the border of island_a
        for start in island_a:
            # queue entries: (current_node, intermediates_so_far, min_score_so_far)
            queue: List[Tuple[str, List[str], int]] = [(start, [], 10**9)]
            visited_from_start: Set[str] = {start}

            while queue:
                curr, intermediates, path_score = queue.pop(0)

                for neighbor, w_score in self.neighbors(curr):
                    step_score = min(path_score, w_score)

                    if neighbor in island_b:
                        # Reached target island
                        if step_score > best_score:
                            best_score = step_score
                            best_intermediates = list(intermediates)
                        continue

                    # Only traverse intermediate nodes NOT in any island
                    if neighbor in all_island_nodes:
                        continue
                    if neighbor in visited_from_start:
                        continue
                    if len(intermediates) >= b_q:
                        # Budget exhausted; cannot go deeper
                        continue

                    visited_from_start.add(neighbor)
                    queue.append((neighbor, intermediates + [neighbor], step_score))

        return float(best_score), best_intermediates

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, workspace: str) -> None:
        path = os.path.join(workspace, "weak_index.json")
        payload = {
            # JSON keys must be strings; encode tuple as "a|||b"
            "sparse_aw": {
                f"{ni}|||{nj}": score
                for (ni, nj), score in self.sparse_aw.items()
            },
            "node_entity_map": self.node_entity_map,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info(f"WeakAdjacencyIndex saved → {path}")

    def load(self, workspace: str) -> bool:
        """
        Load from workspace.  Returns True on success, False if file absent.
        """
        path = os.path.join(workspace, "weak_index.json")
        if not os.path.exists(path):
            logger.warning(f"weak_index.json not found at {path}.")
            return False
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.sparse_aw = {
            tuple(k.split("|||")): v  # type: ignore[misc]
            for k, v in payload["sparse_aw"].items()
        }
        self.node_entity_map = payload["node_entity_map"]
        self._build_adj()
        logger.info(
            f"WeakAdjacencyIndex loaded: {len(self.node_entity_map)} nodes, "
            f"{len(self.sparse_aw)} pairs."
        )
        return True
