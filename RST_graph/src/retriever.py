import logging
import math
from typing import Dict, List, Set, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

class RSTRetriever:
    """
    A pure, simplified Retriever for RST_graph.
    No LLM query parsing, no question-type branching, no edge-type filtering.
    Just: Hybrid Seed Search -> 2-Hop BFS -> Weak Bridging -> Context Assembly.
    """
    def __init__(
        self,
        graph: nx.DiGraph,
        vector_store,
        weak_index,
        ner,
        llm_func,
        embed_func,
        top_k_seeds: int = 15,
        max_bfs_depth: int = 2,
        top_k_islands: int = 0
    ):
        self._graph = graph
        self._vdb = vector_store
        self._weak = weak_index
        self._ner = ner
        self._llm = llm_func
        self._embed = embed_func
        self.top_k_seeds = top_k_seeds
        self.max_bfs_depth = max_bfs_depth
        self.top_k_islands = top_k_islands  # 0 = disabled
        
        # Precompute lemma cache for IDF matching
        self._node_lemma_cache: Dict[str, str] = {}
        self._build_lemma_cache()

    async def retrieve(self, query: str, bridging_budget: int = 3, use_expansion: bool = False, probe_weight: float = 0.5, top_k_islands: int = 0) -> str:
        """Main retrieval pipeline. Returns the formatted context string."""
        import asyncio
        
        # 1. Embed Query + (optional) HyPE expansion — run in parallel
        if use_expansion:
            (embeddings, probe_embeddings) = await asyncio.gather(
                self._embed([query]),
                self._expand_query_hype(query)
            )
        else:
            embeddings = await self._embed([query])
            probe_embeddings = []
        
        h_q = embeddings[0] if embeddings else np.zeros(1536)

        # 2. Extract Keywords (Anchors)
        anchors = self._ner.extract_from_query(query)
        
        # 3. Hybrid Seed Search (Vector + IDF + HyPE probes)
        seeds, sem_scores = self._get_hybrid_seeds(h_q, anchors, probe_embeddings, probe_weight)
        seed_node_ids = [nid for nid, _ in seeds]
        
        if not seed_node_ids:
            return "[No matching seeds found in the graph]"
            
        # 4. Strong Edge BFS Expansion (2-hop)
        islands = self._build_strong_islands(seed_node_ids)
        
        # 5. Island-Level Top-K Filtering (optional)
        k_isl = top_k_islands if top_k_islands > 0 else self.top_k_islands
        if k_isl > 0:
            islands = self._score_and_filter_islands(islands, sem_scores, k_isl)

        # 6. Weak Bridging & Assembly
        assembled_nodes = self._assemble_and_bridge(islands, bridging_budget)
        
        # 7. Format Context
        return self._format_context(assembled_nodes)

    async def _expand_query_hype(self, query: str) -> List[np.ndarray]:
        """
        HyPE: Generate multiple proposition probes and return their embeddings.
        Each probe is a hypothetical document-style proposition the corpus might contain.
        Returns a list of embedding vectors (one per probe).
        """
        from .prompts import HYPE_SYSTEM, HYPE_USER
        prompt = HYPE_USER.format(query=query)
        try:
            raw = await self._llm(prompt, system_prompt=HYPE_SYSTEM)
            probes = [
                line.strip() for line in raw.strip().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            probes = probes[:6]  # cap at 6 probes
            if not probes:
                return []
            logger.info(f"HyPE probes ({len(probes)}): {probes}")
            embeddings = await self._embed(probes)
            return embeddings if embeddings else []
        except Exception as e:
            logger.warning(f"HyPE expansion failed: {e}")
            return []

    def _get_hybrid_seeds(
        self,
        h_q: np.ndarray,
        anchors: List[str],
        probe_embeddings: List[np.ndarray] = None,
        probe_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        vdb_results = self._vdb.query(h_q, top_k=self.top_k_seeds * 3)
        sem_scores = {r["id"]: float(r.get("similarity", 0.0)) for r in vdb_results if r.get("id")}
        
        # HyPE: multi-probe recall with max-pool blending
        if probe_embeddings:
            probe_score_maps = []
            for pv in probe_embeddings:
                res = self._vdb.query(pv, top_k=self.top_k_seeds * 2)
                probe_score_maps.append({
                    r["id"]: float(r.get("similarity", 0.0))
                    for r in res if r.get("id")
                })
            
            all_probe_ids = set().union(*[m.keys() for m in probe_score_maps])
            for nid in all_probe_ids:
                # Max-pool: take the best-matching probe score for each node
                best_probe_score = max(m.get(nid, 0.0) for m in probe_score_maps)
                if nid in sem_scores:
                    # Blend original query score with best probe score
                    sem_scores[nid] = (
                        (1 - probe_weight) * sem_scores[nid]
                        + probe_weight * best_probe_score
                    )
                else:
                    # New node surfaced only by probes (discounted)
                    sem_scores[nid] = probe_weight * best_probe_score

        anchor_tokens = {a.lower() for a in anchors}
        anchor_idfs = self._compute_anchor_idfs(anchor_tokens)
        
        candidate_nodes = set(sem_scores.keys())
        for node_id, lemma_text in self._node_lemma_cache.items():
            if any(a in lemma_text for a in anchor_tokens):
                candidate_nodes.add(node_id)
                
        scored = []
        for node_id in candidate_nodes:
            if not self._graph.has_node(node_id): continue
            s_sem = sem_scores.get(node_id, 0.0)
            s_lex = self._lexical_score_idf(node_id, anchor_tokens, anchor_idfs)
            # Simple weighting: 70% vector, 30% keyword
            score = 0.7 * s_sem + 0.3 * s_lex
            scored.append((node_id, score))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k_seeds], sem_scores

    def _compute_anchor_idfs(self, anchor_tokens: Set[str]) -> Dict[str, float]:
        N = len(self._node_lemma_cache) or 1
        idfs = {}
        for a in anchor_tokens:
            df = sum(1 for text in self._node_lemma_cache.values() if a in text)
            idfs[a] = math.log((N - df + 0.5) / (max(1, df) + 0.5) + 1.0)
        return idfs

    def _lexical_score_idf(self, node_id: str, anchor_tokens: Set[str], anchor_idfs: Dict[str, float]) -> float:
        if not anchor_tokens: return 0.0
        lemma_text = self._node_lemma_cache.get(node_id, "")
        max_possible_score = sum(anchor_idfs.values())
        if max_possible_score == 0: return 0.0
        
        score = sum(anchor_idfs[a] for a in anchor_tokens if a in lemma_text)
        return score / max_possible_score

    def _build_lemma_cache(self) -> None:
        nodes = list(self._graph.nodes(data=True))
        texts = [data.get("description") or nid for nid, data in nodes]
        node_ids = [nid for nid, _ in nodes]
        
        nlp = self._ner._nlp
        for node_id, doc in zip(node_ids, nlp.pipe(texts, batch_size=64)):
            lemma_tokens = [t.lemma_.lower() if not t.is_stop and len(t.text) > 1 else t.text.lower() for t in doc]
            self._node_lemma_cache[node_id] = " ".join(lemma_tokens)

    # --- Step 5: Island-Level Scoring & Filtering ---
    def _score_and_filter_islands(
        self,
        islands: List[Set[str]],
        sem_scores: Dict[str, float],
        top_k: int
    ) -> List[Set[str]]:
        """
        Score each island by the average sem_score of its SEED NODES.
        sem_scores only contains the top-K candidate nodes from hybrid search
        (query + HyPE probes). BFS-expanded nodes are not in sem_scores.
        We use the seed nodes (the 'anchor' that caused this island to be created)
        as a proxy for the island's relevance to the query.
        Island_Score = mean of sem_scores for all nodes in island that are seeds.
        Islands with no seed nodes (impossible in practice) score 0.
        """
        def _island_score(island: Set[str]) -> float:
            seed_scores = [sem_scores[nid] for nid in island if nid in sem_scores]
            return float(np.mean(seed_scores)) if seed_scores else 0.0

        scored_islands = [(isl, _island_score(isl)) for isl in islands]
        scored_islands.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            f"Island filter: {len(islands)} -> top {top_k} | "
            f"scores: {[f'{s:.3f}' for _, s in scored_islands]}"
        )
        return [isl for isl, _ in scored_islands[:top_k]]


    # --- Step 4: Strong Edge BFS ---
    def _build_strong_islands(self, seed_node_ids: List[str]) -> List[Set[str]]:
        raw_islands = []
        for seed in seed_node_ids:
            if not self._graph.has_node(seed): continue
            
            visited = {seed}
            queue = [(seed, 0)]
            while queue:
                curr, depth = queue.pop(0)
                if depth >= self.max_bfs_depth: continue
                
                # Traverse ALL edges regardless of type
                neighbors = list(self._graph.successors(curr)) + list(self._graph.predecessors(curr))
                for nxt in neighbors:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, depth + 1))
            raw_islands.append(visited)
            
        # Merge overlapping islands
        merged = []
        for isl in raw_islands:
            absorbed = False
            for existing in merged:
                if existing & isl:
                    existing |= isl
                    absorbed = True
                    break
            if not absorbed:
                merged.append(set(isl))
        return merged[:12] # Cap at 12 islands to prevent explosion

    # --- Step 5: Weak Bridging & Assembly ---
    def _assemble_and_bridge(self, islands: List[Set[str]], bridging_budget: int) -> Set[str]:
        all_nodes = set()
        for isl in islands:
            all_nodes |= isl
            
        if bridging_budget <= 0:
            return all_nodes
            
        from itertools import combinations
        # Add weak bridge nodes if budget allows
        for idx_a, idx_b in combinations(range(len(islands)), 2):
            score, intermediates = self._weak.island_bridge_score(islands[idx_a], islands[idx_b], bridging_budget)
            if score > 0:
                all_nodes.update(intermediates)
                
        return all_nodes

    # --- Step 6: Format Context ---
    def _format_context(self, nodes: Set[str]) -> str:
        # Sort nodes sequentially to provide readable text to LLM
        def get_source_idx(nid):
            # Try to sort by source chunk index if available
            try: return int(nid.split('-')[-1])
            except: return 999999
            
        sorted_nodes = sorted(list(nodes), key=get_source_idx)
        
        lines = []
        for nid in sorted_nodes:
            if self._graph.has_node(nid):
                desc = self._graph.nodes[nid].get("description", "")
                if desc: lines.append(f"- {desc}")
                
        return "\n".join(lines)
