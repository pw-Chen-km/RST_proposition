"""
Proposition-Aware NER using spaCy (en_core_web_sm).

Adapted from LinearRAG/src/ner.py with three additions:
  - extract_from_node(text)  → entities for a single proposition node
  - extract_from_query(text) → query anchors (A_q)
  - build_node_entity_map(graph) → corpus-level {node_id: [entity, ...]}
    used downstream to build the WeakAdjacencyIndex.
"""

import logging
from typing import Dict, List

import networkx as nx

logger = logging.getLogger(__name__)

# Entity labels to ignore (numerics that don't help bridging)
_SKIP_LABELS = {"ORDINAL", "CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"}

# Query framework non-informative words that shouldn't be used as lexical anchors
_QUERY_STOP_WORDS = {
    "narrative", "text", "context", "detail", "evidence", "account", 
    "novel", "description", "mention", "story", "paragraph", "passage", 
    "information", "describe", "discuss", "book", "author", "document"
}


class PropositionNER:
    """
    Lightweight spaCy NER wrapper for proposition-centric graphs.

    Args:
        model_name: spaCy model to load.  Defaults to 'en_core_web_sm'.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
            self._nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Run:  python -m spacy download {model_name}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_node(self, text: str) -> List[str]:
        """
        Extract and normalise named entities from a single proposition node text.

        Returns a deduplicated list of lower-cased entity strings,
        filtering out numeric / temporal labels.
        """
        return self._extract(text)

    def extract_from_query(self, query: str) -> List[str]:
        """
        Extract query anchors (A_q) from the user query.

        Normalises and filters out both basic stopwords and question framework 
        words like "text", "narrative", "context".
        """
        extracted = self._extract(query)
        # Filter out framework noise
        return [w for w in extracted if w not in _QUERY_STOP_WORDS]

    def build_node_entity_map(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """
        Run NER over every proposition node in the graph.

        Reads the ``description`` attribute of each node (falling back to
        the node id itself if absent) and returns a mapping:

            { node_id: [entity_1, entity_2, ...] }

        This map is consumed by WeakAdjacencyIndex.build() to construct
        the global shared-entity connectivity layer A_w.
        """
        node_entity_map: Dict[str, List[str]] = {}
        nodes = list(graph.nodes(data=True))

        # Batch processing via spaCy pipe for efficiency
        texts = []
        node_ids = []
        for node_id, data in nodes:
            text = data.get("description") or node_id
            texts.append(text)
            node_ids.append(node_id)

        logger.info(f"Running NER over {len(texts)} proposition nodes …")
        for node_id, doc in zip(node_ids, self._nlp.pipe(texts, batch_size=64)):
            entities = self._extract_from_doc(doc)
            node_entity_map[node_id] = entities

        logger.info("NER complete.")
        return node_entity_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, text: str) -> List[str]:
        doc = self._nlp(text)
        return self._extract_from_doc(doc)

    def _extract_from_doc(self, doc) -> List[str]:
        seen: set = set()
        result: List[str] = []
        for ent in doc.ents:
            if ent.label_ in _SKIP_LABELS:
                continue
            normalised = ent.text.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(normalised)
                
        # 2. Extract key Nouns and Proper Nouns directly
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and len(token.text) > 2:
                normalised = token.lemma_.strip().lower()
                if normalised and normalised not in seen:
                    seen.add(normalised)
                    result.append(normalised)
                    
        return result
