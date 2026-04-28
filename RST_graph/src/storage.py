import os
import json
import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
    norm_a[norm_a == 0] = 1e-10
    norm_b[norm_b == 0] = 1e-10
    return np.dot(a, b.T) / (norm_a * norm_b.T)

class GraphStorage:
    def __init__(self, workspace: str, name: str = "graph"):
        self.workspace = workspace
        self.filename = os.path.join(workspace, f"{name}.graphml")
        self.graph = nx.DiGraph()
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                self.graph = nx.read_graphml(self.filename)
                logger.info(f"Loaded graph from {self.filename} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            except Exception as e:
                logger.error(f"Failed to load graphml: {e}")

    def save(self):
        # NetworkX has issues writing dict or list attributes sometimes, convert to string if necessary
        for _, data in self.graph.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    data[k] = json.dumps(v)
        for _, _, data in self.graph.edges(data=True):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    data[k] = json.dumps(v)
        nx.write_graphml(self.graph, self.filename)
        logger.info(f"Saved graph to {self.filename}")

    def upsert_node(self, node_id: str, attributes: Dict[str, Any]):
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(attributes)
        else:
            self.graph.add_node(node_id, **attributes)

    def upsert_edge(self, source: str, target: str, attributes: Dict[str, Any]):
        if self.graph.has_edge(source, target):
            self.graph.edges[source, target].update(attributes)
        else:
            self.graph.add_edge(source, target, **attributes)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.graph.nodes.get(node_id, None)

    def _get_graph(self):
        """Compatibility method for existing downstream code that expects an awaitable or direct property."""
        return self.graph


class VectorStorage:
    def __init__(self, workspace: str, name: str = "vectors"):
        self.workspace = workspace
        self.filename = os.path.join(workspace, f"{name}.json")
        self.data: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                for item in self.data:
                    # Convert arrays to numpy arrays when loading
                    if 'embedding' in item and item['embedding'] is not None:
                        item['embedding'] = np.array(item['embedding'], dtype=np.float32)
                logger.info(f"Loaded vector store {self.filename} with {len(self.data)} records")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")

    def save(self):
        if not self.data:
            return
            
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for item in self.data:
            item_copy = item.copy()
            if 'embedding' in item_copy and isinstance(item_copy['embedding'], np.ndarray):
                item_copy['embedding'] = item_copy['embedding'].tolist()
            serializable_data.append(item_copy)
            
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                
        logger.info(f"Saved vector store to {self.filename}")

    def upsert(self, data_dict: Dict[str, Any]):
        """
        data_dict should contain 'id', 'content', 'embedding' (numpy array), and any metadata.
        """
        existing_idx = next((i for i, item in enumerate(self.data) if item['id'] == data_dict['id']), None)
        if existing_idx is not None:
            self.data[existing_idx].update(data_dict)
        else:
            self.data.append(data_dict)

    def query(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.data:
            return []
        
        embeddings = np.array([item['embedding'] for item in self.data if 'embedding' in item and item['embedding'] is not None])
        valid_items = [item for item in self.data if 'embedding' in item and item['embedding'] is not None]
        
        if len(valid_items) == 0:
            return []
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        similarities = cosine_similarity(embeddings, query_embedding).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            res = valid_items[idx].copy()
            res['similarity'] = float(similarities[idx])
            del res['embedding'] # Avoid passing heavy arrays back
            results.append(res)
        return results


class DocumentStorage:
    def __init__(self, workspace: str, name: str = "documents"):
        self.workspace = workspace
        self.filename = os.path.join(workspace, f"{name}.json")
        self.data = {}
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded doc store {self.filename} with {len(self.data)} records")
            except Exception as e:
                logger.error(f"Failed to load doc store: {e}")

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def upsert(self, doc_id: str, content: Dict[str, Any]):
        self.data[doc_id] = content
        
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(doc_id)
