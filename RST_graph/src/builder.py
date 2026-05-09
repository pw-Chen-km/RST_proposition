import hashlib
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Union

from openai import AsyncOpenAI

from .extractor import extract_propositions
from .local_embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from .storage import DocumentStorage, GraphStorage, VectorStorage
from .weak_index import WeakAdjacencyIndex

logger = logging.getLogger(__name__)

ExtractorFunc = Callable[[str], Awaitable[Dict[str, List[Dict[str, Any]]]]]
EmbeddingFunc = Callable[[List[str]], Awaitable[List[List[float]]]]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    i = 0
    step = max(1, chunk_size - chunk_overlap)
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


class RSTGraphBuilder:
    """
    Build the graph workspace consumed by RSTRAG.

    The persisted workspace contains graph.graphml, vector_store.json,
    weak_index.json, and doc_chunks.json.
    """

    def __init__(
        self,
        workspace_dir: str,
        proposition_extractor: ExtractorFunc = None,
        embedding_func: EmbeddingFunc = None,
        ner=None,
    ):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

        self.client = AsyncOpenAI()
        self.extract_model = os.getenv("PROPOSITION_EXTRACT_MODEL", "gpt-4o")
        self.embedding_model = os.getenv("RST_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.proposition_extractor = proposition_extractor
        self.embedding_func = embedding_func
        self.ner = ner

        self.graph_storage = GraphStorage(workspace_dir, "graph")
        self.vector_storage = VectorStorage(workspace_dir, "vector_store")
        self.doc_storage = DocumentStorage(workspace_dir, "doc_chunks")

    async def extract(self, text_chunk: str) -> Dict[str, List[Dict[str, Any]]]:
        if self.proposition_extractor is not None:
            return await self.proposition_extractor(text_chunk)
        return await extract_propositions(text_chunk, self.client, model=self.extract_model)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.embedding_func is not None:
            return await self.embedding_func(texts)

        return [vector.tolist() for vector in embed_texts(texts, self.embedding_model)]

    async def ainsert(self, texts: Union[str, List[str]]) -> None:
        if isinstance(texts, str):
            texts = [texts]

        chunks: List[str] = []
        for text in texts:
            chunks.extend(split_text(text))
        total_chunks = len(chunks)
        print(
            f"[RST build] 建圖中：{total_chunks} 個 chunk；完成前只存在記憶體，最後一次寫入 "
            f"graph / vector_store / doc_chunks",
            flush=True,
        )

        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            short_id = chunk_id[:8]
            if self.doc_storage.get(chunk_id):
                print(f"[RST build] {idx}/{total_chunks} 已快取，略過 ({short_id})", flush=True)
                continue

            print(
                f"[RST build] {idx}/{total_chunks} 正在呼叫 LLM 抽取命題… ({short_id})",
                flush=True,
            )
            self.doc_storage.upsert(chunk_id, {"content": chunk})

            extraction = await self.extract(chunk)
            nodes = extraction.get("nodes", [])
            edges = extraction.get("edges", [])
            if not nodes:
                print(
                    f"[RST build] {idx}/{total_chunks} LLM 未回傳節點，略過寫圖 ({short_id})",
                    flush=True,
                )
                continue

            node_names = [node["entity_name"] for node in nodes]
            embeddings = []
            for i in range(0, len(node_names), 50):
                embeddings.extend(await self.embed(node_names[i : i + 50]))

            for i, node in enumerate(nodes):
                node_name = node["entity_name"]
                self.graph_storage.upsert_node(
                    node_name,
                    {
                        "entity_type": node["entity_type"],
                        "description": node.get("description", ""),
                        "source_id": chunk_id,
                    },
                )
                if i < len(embeddings) and embeddings[i]:
                    self.vector_storage.upsert(
                        {
                            "id": node_name,
                            "content": node_name,
                            "entity_type": node["entity_type"],
                            "embedding": embeddings[i],
                        }
                    )

            for edge in edges:
                self.graph_storage.upsert_edge(
                    edge["source"],
                    edge["target"],
                    {
                        "keywords": edge.get("keywords", ""),
                        "weight": edge.get("weight", 1.0),
                        "description": edge.get("description", ""),
                        "source_id": chunk_id,
                    },
                )

            print(
                f"[RST build] {idx}/{total_chunks} 已寫入圖（nodes={len(nodes)} edges={len(edges)}）",
                flush=True,
            )

        print("[RST build] 正在將 workspace 固化到磁碟（graphml / json / weak_index）…", flush=True)
        self.doc_storage.save()
        self.graph_storage.save()
        self.vector_storage.save()

        weak_index = WeakAdjacencyIndex()
        weak_index.build(self.graph_storage.graph, ner=self.ner)
        weak_index.save(self.workspace_dir)
        logger.info("RST graph workspace built at %s", self.workspace_dir)
        print(f"[RST build] 建圖完成：{self.workspace_dir}", flush=True)
