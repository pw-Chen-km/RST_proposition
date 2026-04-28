import logging
import os
import networkx as nx
from openai import AsyncOpenAI
import numpy as np

from .storage import GraphStorage, VectorStorage, DocumentStorage
from .weak_index import WeakAdjacencyIndex
from .ner import PropositionNER
from .retriever import RSTRetriever
from .local_embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts, embedding_dimension

logger = logging.getLogger(__name__)

class RSTRAG:
    """
    The main interface for RST_graph (Representative Semantic & Topology Graph).
    """
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.client = AsyncOpenAI()
        self.chat_model = os.getenv("RST_CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("RST_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        
        # Load Storages
        self.graph_storage = GraphStorage(workspace_dir)
        self.vector_storage = VectorStorage(workspace_dir, name="vector_store")
        self.doc_storage = DocumentStorage(workspace_dir)

        
        # Load Weak Index
        self.weak_index = WeakAdjacencyIndex()
        if not self.weak_index.load(workspace_dir):
            logger.warning("Weak Index not found. Creating empty.")
            
        # Init NER
        self.ner = PropositionNER()
        
        # Init Retriever
        self.retriever = RSTRetriever(
            graph=self.graph_storage.graph,
            vector_store=self.vector_storage,
            weak_index=self.weak_index,
            ner=self.ner,
            llm_func=self.llm_func,
            embed_func=self.embed_func
        )

    async def llm_func(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            res = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.0
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    async def embed_func(self, texts: list) -> list:
        if not texts: return []
        try:
            return embed_texts(texts, self.embedding_model)
        except:
            return [np.zeros(embedding_dimension(self.embedding_model)) for _ in texts]

    async def query(self, question: str, bridging_budget: int = 3, use_expansion: bool = False, probe_weight: float = 0.5, top_k_islands: int = 0) -> str:
        """End-to-end question answering pipeline."""
        context = await self.retriever.retrieve(question, bridging_budget, use_expansion, probe_weight, top_k_islands)
        
        prompt = (
            "You are a reading-comprehension assistant grounded in structured evidence.\n\n"
            f"User Question: {question}\n\n"
            "Below is the retrieved structured evidence:\n\n"
            f"{context}\n\n"
            "Based ONLY on the evidence above, provide a concise and accurate answer.\n"
            "If the evidence is insufficient, say \"I do not have enough information to answer.\"\n"
        )
        answer = await self.llm_func(prompt)
        return answer, context
