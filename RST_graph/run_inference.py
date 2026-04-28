import os
import sys
import json
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.main import RSTRAG

logging.basicConfig(level=logging.WARNING)

async def test_one_question():
    workspace_dir = "/Volumes/Untitled/GraphRAG-Benchmark/RST_graph/rst_workspace/Novel-30752"
    print(f"Initializing RST_graph from {workspace_dir} ...")
    rag = RSTRAG(workspace_dir)
    
    # Run a test question
    question = "Who did Alex talk to about the anomaly?"
    print(f"\n[Test Question]: {question}")
    print("-" * 50)
    
    # We set bridging_budget=3 to allow weak connections
    answer, context = await rag.query(question, bridging_budget=3)
    
    print("\n[Retrieved Context]:")
    print(context)
    print("\n[Generated Answer]:")
    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_one_question())
