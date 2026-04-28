import asyncio
import os
import shutil

from RST_graph.src.builder import RSTGraphBuilder
from RST_graph.src.storage import GraphStorage, VectorStorage
from RST_graph.src.weak_index import WeakAdjacencyIndex


PASSAGE = (
    "Title: Aurora Lab\n"
    "Maya Chen calibrated the telescope at Aurora Lab. "
    "After the storm, she asked Leo Park to compare the new readings with the old logbook."
)


async def mock_extract(_text: str):
    return {
        "nodes": [
            {
                "entity_name": "Maya Chen calibrated the telescope at Aurora Lab",
                "entity_type": "event",
                "description": "Maya Chen calibrated the telescope at Aurora Lab.",
            },
            {
                "entity_name": "Maya Chen asked Leo Park to compare readings",
                "entity_type": "event",
                "description": "After the storm, Maya Chen asked Leo Park to compare the new readings with the old logbook.",
            },
        ],
        "edges": [
            {
                "source": "Maya Chen calibrated the telescope at Aurora Lab",
                "target": "Maya Chen asked Leo Park to compare readings",
                "keywords": "TEMPORAL",
                "weight": 1.0,
                "description": "Maya calibrated the telescope before asking Leo to compare readings.",
            }
        ],
    }


async def mock_embed(texts):
    vectors = []
    for index, _text in enumerate(texts):
        vector = [0.0] * 1536
        vector[index % 1536] = 1.0
        vectors.append(vector)
    return vectors


class MockNER:
    def build_node_entity_map(self, graph):
        return {
            node_id: [
                token.lower().strip(".,")
                for token in str(data.get("description", node_id)).split()
                if token.istitle()
            ]
            for node_id, data in graph.nodes(data=True)
        }


async def main():
    os.environ.setdefault("OPENAI_API_KEY", "test-key-for-local-mock")
    workspace = os.path.abspath(os.path.join("rst_workspaces", "short_passage_mock"))
    if os.path.exists(workspace):
        shutil.rmtree(workspace)

    builder = RSTGraphBuilder(
        workspace,
        proposition_extractor=mock_extract,
        embedding_func=mock_embed,
        ner=MockNER(),
    )
    await builder.ainsert(PASSAGE)

    graph_storage = GraphStorage(workspace)
    vector_storage = VectorStorage(workspace, name="vector_store")
    weak_index = WeakAdjacencyIndex()
    weak_index.load(workspace)

    print(f"workspace={workspace}")
    print(f"nodes={graph_storage.graph.number_of_nodes()}")
    print(f"edges={graph_storage.graph.number_of_edges()}")
    print(f"vectors={len(vector_storage.data)}")
    print(f"weak_pairs={len(weak_index.sparse_aw)}")


if __name__ == "__main__":
    asyncio.run(main())
