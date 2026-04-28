import asyncio
import os
import shutil

from run_rst_datasets_experiment import (
    build_workspace_for_source,
    corpus_by_source,
    group_questions_by_source,
    safe_workspace_name,
)
from RST_graph.src.builder import RSTGraphBuilder


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


async def mock_extract(text: str):
    if "Maya" in text:
        name = "Maya reviewed the blue notebook"
        desc = "Maya reviewed the blue notebook in the archive."
    else:
        name = "Noah repaired the red radio"
        desc = "Noah repaired the red radio in the workshop."
    return {
        "nodes": [{"entity_name": name, "entity_type": "event", "description": desc}],
        "edges": [],
    }


async def mock_embed(texts):
    vectors = []
    for index, _text in enumerate(texts):
        vector = [0.0] * 1536
        vector[index % 1536] = 1.0
        vectors.append(vector)
    return vectors


async def build_mock_workspace(source, corpus_item, workspace_dir, force_rebuild):
    workspace = os.path.abspath(os.path.join(workspace_dir, safe_workspace_name(source)))
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    builder = RSTGraphBuilder(
        workspace,
        proposition_extractor=mock_extract,
        embedding_func=mock_embed,
        ner=MockNER(),
    )
    await builder.ainsert(corpus_item["context"])
    return workspace


async def main():
    os.environ.setdefault("OPENAI_API_KEY", "test-key-for-local-mock")
    corpus_items = [
        {"corpus_name": "Doc-A", "context": "Maya reviewed the blue notebook in the archive."},
        {"corpus_name": "Doc-B", "context": "Noah repaired the red radio in the workshop."},
    ]
    questions = [
        {"id": "q1", "source": "Doc-A", "question": "Who reviewed the blue notebook?"},
        {"id": "q2", "source": "Doc-B", "question": "Who repaired the red radio?"},
    ]
    grouped = group_questions_by_source(questions)
    corpus_lookup = corpus_by_source(corpus_items)
    workspace_dir = os.path.abspath(os.path.join("rst_workspaces", "routing_mock"))
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)

    workspaces = {}
    for source in grouped:
        workspaces[source] = await build_mock_workspace(source, corpus_lookup[source], workspace_dir, True)

    assert set(workspaces) == {"Doc-A", "Doc-B"}
    assert os.path.exists(os.path.join(workspaces["Doc-A"], "graph.graphml"))
    assert os.path.exists(os.path.join(workspaces["Doc-B"], "graph.graphml"))
    assert workspaces["Doc-A"] != workspaces["Doc-B"]

    print("sources=" + ",".join(sorted(grouped)))
    print("workspace_doc_a=" + workspaces["Doc-A"])
    print("workspace_doc_b=" + workspaces["Doc-B"])
    print("routing_ok=True")


if __name__ == "__main__":
    asyncio.run(main())
