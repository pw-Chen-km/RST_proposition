import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple


DATASET_CONFIG = {
    "hotpotqa": {
        "source": "HotpotQA-1000",
        "display_name": "HotpotQA",
        "question_id_key": "_id",
    },
    "2wikimultihopqa": {
        "source": "2WikiMultiHopQA-1000",
        "display_name": "2WikiMultiHopQA",
        "question_id_key": "_id",
    },
    "musique": {
        "source": "MuSiQue-1000",
        "display_name": "MuSiQue",
        "question_id_key": "id",
    },
}


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_pooled_corpus(passages: Sequence[Dict[str, Any]]) -> str:
    sections = []
    for idx, passage in enumerate(passages):
        title = str(passage.get("title", f"passage-{idx}")).strip()
        text = str(passage.get("text", "")).strip()
        if text:
            sections.append(f"Title: {title}\n{text}")
    return "\n\n".join(sections)


def context_pairs(item: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    pairs = []
    for row in item.get("context", []):
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            title, sentences = row[0], row[1]
            pairs.append((str(title), [str(sentence).strip() for sentence in sentences]))
    return pairs


def evidence_from_supporting_facts(item: Dict[str, Any]) -> str:
    pairs = context_pairs(item)
    lookup = {
        (title, idx): sentence
        for title, sentences in pairs
        for idx, sentence in enumerate(sentences)
        if sentence
    }

    evidence = []
    for fact in item.get("supporting_facts", []):
        if isinstance(fact, (list, tuple)) and len(fact) >= 2:
            title = str(fact[0])
            sent_id = int(fact[1])
            sentence = lookup.get((title, sent_id))
            if sentence:
                evidence.append(f"{title}: {sentence}")

    if evidence:
        return " ".join(evidence)

    triples = item.get("evidences", [])
    if triples:
        return " ".join(" ".join(str(part) for part in triple) for triple in triples)
    return ""


def evidence_from_musique(item: Dict[str, Any]) -> str:
    paragraphs = item.get("paragraphs", [])
    by_idx = {p.get("idx"): p for p in paragraphs}
    supporting_idxs = {
        step.get("paragraph_support_idx")
        for step in item.get("question_decomposition", [])
        if step.get("paragraph_support_idx") is not None
    }

    evidence = []
    for paragraph in paragraphs:
        idx = paragraph.get("idx")
        if paragraph.get("is_supporting") or idx in supporting_idxs:
            title = str(paragraph.get("title", "")).strip()
            text = str(paragraph.get("paragraph_text", "")).strip()
            if text:
                evidence.append(f"{title}: {text}" if title else text)

    for idx in sorted(supporting_idxs):
        paragraph = by_idx.get(idx)
        if not paragraph:
            continue
        title = str(paragraph.get("title", "")).strip()
        text = str(paragraph.get("paragraph_text", "")).strip()
        formatted = f"{title}: {text}" if title else text
        if text and formatted not in evidence:
            evidence.append(formatted)

    return " ".join(evidence)


def convert_questions(dataset_name: str, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cfg = DATASET_CONFIG[dataset_name]
    source = cfg["source"]
    id_key = cfg["question_id_key"]
    converted = []

    for index, item in enumerate(items):
        raw_id = str(item.get(id_key) or index)
        if dataset_name == "musique":
            evidence = evidence_from_musique(item)
        else:
            evidence = evidence_from_supporting_facts(item)

        converted.append(
            {
                "id": f"{source}-{raw_id}",
                "source": source,
                "question": str(item["question"]),
                "answer": str(item.get("answer", "")),
                "question_type": "Complex Reasoning",
                "evidence": evidence,
                "dataset": cfg["display_name"],
                "original_id": raw_id,
                "hotpot_type": item.get("type"),
                "hotpot_level": item.get("level"),
            }
        )

    return converted


def convert_dataset(dataset_name: str, data_dir: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    cfg = DATASET_CONFIG[dataset_name]
    q_path = os.path.join(data_dir, f"{dataset_name}.json")
    c_path = os.path.join(data_dir, f"{dataset_name}_corpus.json")

    questions = read_json(q_path)
    corpus = read_json(c_path)

    corpus_item = {
        "corpus_name": cfg["source"],
        "context": format_pooled_corpus(corpus),
    }
    question_items = convert_questions(dataset_name, questions)
    return corpus_item, question_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare HippoRAG legacy 1,000-question multi-hop datasets in GraphRAG-Bench JSON format."
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join("HippoRAG", "reproduce", "dataset"),
        help="Directory containing hotpotqa/musique/2wikimultihopqa JSON files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "2wikimultihopqa", "musique"],
        choices=sorted(DATASET_CONFIG),
    )
    parser.add_argument(
        "--corpus_output",
        default=os.path.join("Datasets", "Corpus", "hipporag_legacy_1000.json"),
    )
    parser.add_argument(
        "--questions_output",
        default=os.path.join("Datasets", "Questions", "hipporag_legacy_1000.json"),
    )
    args = parser.parse_args()

    all_corpus = []
    all_questions = []
    for dataset_name in args.datasets:
        corpus_item, question_items = convert_dataset(dataset_name, args.data_dir)
        all_corpus.append(corpus_item)
        all_questions.extend(question_items)
        print(
            f"{dataset_name}: {len(question_items)} questions -> source {corpus_item['corpus_name']} "
            f"({len(corpus_item['context'])} corpus chars)"
        )

    write_json(args.corpus_output, all_corpus)
    write_json(args.questions_output, all_questions)
    print(f"Wrote {len(all_corpus)} corpus items to {args.corpus_output}")
    print(f"Wrote {len(all_questions)} question items to {args.questions_output}")


if __name__ == "__main__":
    main()
