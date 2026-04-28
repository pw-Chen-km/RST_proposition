import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _safe_source_id(raw_id: Any) -> str:
    text = str(raw_id)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return f"HotpotQA-{text or 'unknown'}"


def _get_example_id(example: Dict[str, Any], index: int) -> str:
    return str(example.get("_id") or example.get("id") or index)


def _normalise_context(context: Any) -> List[Tuple[str, List[str]]]:
    """Return [(title, sentences), ...] for HF and local HotpotQA variants."""
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [
            (str(title), [str(sentence).strip() for sentence in sent_list])
            for title, sent_list in zip(titles, sentences)
        ]

    if isinstance(context, list):
        pairs = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sentences = item[0], item[1]
                pairs.append((str(title), [str(sentence).strip() for sentence in sentences]))
        return pairs

    raise ValueError(f"Unsupported HotpotQA context format: {type(context).__name__}")


def _normalise_supporting_facts(supporting_facts: Any) -> List[Tuple[str, int]]:
    """Return [(title, sentence_index), ...] for HF and local HotpotQA variants."""
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", supporting_facts.get("sentence_id", []))
        return [(str(title), int(sent_id)) for title, sent_id in zip(titles, sent_ids)]

    if isinstance(supporting_facts, list):
        facts = []
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                facts.append((str(item[0]), int(item[1])))
        return facts

    return []


def _format_context(context_pairs: Sequence[Tuple[str, Sequence[str]]]) -> str:
    sections = []
    for title, sentences in context_pairs:
        text = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())
        if text:
            sections.append(f"Title: {title}\n{text}")
    return "\n\n".join(sections)


def _format_evidence(
    context_pairs: Sequence[Tuple[str, Sequence[str]]],
    supporting_facts: Sequence[Tuple[str, int]],
) -> str:
    sentence_lookup = {
        (title, idx): sentence.strip()
        for title, sentences in context_pairs
        for idx, sentence in enumerate(sentences)
        if sentence and sentence.strip()
    }

    evidence_sentences = []
    for title, sent_id in supporting_facts:
        sentence = sentence_lookup.get((title, sent_id))
        if sentence:
            evidence_sentences.append(f"{title}: {sentence}")
    return " ".join(evidence_sentences)


def _iter_hf_examples(split: str) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    if split.lower() == "all":
        dataset_dict = load_dataset("hotpotqa/hotpot_qa", "distractor")
        for split_name in dataset_dict:
            for example in dataset_dict[split_name]:
                item = dict(example)
                item["_hf_split"] = split_name
                yield item
        return

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    for example in dataset:
        item = dict(example)
        item["_hf_split"] = split
        yield item


def _iter_local_examples(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    for example in data:
        yield example


def convert_examples(examples: Iterable[Dict[str, Any]], sample_size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus_items: List[Dict[str, Any]] = []
    question_items: List[Dict[str, Any]] = []

    for index, example in enumerate(examples):
        if sample_size and len(question_items) >= sample_size:
            break

        raw_id = _get_example_id(example, index)
        source_id = _safe_source_id(raw_id)
        context_pairs = _normalise_context(example["context"])
        supporting_facts = _normalise_supporting_facts(example.get("supporting_facts", []))

        corpus_items.append(
            {
                "corpus_name": source_id,
                "context": _format_context(context_pairs),
            }
        )
        question_items.append(
            {
                "id": source_id,
                "source": source_id,
                "question": str(example["question"]),
                "answer": str(example.get("answer", "")),
                "question_type": "Complex Reasoning",
                "evidence": _format_evidence(context_pairs, supporting_facts),
                "hotpot_type": example.get("type"),
                "hotpot_level": example.get("level"),
                "hotpot_split": example.get("_hf_split"),
            }
        )

    return corpus_items, question_items


def write_json(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare HotpotQA distractor data in GraphRAG-Bench JSON format."
    )
    parser.add_argument("--sample_size", type=int, default=50, help="0 means use all available examples.")
    parser.add_argument("--split", default="validation", help="HF split name, or 'all' for train + validation.")
    parser.add_argument("--local_file", default=None, help="Optional local HotpotQA JSON fallback.")
    parser.add_argument(
        "--corpus_output",
        default=os.path.join("Datasets", "Corpus", "hotpotqa_distractor_sample.json"),
    )
    parser.add_argument(
        "--questions_output",
        default=os.path.join("Datasets", "Questions", "hotpotqa_distractor_sample.json"),
    )
    args = parser.parse_args()

    if args.local_file:
        examples = _iter_local_examples(args.local_file)
    else:
        examples = _iter_hf_examples(args.split)

    corpus_items, question_items = convert_examples(examples, args.sample_size)
    write_json(args.corpus_output, corpus_items)
    write_json(args.questions_output, question_items)

    print(f"Wrote {len(corpus_items)} corpus items to {args.corpus_output}")
    print(f"Wrote {len(question_items)} question items to {args.questions_output}")
    if question_items:
        print(f"First source: {question_items[0]['source']}")


if __name__ == "__main__":
    main()
