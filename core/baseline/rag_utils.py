"""
Shared helpers for baseline entries (vanilla, SimpleRAG).

SimpleRAG treats the raw task input as the retrieval key and the dataset
annotation (label + notes) as the retrieval value. This module defines how
inputs are serialized into keys, how annotations become values, and how
retrieved memories are formatted into prompts.
"""

import json
from typing import Any, Dict, List

# Internal fields that must never leak into prompts or stored values
_INTERNAL_FIELDS = (
    "id", "embedding",
    "_score", "_vector_score", "_bm25_score", "_dense_score",
)

# Raw task input fields used as the retrieval key per task
INPUT_KEY_FIELDS = {
    "issue": ["problem_statement"],
    "test": ["problem_statement", "test_patch"],
}

# Ground-truth annotation label per task
ANNOTATION_LABEL = {
    "issue": "underspecified",
    "test": "reasonable",
}


def build_input_key(sample: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    Build a retrieval key from the raw task input.

    Args:
        sample: Dataset sample containing raw input fields
        task: Task type ('issue' or 'test')

    Returns:
        Key dictionary with the raw input fields for the task
    """
    if task not in INPUT_KEY_FIELDS:
        raise ValueError(f"Unknown task: {task}")
    return {field: sample.get(field, "") for field in INPUT_KEY_FIELDS[task]}


def build_annotation_value(sample: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    Build a retrieval value from the sample's ground-truth annotation.

    Args:
        sample: Dataset sample with label and notes
        task: Task type ('issue' or 'test')

    Returns:
        Value dictionary containing the label and annotator notes
    """
    label = ANNOTATION_LABEL[task]
    return {
        label: bool(sample.get(label)),
        "notes": list(sample.get("notes") or []),
    }


def is_labelled(sample: Dict[str, Any], task: str) -> bool:
    """Check whether the sample carries a ground-truth label."""
    return sample.get(ANNOTATION_LABEL[task]) is not None


def format_annotated_examples(entries: List[Dict[str, Any]], task: str) -> str:
    """
    Format retrieved memories as annotated few-shot examples for prompts.

    Args:
        entries: Retrieved KB memories (flattened key/value fields, may contain
            internal fields such as embeddings and scores)
        task: Task type ('issue' or 'test')

    Returns:
        JSON string of examples, each holding the raw input and its annotation
    """
    label = ANNOTATION_LABEL[task]
    examples = []
    for entry in entries:
        example = {k: entry.get(k) for k in INPUT_KEY_FIELDS[task] if k in entry}
        example["annotation"] = {
            label: entry.get(label),
            "notes": entry.get("notes", []),
        }
        examples.append(example)
    return json.dumps(examples, indent=2, ensure_ascii=False)


def strip_internal_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the entry without internal KB fields."""
    return {k: v for k, v in entry.items() if k not in _INTERNAL_FIELDS}
