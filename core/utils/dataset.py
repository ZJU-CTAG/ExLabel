"""Dataset loading helpers shared by the labelling entry points."""

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def load_split(dataset_path: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load a split from a HuggingFace dataset or a JSONL inference file.

    A standalone JSONL file is treated as the requested inference split.  This
    keeps the command-line labellers compatible with the raw benchmark export,
    which does not contain a separate ``test`` directory/split.
    """
    if str(dataset_path).lower().endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with Path(dataset_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {dataset_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected a JSON object at {dataset_path}:{line_number}"
                    )
                rows.append(row)
        return rows

    return list(load_dataset(dataset_path, split=split))
