import json
import re
from typing import Any, Dict, Optional


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first balanced JSON object from text.

    Handles nested braces and braces inside string literals, so a stray
    natural-language prefix/suffix does not defeat parsing.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
    return None


def parse_json_output(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from model output.

    Resolution order:
      1. A <result>...</result> tagged block, if present (models are prompted
         to wrap their answer this way).
      2. The raw content after stripping a Markdown code fence / BOM.
      3. The first balanced JSON object anywhere in the content (fallback).

    Args:
        content: Raw output content from the model

    Returns:
        Parsed dictionary on success, None on failure
    """
    if not content or not content.strip():
        raise ValueError("Empty or whitespace-only content provided for JSON parsing")

    cleaned = content.strip().lstrip("﻿")

    # 1) Prefer the <result>...</result> tagged block.
    tag_match = re.search(r"<result>\s*(.*?)\s*</result>", cleaned, flags=re.DOTALL)
    if tag_match:
        cleaned = tag_match.group(1).strip()

    # 2) Strip a surrounding Markdown code fence, if any.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 3) Fallback: pull the first balanced JSON object out of the text.
        extracted = _extract_json_object(cleaned)
        if extracted is not None:
            return extracted
        raise ValueError(
            f"Failed to parse JSON from content. Error: no valid JSON object found. "
            f"Content: {cleaned}"
        )
