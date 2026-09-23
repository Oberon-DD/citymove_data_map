#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CITY-MOVE | Synthetic Text Generation (Simplified)

Generates neutral paraphrases of short urban dataset descriptions
via a locally deployed Mistral-7B-Instruct-v0.1 (Q8 quantized, llama.cpp).

No label-based colouring — the LLM just rephrases the text as-is.
No similarity filtering — all valid outputs are accepted.
Each generated variant is printed to stdout for inspection.

Author: Oberon
Date: 2026-02
"""

import requests
import re
import logging
import time
from typing import List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

LLM_BASE_URL = "http://localhost:8080"

GENERATION_PROMPT = (
    "Create ONE concise variant (1-2 sentences) of this dataset description. "
    "Rephrase using different words but keep the same meaning. "
    "Output ONLY the variant, nothing else."
)

LLM_PARAMS = {
    "temperature": 0.8, # TRY 0.6? 
    "top_k": 50,
    "top_p": 0.92,
    "n_predict": 80,
    "repeat_penalty": 1.25,
    "frequency_penalty": 0.5,
    "stop": [
        "</s>", "Assistant:", "User:", "\n\n",
        "\nVariant", "\n1.", "\n2.", "\n3.",
        "\n-", "\n*", "\n•", "###", "<br>",
        "Here", "Another", "Alternatively",
        "Example:", "Variant:", "Note:", "Description:", 
        "<|endofvariantassistant<", "<|end|>"
    ],
}

MAX_RETRIES = 3
REQUEST_TIMEOUT = 20
MIN_OUTPUT_LENGTH = 20
MAX_OUTPUT_LENGTH = 500


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("synthetic_generation")


# =============================================================================
# CLEANING
# =============================================================================

_META_PATTERNS = [
    r"^Here'?s?\s+(a\s+)?(concise\s+)?(short\s+)?(variant|example|version):?\s*",
    r"^Variant\s*\d*:?\s*",
    r"^Example\s*\d*:?\s*",
    r"^Alternative\s*\d*:?\s*",
    r"^Rephrased\s*:?\s*",
    r"^Brief\s+version:?\s*",
    r"^Short\s+version:?\s*",
    r"^\d+[\.\)]\s+",
    r"^[\-\*\•]\s+",
]

_FAILURE_TOKENS = {"NONE", "N/A", "UNKNOWN", "ERROR", "NULL", "[EMPTY]", "..."}


def _strip_meta_prefixes(text: str) -> str:
    for pattern in _META_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def _first_meaningful_line(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    if len(lines[0]) < 50:
        return lines[0] + " - " + lines[1]
    return lines[0]


def _truncate_at_sentence(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    for sep in [". ", "! ", "? "]:
        idx = text.rfind(sep, 0, max_len)
        if idx > 0:
            return text[: idx + 1]
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def clean_llm_output(text: str) -> str:
    """Clean and validate raw LLM output."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()
    for marker in ["<|assistant|>", "<|user|>", "Assistant:", "User:"]:
        t = t.replace(marker, "")
    t = _strip_meta_prefixes(t.strip())
    t = _first_meaningful_line(t)
    t = t.strip("\"'")
    if t.upper() in _FAILURE_TOKENS:
        return ""
    if len(t) < MIN_OUTPUT_LENGTH:
        return ""
    if len(t) > MAX_OUTPUT_LENGTH:
        t = _truncate_at_sentence(t, MAX_OUTPUT_LENGTH)
    return t


def _is_list_output(text: str) -> bool:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    numbered = sum(1 for l in lines if re.match(r"^\d+[\.\)]", l))
    bulleted = sum(1 for l in lines if re.match(r"^[\-\*\•]", l))
    return (numbered >= 2) or (bulleted >= 2)


# =============================================================================
# LLM CALL
# =============================================================================

def _call_llm(doc: str) -> Optional[str]:
    """Call local LLM for one paraphrase attempt."""
    prompt = f"{GENERATION_PROMPT}\n\nOriginal: {doc}\n\nVariant:"
    params = {**LLM_PARAMS, "prompt": prompt}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{LLM_BASE_URL}/completion",
                json=params, headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                content = resp.json().get("content", "").strip()
                return content if content else None
            logger.warning("API status %d (attempt %d/%d)", resp.status_code, attempt, MAX_RETRIES)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to %s", LLM_BASE_URL)
            return None
        except requests.exceptions.Timeout:
            logger.warning("Timeout (attempt %d/%d)", attempt, MAX_RETRIES)
        except Exception as e:
            logger.error("LLM error: %s", e)
            return None
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def _try_generate(doc, results):
    """Attempt one LLM call, clean output, check for duplicates. Returns cleaned text or None."""
    raw = _call_llm(doc)
    if raw is None:
        return None

    # Extract first item if LLM returned a list
    if _is_list_output(raw):
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        raw = lines[0] if lines else ""

    cleaned = clean_llm_output(raw)
    if not cleaned:
        return None

    # Only filter exact duplicates
    if cleaned == doc or cleaned in results:
        return None

    return cleaned


def generate_synthetic_data(
    doc: str,
    label: str,
    num_augmentations: int = 5,
    **_kwargs,
) -> Tuple[List[str], List[str]]:
    """
    Generate synthetic paraphrases of *doc*.

    Label-neutral: the label is NOT used to colour the prompt.
    No similarity filtering: every cleaned, non-duplicate output is kept.
    Every accepted variant is printed for inspection.

    Returns (synthetic_docs, synthetic_labels).
    """
    results: List[str] = []
    max_attempts = num_augmentations * 4

    logger.info(
        "Generating %d variants for [%s] (%d chars)",
        num_augmentations, label, len(doc),
    )

    for attempt in range(1, max_attempts + 1):
        if len(results) >= num_augmentations:
            break

        cleaned = _try_generate(doc, results)
        if cleaned is None:
            continue

        results.append(cleaned)
        print(f"  SYNTH [{label}] {len(results)}/{num_augmentations}: {cleaned}")

    if len(results) < num_augmentations:
        logger.warning("Generated %d/%d variants (%d attempts)",
                        len(results), num_augmentations, attempt)
    else:
        logger.info("\u2713 Generated %d variants", len(results))

    return results, [label] * len(results)


# =============================================================================
# STANDALONE TEST
# =============================================================================

def test_generation():
    test_cases = [
        ("Presence and coverage of bike share programs", "Active Transport Environment"),
        ("Air quality monitoring stations measuring PM2.5 and NO2 levels", "Air Quality"),
        ("Park visitor counts from recreational facilities", "Recreational and Sports Environment"),
    ]

    print("\n" + "=" * 80)
    print("SYNTHETIC GENERATION TEST")
    print("=" * 80)

    for doc, label in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Original [{label}]: {doc}")
        print(f"{'─' * 80}")
        synth, _ = generate_synthetic_data(doc, label, num_augmentations=3)
        if not synth:
            print("  ⚠ No variants generated")
        print()

    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_generation()
