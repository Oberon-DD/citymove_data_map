#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Optimized Synthetic Generation for Short Urban Dataset Descriptions

Specifically calibrated for:
- Median length: 93 characters (14 words)
- P95 length: 404 characters (101 tokens)
- Very concise dataset titles and descriptions
- Global Action Plan on Physical Activity categories

Author: Ultra-optimized for short texts
Date: 2026-01-31
"""

import requests
import re
import logging
from typing import List, Tuple, Optional, Dict
import time
import numpy as np

from sentence_transformers import SentenceTransformer

# =========================
# CONFIGURATION
# =========================

LLM_BASE_URL = 'http://localhost:8080'

# Label-specific contexts adapted for SHORT texts
LABEL_CONTEXTS = {
    "Air Quality": (
        "Create ONE concise variant (1-2 sentences) of this air quality dataset description. "
        "Use synonyms: pollution→emissions, monitoring→measuring, sensor→detector, ambient→atmospheric. "
        "Keep it brief and technical."
    ),
    
    "Transport PA": (
        "Create ONE concise variant (1-2 sentences) of this active transport dataset. "
        "Use synonyms: walking→pedestrian, cycling→biking, trips→journeys, commute→travel. "
        "Keep it brief."
    ),
    
    "Recreational PA": (
        "Create ONE concise variant (1-2 sentences) of this recreational activity dataset. "
        "Use synonyms: exercise→physical activity, sports→athletics, leisure→recreation. "
        "Keep it brief."
    ),
    
    "Total PA": (
        "Create ONE concise variant (1-2 sentences) of this physical activity dataset. "
        "Use synonyms: activity→movement, sedentary→inactive, levels→rates. "
        "Keep it brief."
    ),
    
    "Recreational and Sports Environment": (
        "Create ONE concise variant (1-2 sentences) of this sports/recreation facility dataset. "
        "Use synonyms: parks→green spaces, facilities→amenities, playground→play area. "
        "Keep it brief."
    ),
    
    "Active Transport Environment": (
        "Create ONE concise variant (1-2 sentences) of this transport infrastructure dataset. "
        "Use synonyms: bike lanes→cycling paths, sidewalks→footpaths, walkability→pedestrian access. "
        "Keep it brief."
    ),
    
    "Demographics & SES": (
        "Create ONE concise variant (1-2 sentences) of this demographic dataset. "
        "Use synonyms: population→residents, income→earnings, employment→occupation. "
        "Keep it brief."
    ),
    
    "NCD's": (
        "Create ONE concise variant (1-2 sentences) of this health/disease dataset. "
        "Use synonyms: disease→illness, prevalence→occurrence, mortality→death rate. "
        "Keep it brief."
    ),
    
    "Other": (
        "Create ONE concise variant (1-2 sentences) of this urban dataset description. "
        "Rephrase while maintaining key information. Keep it brief."
    ),
}

DEFAULT_CONTEXT = (
    "Create ONE concise variant (1-2 sentences) of this dataset description. "
    "Rephrase using synonyms. Keep it brief and informative."
)

# LLM parameters optimized for SHORT outputs (median 93 chars, p95 404 chars)
LLM_PARAMS = {
    'temperature': 0.8,        # Higher for diversity in short texts
    'top_k': 50,               # More options for variety
    'top_p': 0.92,
    'n_predict': 80,           # 80 tokens ≈ 320 chars (good for median 93, max ~400)
    'repeat_penalty': 1.25,    # High penalty - avoid repetition in short texts
    'frequency_penalty': 0.5,  # Additional anti-repetition
    'stop': [
        "</s>",
        "Assistant:",
        "User:",
        "\n\n",                # Any double newline
        "\nVariant",
        "\n1.", "\n2.", "\n3.",
        "\n-", "\n*", "\n•",
        "###",
        "Here",                # "Here is..." / "Here's..."
        "Another",
        "Alternatively",
        "Example:",
        "Variant:",
        "Note:",               # "Note: ..." at end
        "Description:",        # Meta-labeling
    ],
}

# Generation settings for short texts
MAX_RETRIES = 3
REQUEST_TIMEOUT = 20           # Shorter timeout for short generations
MIN_OUTPUT_LENGTH = 20         # Minimum 20 chars (very short but valid)
MAX_OUTPUT_LENGTH = 500        # Hard cap at 500 chars
MIN_SIMILARITY = 0.25          # Lower for short texts (less word overlap)
MAX_SIMILARITY = 0.88          # Slightly higher threshold for very short texts

# Embedding model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# =========================
# GLOBAL EMBEDDING MODEL
# =========================

try:
    logger = logging.getLogger('synthetic_generation')
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    EMBEDDING_MODEL = None

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('synthetic_generation')

# =========================
# UTILITY FUNCTIONS
# =========================

def semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using sentence embeddings."""
    if EMBEDDING_MODEL is None:
        logger.warning("Embedding model not loaded, falling back to Jaccard")
        return jaccard_similarity(text1, text2)
    
    try:
        emb1 = EMBEDDING_MODEL.encode(text1, convert_to_tensor=False)
        emb2 = EMBEDDING_MODEL.encode(text2, convert_to_tensor=False)
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


def jaccard_similarity(text1: str, text2: str) -> float:
    """Fallback: Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0


def clean_llm_output(text: str) -> str:
    """
    Aggressively clean LLM output for very short texts.
    """
    if not text or not isinstance(text, str):
        return ""
    
    t = text.strip()
    
    # Remove chat markers
    for marker in ["<|assistant|>", "<|user|>", "Assistant:", "User:"]:
        t = t.replace(marker, "")
    t = t.strip()
    
    # Remove meta-discussion patterns
    meta_patterns = [
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
    for pattern in meta_patterns:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE).strip()
    
    # For short texts, take only first line
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    if not lines:
        return ""
    
    # If multiple lines, concatenate first two if both are short
    if len(lines) == 1:
        t = lines[0]
    elif len(lines) >= 2 and len(lines[0]) < 50:
        # First line might be title, second might be description
        t = lines[0] + " - " + lines[1]
    else:
        t = lines[0]
    
    # Remove quotes that might wrap the output
    t = t.strip('"\'')
    
    # Failure indicators
    if t.upper() in ["NONE", "N/A", "UNKNOWN", "ERROR", "NULL", "[EMPTY]", "..."]:
        return ""
    
    # Length validation
    if len(t) < MIN_OUTPUT_LENGTH:
        logger.debug(f"Output too short ({len(t)} chars): {t}")
        return ""
    
    if len(t) > MAX_OUTPUT_LENGTH:
        logger.debug(f"Output too long ({len(t)} chars), truncating")
        # Truncate at last sentence
        sentences = re.split(r'[.!?]\s+', t[:MAX_OUTPUT_LENGTH])
        if len(sentences) > 1:
            t = '. '.join(sentences[:-1]) + '.'
        else:
            t = t[:MAX_OUTPUT_LENGTH].strip()
    
    return t


def detect_list_output(text: str) -> bool:
    """Detect if LLM generated a list."""
    # Multiple numbered items
    if len(re.findall(r'\n\s*\d+[\.\)]\s+', text)) >= 2:
        return True
    # Multiple bullet points
    if len(re.findall(r'\n\s*[\-\*\•]\s+', text)) >= 2:
        return True
    # "Variant 1", "Variant 2"
    if len(re.findall(r'variant\s+\d+', text, re.IGNORECASE)) >= 2:
        return True
    # Multiple lines that all start with capital letters (list items)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 3:
        capital_starts = sum(1 for l in lines if l[0].isupper())
        if capital_starts >= 3:
            return True
    return False


# =========================
# CORE GENERATION
# =========================

def get_context_for_label(label: str) -> str:
    """Get label-specific context."""
    label_normalized = label.strip()
    
    # Exact match
    if label_normalized in LABEL_CONTEXTS:
        return LABEL_CONTEXTS[label_normalized]
    
    # Case-insensitive
    for key in LABEL_CONTEXTS:
        if key.lower() == label_normalized.lower():
            return LABEL_CONTEXTS[key]
    
    logger.debug(f"No specific context for '{label}', using default")
    return DEFAULT_CONTEXT


def generate_synthetic_text_via_llm(
    context: str,
    user_input: str,
    base_url: str = LLM_BASE_URL,
    **override_params
) -> Optional[str]:
    """Call local LLM to generate ONE synthetic variant."""
    
    # Ultra-strict prompt for short outputs
    prompt = (
        f"{context}\n\n"
        f"Original: {user_input}\n\n"
        f"Variant:"  # Simple, direct prompt
    )
    
    params = {**LLM_PARAMS, **override_params}
    params['prompt'] = prompt
    
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f'{base_url}/completion',
                json=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'content' in result:
                    return result['content'].strip()
                else:
                    logger.warning(f"Unexpected API response: {result}")
                    return None
            else:
                logger.warning(f"API status {response.status_code} (attempt {attempt}/{MAX_RETRIES})")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout (attempt {attempt}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to {base_url}")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
        
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    
    return None


def generate_synthetic_data(
    doc: str,
    label: str,
    num_augmentations: int = 5,
    context: Optional[str] = None,
    max_attempts_multiplier: int = 4
) -> Tuple[List[str], List[str]]:
    """
    Generate synthetic variants for SHORT urban dataset descriptions.
    
    Args:
        doc: Original text (typically 50-400 chars)
        label: Classification label
        num_augmentations: Number of variants
        context: Optional context override
        max_attempts_multiplier: Max attempts = num_augmentations * this
        
    Returns:
        (synthetic_docs, synthetic_labels)
    """
    synthetic_docs: List[str] = []
    synthetic_labels: List[str] = []
    
    if context is None:
        context = get_context_for_label(label)
    
    max_attempts = num_augmentations * max_attempts_multiplier
    attempts = 0
    list_outputs = 0
    too_similar = 0
    too_different = 0
    
    logger.info(f"Generating {num_augmentations} variants for '{label}' (original: {len(doc)} chars)")
    
    while len(synthetic_docs) < num_augmentations and attempts < max_attempts:
        attempts += 1
        
        raw_output = generate_synthetic_text_via_llm(context, doc)
        
        if raw_output is None:
            continue
        
        # Detect lists BEFORE cleaning
        if detect_list_output(raw_output):
            list_outputs += 1
            logger.debug(f"List detected (attempt {attempts})")
            # Extract first item
            lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
            if lines:
                raw_output = lines[0]
        
        cleaned = clean_llm_output(raw_output)
        
        if not cleaned:
            continue
        
        # Semantic similarity
        similarity = semantic_similarity(doc, cleaned)
        
        if similarity < MIN_SIMILARITY:
            too_different += 1
            logger.debug(f"Too different (sim={similarity:.3f}): {cleaned[:50]}")
            continue
        
        if similarity > MAX_SIMILARITY:
            too_similar += 1
            logger.debug(f"Too similar (sim={similarity:.3f}): {cleaned[:50]}")
            continue
        
        # Exact duplicates
        if cleaned == doc or cleaned in synthetic_docs:
            continue
        
        # SUCCESS
        synthetic_docs.append(cleaned)
        synthetic_labels.append(label)
        logger.debug(
            f"✓ Variant {len(synthetic_docs)}/{num_augmentations} "
            f"(sim={similarity:.3f}, {len(cleaned)} chars)"
        )
    
    # Report
    success_rate = len(synthetic_docs) / attempts if attempts > 0 else 0
    
    if len(synthetic_docs) < num_augmentations:
        logger.warning(
            f"Generated {len(synthetic_docs)}/{num_augmentations} variants "
            f"(success rate: {success_rate:.1%}, attempts: {attempts})"
        )
    else:
        logger.info(f"✓ Generated {len(synthetic_docs)} variants (success: {success_rate:.1%})")
    
    if list_outputs > 0:
        logger.warning(f"⚠ Lists: {list_outputs}/{attempts} ({list_outputs/attempts:.1%})")
    
    if too_similar > 5:
        logger.info(f"ℹ Too similar: {too_similar} (consider lowering MAX_SIMILARITY)")
    
    if too_different > 5:
        logger.info(f"ℹ Too different: {too_different} (consider raising MIN_SIMILARITY)")
    
    return synthetic_docs, synthetic_labels


# =========================
# TESTING
# =========================

def test_generation():
    """Test with real examples from your data."""
    
    # Real examples from your length analysis
    test_cases = [
        {
            'doc': "Presence and coverage of bike share programs",
            'label': "Active Transport Environment",
            'note': "10th percentile (44 chars)"
        },
        {
            'doc': "Income per inhabitant 2008-2019 (b): These are the figures from the municipal budgets.",
            'label': "Demographics & SES",
            'note': "50th percentile (86 chars)"
        },
        {
            'doc': "Metadata Record of collection and transportation of solid waste of the District Municipality of Pueblo Libre - [MPL]: This dataset contains detailed information about the record of collection and transportation of solid waste carried out by the District Municipality of Pueblo Libre.",
            'label': "Other",
            'note': "90th percentile (283 chars)"
        },
        {
            'doc': "Air quality monitoring stations measuring PM2.5 and NO2 levels",
            'label': "Air Quality",
            'note': "Air quality example"
        },
        {
            'doc': "Park visitor counts from recreational facilities",
            'label': "Recreational and Sports Environment",
            'note': "Recreation example"
        },
    ]
    
    print("\n" + "="*80)
    print("SYNTHETIC GENERATION TEST - Real Urban Dataset Examples")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        doc = test_case['doc']
        label = test_case['label']
        note = test_case['note']
        
        print(f"\n{'─'*80}")
        print(f"TEST {i}: {label} ({note})")
        print(f"{'─'*80}")
        print(f"\nOriginal ({len(doc)} chars):")
        print(f"  {doc}")
        print(f"\nGenerating 3 variants...")
        
        synthetic_docs, _ = generate_synthetic_data(doc, label, num_augmentations=3)
        
        if synthetic_docs:
            print()
            for j, syn in enumerate(synthetic_docs, 1):
                sim = semantic_similarity(doc, syn)
                print(f"  {j}. (sim={sim:.3f}, {len(syn)} chars)")
                print(f"     {syn}")
        else:
            print("\n  ⚠ No variants generated")
        
        print()
    
    print("="*80 + "\n")


if __name__ == "__main__":
    test_generation()