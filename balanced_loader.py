"""
Balanced Dataset Loader for CLARITY

Provides utilities for loading balanced test/validation datasets
with equal representation across all three CLARITY labels.
"""

import random
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datasets import load_dataset


def stratified_sample(data: List[Dict], label_key: str, samples_per_label: Optional[int] = None) -> List[Dict]:
    """
    Sample data with equal representation per label.
    
    Args:
        data: List of dicts with label_key
        label_key: Key to extract label from each dict
        samples_per_label: Number of samples per label (None = use minimum count)
    
    Returns:
        Balanced list of samples
    """
    # Group by label
    label_groups = {}
    for item in data:
        label = item.get(label_key)
        if label is None:
            continue
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # Determine samples per label
    if samples_per_label is None:
        # Use minimum count across all labels
        counts = [len(items) for items in label_groups.values()]
        if not counts:
            return []
        samples_per_label = min(counts)
    
    # Sample equally from each label
    balanced_data = []
    for label, items in label_groups.items():
        # Sample without replacement
        sampled = random.sample(items, min(samples_per_label, len(items)))
        balanced_data.extend(sampled)
    
    return balanced_data


def load_balanced_test_data(
    split: str = "test",
    samples_per_label: Optional[int] = None,
    dataset_name: str = "ailsntua/QEvasion"
) -> Tuple[List[Dict], List[str]]:
    """
    Load balanced test data from QEvasion dataset.
    
    Args:
        split: Dataset split to use ("test" or "train")
        samples_per_label: Number of samples per label (None = use minimum)
        dataset_name: HuggingFace dataset name
    
    Returns:
        Tuple of (examples, labels) where examples are dicts with 'question' and 'answer'
    """
    print(f"Loading {split} split from {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(dataset.keys())}")
    
    split_data = dataset[split]
    
    # Convert to list of dicts
    examples = []
    for item in split_data:
        # Map clarity_label to CLARITY format
        clarity_label = item.get("clarity_label", "")
        
        # Map QEvasion labels to CLARITY format
        label_mapping = {
            "Clear Reply": "Direct Reply",
            "Clear Non-Reply": "Direct Non-Reply",
            "Ambivalent Reply": "Indirect",
            "Ambivalent": "Indirect",
        }
        
        mapped_label = label_mapping.get(clarity_label, clarity_label)
        
        # Only include valid CLARITY labels
        if mapped_label not in ["Direct Reply", "Direct Non-Reply", "Indirect"]:
            continue
        
        examples.append({
            "question": str(item.get("interview_question", item.get("question", ""))),
            "answer": str(item.get("interview_answer", "")),
            "clarity_label": mapped_label,
            "original_label": clarity_label
        })
    
    print(f"Loaded {len(examples)} examples from {split} split")
    
    # Show label distribution before balancing
    labels_before = [ex["clarity_label"] for ex in examples]
    label_counts = Counter(labels_before)
    print(f"Label distribution before balancing: {dict(label_counts)}")
    
    # Balance the dataset
    balanced_examples = stratified_sample(examples, "clarity_label", samples_per_label)
    
    # Show label distribution after balancing
    labels_after = [ex["clarity_label"] for ex in balanced_examples]
    label_counts_after = Counter(labels_after)
    print(f"Label distribution after balancing: {dict(label_counts_after)}")
    print(f"Total balanced samples: {len(balanced_examples)}")
    
    # Extract labels
    labels = [ex["clarity_label"] for ex in balanced_examples]
    
    return balanced_examples, labels


def load_clarity_eval_data(
    eval_file: str = "/Users/andrearachetta/Desktop/CLARITY-SemEval-2026/dataset/clarity_task_evaluation_dataset.csv"
) -> Tuple[List[Dict], List[int]]:
    """
    Load CLARITY evaluation dataset (no labels).
    
    Args:
        eval_file: Path to evaluation CSV file
    
    Returns:
        Tuple of (examples, indices) where examples are dicts with 'question' and 'answer'
    """
    print(f"Loading CLARITY evaluation dataset from {eval_file}...")
    
    try:
        df = pd.read_csv(eval_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    examples = []
    indices = []
    
    for idx, row in df.iterrows():
        question = str(row.get("interview_question", row.get("question", ""))).strip()
        answer = str(row.get("interview_answer", "")).strip()
        
        if question and answer:
            examples.append({
                "question": question,
                "answer": answer,
                "index": idx
            })
            indices.append(idx)
    
    print(f"Loaded {len(examples)} examples from evaluation dataset")
    
    return examples, indices


def get_label_distribution(data: List[Dict], label_key: str) -> Dict[str, int]:
    """Get label distribution from data."""
    labels = [item.get(label_key) for item in data if item.get(label_key)]
    return dict(Counter(labels))
