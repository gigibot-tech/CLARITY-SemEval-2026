#!/usr/bin/env python3
"""
Generate Rationale Dataset from QEvasion using MuleRouter (Qwen reasoning model)

This script:
1. Loads QEvasion dataset from HuggingFace
2. Calls MuleRouter API to generate native reasoning for each example
3. Saves results incrementally to CSV
4. Creates RationaleTraining.jsonl for knowledge distillation
"""

import os
import json
import csv
import time
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Use INFO for normal operation, DEBUG logs are still shown for errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rationale_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MULEROUTER_API_KEY = "sk-mr-f53f882fbbb65703ddf6397db2871fd866d1c08a46562afe5552ed5754bbd882"
MULEROUTER_MODEL = "qwen3-max"  # Reasoning-capable model
MULEROUTER_BASE_URL = "https://api.mulerouter.ai/vendors/openai/v1"
OUTPUT_CSV = "qevasion_rationale_dataset.csv"
OUTPUT_JSONL = "RationaleTraining.jsonl"
CHECKPOINT_FILE = "rationale_checkpoint.json"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Initialize MuleRouter client (OpenAI-compatible API)
mulerouter_client = OpenAI(
    base_url=MULEROUTER_BASE_URL,
    api_key=MULEROUTER_API_KEY,
)


def extract_reasoning_and_verdict(response_text: str) -> Tuple[str, str]:
    """
    Extract reasoning and semantic verdict from MuleRouter response.
    
    Expected format: reasoning in <think> tags, verdict as "Verdict: Clear Reply" etc.
    Returns semantic verdict: "Clear Reply", "Clear Non-Reply", or "Ambivalent"
    """
    import re
    reasoning = ""
    verdict = ""
    
    # Try to extract reasoning from <think> tags
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
    else:
        # If no <think> tags, try to find reasoning before "Verdict:"
        verdict_match = re.search(r'Verdict:\s*([^\n]+)', response_text, re.IGNORECASE)
        if verdict_match:
            # Everything before "Verdict:" is reasoning
            reasoning = response_text[:verdict_match.start()].strip()
        else:
            # Fallback: use entire response as reasoning
            reasoning = response_text.strip()
    
    # Extract semantic verdict - look for "Verdict: [label]"
    verdict_match = re.search(r'Verdict:\s*([^\n]+)', response_text, re.IGNORECASE)
    if verdict_match:
        verdict_text = verdict_match.group(1).strip()
        # Normalize verdict to match expected labels
        verdict_lower = verdict_text.lower()
        if "clear reply" in verdict_lower or verdict_lower == "clear":
            verdict = "Clear Reply"
        elif "clear non-reply" in verdict_lower or "non-reply" in verdict_lower:
            verdict = "Clear Non-Reply"
        elif "ambivalent" in verdict_lower:
            verdict = "Ambivalent"
        else:
            # Try to infer from text content
            if "clear" in verdict_lower and "non" not in verdict_lower:
                verdict = "Clear Reply"
            elif "non" in verdict_lower or "evasive" in verdict_lower:
                verdict = "Clear Non-Reply"
            elif "ambivalent" in verdict_lower or "unclear" in verdict_lower:
                verdict = "Ambivalent"
            else:
                verdict = verdict_text  # Return as-is if can't normalize
    else:
        # Try to infer from response text
        response_lower = response_text.lower()
        if "clear reply" in response_lower:
            verdict = "Clear Reply"
        elif "clear non-reply" in response_lower or "non-reply" in response_lower:
            verdict = "Clear Non-Reply"
        elif "ambivalent" in response_lower:
            verdict = "Ambivalent"
        else:
            verdict = ""  # Unknown
    
    return reasoning, verdict


def get_initial_reasoning(question: str, answer: str, clarity_label: str) -> Tuple[str, str, Optional[str]]:
    """
    Get initial reasoning attempt from MuleRouter.
    
    Returns:
        Tuple of (reasoning_text, verdict_semantic, error_message)
        error_message is None if successful, otherwise contains error details
    """
    prompt = f"""The following interview question and answer have been labeled as: {clarity_label}.

Question: {question}
Answer: {answer}

Analyze the logic behind this label. Provide your step-by-step reasoning, then conclude with a verdict.

Format your response as:
<think>
[Your reasoning here]
</think>
Verdict: [One of: "Clear Reply", "Clear Non-Reply", or "Ambivalent"]"""
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Calling MuleRouter API for initial reasoning (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            response = mulerouter_client.chat.completions.create(
                model=MULEROUTER_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,  # Allow enough tokens for reasoning
            )
            
            # Extract content from OpenAI-compatible response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            elif isinstance(response, dict):
                response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                response_text = str(response)
            
            logger.debug(f"Extracted response_text length: {len(response_text) if response_text else 0}")
            if response_text:
                logger.debug(f"Response text preview: {response_text[:200]}")
            
            if not response_text:
                # Log full response structure for debugging
                response_str = str(response)
                if hasattr(response, '__dict__'):
                    response_str = json.dumps(response.__dict__, indent=2, default=str)
                elif isinstance(response, dict):
                    response_str = json.dumps(response, indent=2, default=str)
                
                error_msg = f"Empty response from MuleRouter API (attempt {attempt + 1})"
                logger.error(error_msg)
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response structure:\n{response_str[:1000]}")
                last_error = f"{error_msg}. Response: {response_str[:500]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return "", "", last_error
            
            reasoning, verdict = extract_reasoning_and_verdict(response_text)
            
            if not reasoning or not verdict:
                error_msg = f"Failed to extract reasoning/verdict from response (attempt {attempt + 1}). Response: {response_text[:200]}..."
                logger.warning(error_msg)
                last_error = error_msg
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return "", "", error_msg
            
            logger.info(f"Successfully extracted reasoning (length: {len(reasoning)}, verdict: {verdict})")
            return reasoning, verdict, None
            
        except Exception as e:
            import traceback
            error_msg = f"MuleRouter API error (attempt {attempt + 1}): {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            last_error = error_msg
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"All {MAX_RETRIES} retries failed for question: {question[:50]}...")
                return "", "", error_msg
    
    return "", "", last_error or "Unknown error"


def get_corrective_reasoning(
    question: str,
    answer: str,
    true_label: str,
    evasion_label: Optional[str],
    initial_verdict: str,
    initial_reasoning: str
) -> Tuple[str, str, Optional[str]]:
    """
    Perform corrective reasoning turn when initial verdict is wrong.
    
    Returns:
        Tuple of (corrective_reasoning_text, final_verdict_semantic, error_message)
        error_message is None if successful, otherwise contains error details
    """
    evasion_context = f" The evasion type is: {evasion_label}." if evasion_label else ""
    
    prompt = f"""Actually, the correct label is {true_label}.{evasion_context}

Your initial analysis concluded: {initial_verdict}
Your initial reasoning: {initial_reasoning[:500]}...

Look at the response again, specifically looking for:
- Pivoting away from the question
- Red herrings or distractions
- Non-committal language
- Vague or ambiguous statements

Re-analyze your logic to see where you missed the evasion. Provide your corrected reasoning.

Format your response as:
<think>
[Your corrected reasoning here]
</think>
Verdict: [Correct semantic label: "Clear Reply", "Clear Non-Reply", or "Ambivalent"]"""
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Calling MuleRouter API for corrective reasoning (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            response = mulerouter_client.chat.completions.create(
                model=MULEROUTER_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            
            # Extract content from OpenAI-compatible response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            elif isinstance(response, dict):
                response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                response_text = str(response)
            
            logger.debug(f"Extracted corrective response_text length: {len(response_text) if response_text else 0}")
            
            if not response_text:
                # Log full response structure for debugging
                response_str = str(response)
                if hasattr(response, '__dict__'):
                    response_str = json.dumps(response.__dict__, indent=2, default=str)
                elif isinstance(response, dict):
                    response_str = json.dumps(response, indent=2, default=str)
                
                error_msg = f"Empty response from MuleRouter API for corrective reasoning (attempt {attempt + 1})"
                logger.error(error_msg)
                logger.error(f"Corrective response type: {type(response)}")
                logger.error(f"Corrective response structure:\n{response_str[:1000]}")
                last_error = f"{error_msg}. Response: {response_str[:500]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return "", initial_verdict, last_error
            
            reasoning, verdict = extract_reasoning_and_verdict(response_text)
            
            if not reasoning or not verdict:
                error_msg = f"Failed to extract reasoning/verdict from corrective response (attempt {attempt + 1}). Response: {response_text[:200]}..."
                logger.warning(error_msg)
                last_error = error_msg
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return "", initial_verdict, error_msg
            
            logger.info(f"Successfully extracted corrective reasoning (length: {len(reasoning)}, verdict: {verdict})")
            return reasoning, verdict, None
            
        except Exception as e:
            import traceback
            error_msg = f"MuleRouter API error for corrective reasoning (attempt {attempt + 1}): {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            last_error = error_msg
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"All {MAX_RETRIES} corrective retries failed")
                return "", initial_verdict, error_msg
    
    return "", initial_verdict, last_error or "Unknown error"


def get_native_reasoning_with_correction(
    question: str, 
    answer: str, 
    clarity_label: str,
    evasion_label: Optional[str] = None
) -> Tuple[str, str, str, str, bool, Optional[str]]:
    """
    Multi-turn reasoning with correction if needed.
    
    Returns:
        (initial_reasoning, initial_verdict, corrective_reasoning, final_verdict, correction_applied, error_message)
        error_message is None if successful, otherwise contains error details
    """
    # Step A: Initial attempt
    initial_reasoning, initial_verdict, error = get_initial_reasoning(question, answer, clarity_label)
    
    # If initial reasoning failed, return error
    if error:
        return initial_reasoning, initial_verdict, "", "", False, error
    
    # Check if correction needed
    if initial_verdict != clarity_label and initial_verdict:
        logger.info(f"Initial verdict '{initial_verdict}' doesn't match '{clarity_label}', applying correction...")
        # Step B: Corrective turn
        corrective_reasoning, final_verdict, corrective_error = get_corrective_reasoning(
            question, answer, clarity_label, evasion_label, initial_verdict, initial_reasoning
        )
        if corrective_error:
            return initial_reasoning, initial_verdict, corrective_reasoning, initial_verdict, True, corrective_error
        return initial_reasoning, initial_verdict, corrective_reasoning, final_verdict, True, None
    else:
        return initial_reasoning, initial_verdict, "", initial_verdict, False, None


def load_checkpoint() -> int:
    """Load checkpoint to resume from last processed index."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                return checkpoint.get('last_index', 0)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return 0


def save_checkpoint(index: int):
    """Save checkpoint with last processed index."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({'last_index': index, 'timestamp': datetime.now().isoformat()}, f)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def validate_csv_header(csv_path: str, expected_fieldnames: list) -> bool:
    """
    Check if existing CSV file has the correct header.
    Returns True if header matches, False otherwise.
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            if existing_fields is None:
                return False
            # Compare fieldnames (order matters for CSV)
            return list(existing_fields) == expected_fieldnames
    except Exception as e:
        logger.warning(f"Error reading CSV header: {e}")
        return False


def initialize_csv(csv_path: str, fieldnames: list) -> Tuple[bool, str]:
    """
    Initialize CSV file with header if it doesn't exist.
    If file exists but header doesn't match, create a new file with timestamp.
    
    Returns:
        Tuple of (file_existed, actual_csv_path)
    """
    file_exists = os.path.exists(csv_path)
    
    if file_exists:
        # Validate header
        if not validate_csv_header(csv_path, fieldnames):
            logger.warning(f"CSV header doesn't match expected fields. Creating new file...")
            # Create new file with timestamp
            csv_path_obj = Path(csv_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_csv_path = csv_path_obj.parent / f"{csv_path_obj.stem}_{timestamp}{csv_path_obj.suffix}"
            logger.info(f"Creating new CSV file: {new_csv_path}")
            csv_path = str(new_csv_path)
            file_exists = False
    
    if not file_exists:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f"Initialized CSV file: {csv_path}")
    
    return file_exists, csv_path


def append_to_csv(csv_path: str, fieldnames: list, row: Dict[str, Any]):
    """Append a row to CSV file."""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def append_to_jsonl(jsonl_path: str, data: Dict[str, Any]):
    """Append a JSON object to JSONL file."""
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def process_dataset(
    dataset,
    start_index: int = 0,
    csv_path: str = OUTPUT_CSV,
    jsonl_path: str = OUTPUT_JSONL
):
    """
    Process QEvasion dataset and generate rationale data.
    
    Args:
        dataset: HuggingFace dataset
        start_index: Index to start from (for resuming)
        csv_path: Path to output CSV file
        jsonl_path: Path to output JSONL file
    """
    logger.info(f"Starting processing from index {start_index}")
    
    # Get all fieldnames for CSV (excluding annotator columns)
    original_fields = list(dataset[0].keys())
    # Remove annotator columns
    annotator_columns = ['annotator_id', 'annotator1', 'annotator2', 'annotator3']
    original_fields = [f for f in original_fields if f not in annotator_columns]
    
    new_fields = [
        'initial_reasoning',
        'initial_verdict',
        'corrective_reasoning',
        'final_verdict',
        'correction_applied',
        'verdict_match'
    ]
    all_fields = original_fields + new_fields
    
    # Initialize CSV file (may return new path if header doesn't match)
    csv_exists, csv_path = initialize_csv(csv_path, all_fields)
    
    # Check if JSONL file exists
    jsonl_exists = os.path.exists(jsonl_path)
    
    total_examples = len(dataset)
    logger.info(f"Processing {total_examples} examples (starting from index {start_index})")
    
    processed_count = 0
    matching_count = 0
    
    for idx in range(start_index, total_examples):
        try:
            example = dataset[idx]
            question = str(example.get('interview_question', ''))
            answer = str(example.get('interview_answer', ''))
            clarity_label = str(example.get('clarity_label', ''))
            
            if not question or not answer:
                logger.warning(f"Skipping example {idx}: missing question or answer")
                continue
            
            logger.info(f"[{idx+1}/{total_examples}] Processing example {idx}...")
            logger.debug(f"Question: {question[:100]}...")
            
            # Get evasion_label for corrective prompts
            evasion_label = str(example.get('evasion_label', '')) if example.get('evasion_label') else None
            
            # Get native reasoning with multi-turn correction
            initial_reasoning, initial_verdict, corrective_reasoning, final_verdict, correction_applied, error = \
                get_native_reasoning_with_correction(question, answer, clarity_label, evasion_label)
            
            # Validate that reasoning was generated
            if error or not initial_reasoning or not initial_verdict:
                logger.error(f"Failed to generate reasoning for example {idx}. Skipping row...")
                logger.error(f"  Error: {error if error else 'No error message but reasoning is empty'}")
                logger.error(f"  initial_reasoning length: {len(initial_reasoning) if initial_reasoning else 0}")
                logger.error(f"  initial_reasoning preview: '{initial_reasoning[:200] if initial_reasoning else 'EMPTY'}'")
                logger.error(f"  initial_verdict: '{initial_verdict}'")
                logger.error(f"  Question: {question[:100]}...")
                if error:
                    logger.error(f"  Full error details: {error}")
                continue
            
            # Check if final verdict matches clarity_label
            verdict_match = (final_verdict == clarity_label) if final_verdict else False
            
            if verdict_match:
                matching_count += 1
            
            # Prepare CSV row (excluding annotator columns)
            # Filter example dict to only include fields in all_fields
            filtered_example = {k: v for k, v in example.items() if k in original_fields}
            
            # Convert boolean values to strings for CSV compatibility
            csv_row = {**filtered_example, **{
                'initial_reasoning': initial_reasoning,
                'initial_verdict': initial_verdict,
                'corrective_reasoning': corrective_reasoning if corrective_reasoning else '',
                'final_verdict': final_verdict if final_verdict else initial_verdict,
                'correction_applied': str(correction_applied),  # Convert bool to string
                'verdict_match': str(verdict_match)  # Convert bool to string
            }}
            
            # Save to CSV incrementally
            append_to_csv(csv_path, all_fields, csv_row)
            
            # If final verdict matches, add to training JSONL
            if verdict_match:
                if correction_applied:
                    # Multi-turn conversation format
                    training_data = {
                        "instruction": "Analyze this politician's answer for evasion. Determine if the answer is clear, evasive, or ambivalent.",
                        "input": f"Q: {question}\nA: {answer}",
                        "output": f"<think>\n{initial_reasoning}\n</think>\nVerdict: {initial_verdict}\n\nActually, the correct label is {clarity_label}.{' The evasion type is: ' + evasion_label + '.' if evasion_label else ''}\n\n<think>\n{corrective_reasoning}\n</think>\nVerdict: {final_verdict}"
                    }
                else:
                    # Single-turn format
                    training_data = {
                        "instruction": "Analyze this politician's answer for evasion. Determine if the answer is clear, evasive, or ambivalent.",
                        "input": f"Q: {question}\nA: {answer}",
                        "output": f"<think>\n{initial_reasoning}\n</think>\nVerdict: {final_verdict}"
                    }
                append_to_jsonl(jsonl_path, training_data)
            
            processed_count += 1
            
            # Save checkpoint every 10 examples
            if processed_count % 10 == 0:
                save_checkpoint(idx + 1)
                logger.info(f"Progress: {processed_count} processed, {matching_count} matching verdicts")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final checkpoint
    save_checkpoint(total_examples)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Matching verdicts: {matching_count}")
    logger.info(f"Match rate: {matching_count/processed_count*100:.2f}%" if processed_count > 0 else "N/A")
    logger.info(f"CSV saved to: {csv_path}")
    logger.info(f"Training JSONL saved to: {jsonl_path}")
    logger.info(f"{'='*60}")


def sample_balanced_by_label(dataset, per_label_cap: int, seed: int = 42):
    """
    Return a balanced list of examples sampled by clarity_label.
    """
    if per_label_cap <= 0:
        return dataset

    buckets = {}
    for i in range(len(dataset)):
        label = str(dataset[i].get("clarity_label", "")).strip()
        if not label:
            continue
        buckets.setdefault(label, []).append(i)

    if not buckets:
        logger.warning("No labels found for balancing; using full dataset.")
        return dataset

    rng = random.Random(seed)
    selected_indices = []
    for label, idxs in sorted(buckets.items()):
        k = min(per_label_cap, len(idxs))
        sampled = rng.sample(idxs, k)
        selected_indices.extend(sampled)
        logger.info(f"Balanced sample for '{label}': selected {k}/{len(idxs)}")

    rng.shuffle(selected_indices)
    sampled_dataset = [dataset[i] for i in selected_indices]
    logger.info(f"Balanced sampled dataset size: {len(sampled_dataset)}")
    return sampled_dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate rationale dataset from QEvasion.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--per-label-cap",
        type=int,
        default=0,
        help="If >0, sample up to this many examples per clarity_label before generation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default=OUTPUT_CSV)
    parser.add_argument("--output-jsonl", type=str, default=OUTPUT_JSONL)
    parser.add_argument("--checkpoint-file", type=str, default=CHECKPOINT_FILE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file if present (non-interactive).",
    )
    args = parser.parse_args()

    globals()["OUTPUT_CSV"] = args.output_csv
    globals()["OUTPUT_JSONL"] = args.output_jsonl
    globals()["CHECKPOINT_FILE"] = args.checkpoint_file

    logger.info("="*60)
    logger.info("QEvasion Rationale Dataset Generation")
    logger.info("="*60)
    logger.info(f"MuleRouter Model: {MULEROUTER_MODEL}")
    logger.info(f"MuleRouter Base URL: {MULEROUTER_BASE_URL}")
    logger.info(f"Output CSV: {OUTPUT_CSV}")
    logger.info(f"Output JSONL: {OUTPUT_JSONL}")
    
    # Check if MuleRouter is accessible
    try:
        logger.info("Checking MuleRouter connection...")
        # Test connection by making a simple API call
        test_response = mulerouter_client.models.list()
        logger.info(f"MuleRouter connected successfully")
    except Exception as e:
        logger.warning(f"Could not verify MuleRouter connection: {e}")
        logger.info("Continuing anyway - will test connection on first API call")
    
    # Load QEvasion dataset
    logger.info("Loading QEvasion dataset...")
    try:
        dataset_dict = load_dataset('ailsntua/QEvasion')
        work_dataset = dataset_dict[args.split]
        logger.info(f"Loaded {args.split} dataset with {len(work_dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Optional class-balanced sampling.
    if args.per_label_cap > 0:
        work_dataset = sample_balanced_by_label(work_dataset, per_label_cap=args.per_label_cap, seed=args.seed)
    
    # Load checkpoint if exists
    start_index = 0
    if args.resume:
        start_index = load_checkpoint()
        if start_index > 0:
            logger.info(f"Resuming from checkpoint index {start_index}")
    else:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("Removed old checkpoint (fresh run requested).")
    
    # Process dataset
    # Note: csv_path may be updated if header doesn't match
    process_dataset(work_dataset, start_index=start_index, csv_path=OUTPUT_CSV, jsonl_path=OUTPUT_JSONL)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
