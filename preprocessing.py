"""
Preprocessing functions for Qwen3 generative classification.
"""


def get_max_token_length(dataset, tokenizer, split="train", max_samples=None, cap=1024):
    """
    Max token length for "Question: {q}\n\nAnswer: {a}" over the dataset.
    Returns min(actual_max, cap) so you don't OOM. Uses train split by default.
    """
    data = dataset[split]
    n = len(data["interview_question"])
    if max_samples is not None:
        n = min(n, max_samples)
    lengths = []
    for i in range(n):
        q = str(data["interview_question"][i])
        a = str(data["interview_answer"][i])
        text = f"Question: {q}\n\nAnswer: {a}"
        lengths.append(len(tokenizer.encode(text, add_special_tokens=True)))
    actual_max = max(lengths)
    return min(actual_max, cap)


def preprocess_function_qwen_generative(examples, tokenizer, label_to_id, enable_thinking=True, max_length=512):
    """
    Tokenize with JSON prompt for 3-class generative classification.
    
    Args:
        examples: Dataset examples dict with 'interview_question', 'interview_answer', 'clarity_label'
        tokenizer: HuggingFace tokenizer
        label_to_id: Mapping from label name to ID
        enable_thinking: Whether to enable thinking mode in chat template
        max_length: Maximum sequence length
    
    Returns:
        Tokenized inputs with 'input_ids', 'attention_mask', 'labels'
    """
    texts = []
    labels = []

    CLASSIFICATION_PROMPT = """Classify this political interview answer into one of three categories:

1. "Ambivalent": The answer is unclear, mixed, or doesn't clearly indicate whether the question was answered
2. "Clear Non-Reply": The answer clearly avoids or evades the question without providing a direct answer
3. "Clear Reply": The answer directly addresses and answers the question

Question: {question}
Answer: {answer}

Respond with JSON format: {{"label": "Ambivalent"|"Clear Non-Reply"|"Clear Reply"}}"""

    for q, a, label in zip(examples["interview_question"], examples["interview_answer"], examples["clarity_label"]):
        prompt = CLASSIFICATION_PROMPT.format(question=str(q), answer=str(a))
        
        # Use Qwen chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Important for generation
            enable_thinking=enable_thinking
        )
        texts.append(formatted_text)
        labels.append(label_to_id[label])  # Use 3-class mapping

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    tokenized['labels'] = labels
    return tokenized


def preprocess_function_qwen_classification_head(examples, tokenizer, label_to_id, max_length=512):
    """
    Tokenize for classification head: short "Question / Answer" text only (no reasoning, no long prompt).
    Same columns as generative: input_ids, attention_mask, labels.
    """
    texts = []
    labels = []
    for q, a, label in zip(
        examples["interview_question"],
        examples["interview_answer"],
        examples["clarity_label"],
    ):
        text = f"Question: {str(q)}\n\nAnswer: {str(a)}"
        texts.append(text)
        labels.append(label_to_id[label])

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized["labels"] = labels
    return tokenized
