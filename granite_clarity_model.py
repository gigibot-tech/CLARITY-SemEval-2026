import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Global variables for Granite model (lazy loading)
_granite_model = None
_granite_tokenizer = None
_granite_device = None


def _get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _load_granite_model():
    """Lazy load Granite model and tokenizer."""
    global _granite_model, _granite_tokenizer, _granite_device
    
    if _granite_model is None:
        model_name = "ibm-granite/granite-3.2-2b-instruct"
        _granite_device = _get_device()
        
        print(f"Loading Granite model: {model_name} on {_granite_device}")
        _granite_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _granite_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        _granite_model.eval()
        print("Granite model loaded successfully")
    
    return _granite_model, _granite_tokenizer, _granite_device


def call_granite_model(messages, temperature=0.7, max_new_tokens=512):
    """
    Call IBM Granite 3.2-2B-Instruct model with reasoning capabilities.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature (default 0.7 for self-consistency)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string (includes reasoning if thinking=True)
    """
    model, tokenizer, device = _load_granite_model()
    
    # Apply chat template with thinking=True for reasoning
    try:
        # Try to use thinking=True if supported by the tokenizer
        # Granite 3.2 supports reasoning via thinking parameter
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking=True  # Enable reasoning mode
            )
        except TypeError:
            # Fallback if thinking parameter not supported
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
    except Exception as e:
        # Fallback: manual formatting
        if isinstance(messages, list) and len(messages) > 0:
            # Try to extract user message
            user_msg = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            
            if user_msg:
                formatted = user_msg
            else:
                formatted = str(messages[-1].get("content", ""))
        else:
            formatted = str(messages)
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    # Generate with reasoning
    with torch.no_grad():
        # Granite 3.2 supports thinking via special tokens or generation config
        # We'll use the standard generation and let the model use its reasoning
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip()
