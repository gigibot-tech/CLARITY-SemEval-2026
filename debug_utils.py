"""
Debug utilities for generative mode evaluation.
"""
import torch
import re

def debug_generative_output(model, tokenizer, input_ids, attention_mask, labels=None, num_examples=3):
    """Debug what the generative model outputs and how it's parsed."""
    
    if model.mode != "generative":
        print("⚠️ Model is not in generative mode!")
        return
    
    model.eval()
    with torch.no_grad():
        # Generate for first few examples
        batch_size = min(num_examples, input_ids.shape[0])
        input_ids_subset = input_ids[:batch_size]
        attention_mask_subset = attention_mask[:batch_size]
        
        print("=" * 80)
        print("GENERATIVE MODE DEBUG")
        print("=" * 80)
        
        # Generate text
        outputs = model.model.generate(
            input_ids=input_ids_subset,
            attention_mask=attention_mask_subset,
            max_new_tokens=200,  # Increased to see past thinking tags
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Decode full prompt + generation
        for i in range(batch_size):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i+1}")
            print(f"{'='*80}")
            
            # Full text (prompt + generation)
            full_text = tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
            print(f"\n📝 FULL TEXT (prompt + generation):")
            print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
            
            # Just the prompt part
            prompt_text = tokenizer.decode(input_ids_subset[i], skip_special_tokens=True)
            print(f"\n📋 PROMPT (input):")
            print(prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text)
            
            # Just the generated part
            generated_tokens = outputs.sequences[i][input_ids_subset.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"\n✨ GENERATED TEXT (new tokens only):")
            print(f"'{generated_text}'")
            print(f"Length: {len(generated_text)} chars")
            
            # Extract text after thinking tags
            text_after_thinking = generated_text
            if '`</think>`' in generated_text:
                parts = generated_text.split('`</think>`')
                if len(parts) > 1:
                    text_after_thinking = parts[-1].strip()
                    print(f"\n🔍 TEXT AFTER THINKING TAGS:")
                    print(f"'{text_after_thinking}'")
            
            # Try parsing
            parsed_label_id = model._parse_generated_text(text_after_thinking)
            parsed_label_name = model.id2label.get(parsed_label_id, "UNKNOWN") if parsed_label_id is not None else "UNKNOWN"
            
            print(f"\n🔍 PARSING RESULT:")
            print(f"  Parsed Label ID: {parsed_label_id}")
            print(f"  Parsed Label Name: '{parsed_label_name}'")
            
            if labels is not None:
                expected_label_id = labels[i].item()
                expected_label_name = model.id2label.get(expected_label_id, "UNKNOWN")
                print(f"\n✅ EXPECTED:")
                print(f"  Expected Label ID: {expected_label_id}")
                print(f"  Expected Label Name: '{expected_label_name}'")
                print(f"  Match: {'✅ CORRECT' if parsed_label_id == expected_label_id else '❌ WRONG'}")
            
            # Show token IDs for debugging
            print(f"\n🔢 GENERATED TOKEN IDs (first 20):")
            print(generated_tokens.tolist()[:20])
            
            # Show what tokens map to
            print(f"\n🔤 TOKEN MAPPINGS (first 10):")
            for j, token_id in enumerate(generated_tokens[:10]):
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                print(f"  Token {j}: ID={token_id} -> '{token_str}'")
