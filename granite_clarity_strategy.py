"""
Granite CLARITY Strategy

Uses IBM Granite 3.2-2B-Instruct with reasoning capabilities for CLARITY classification.
Generates JSON-structured output with reasoning and label prediction.
"""

import json
import re
from granite_clarity_model import call_granite_model


class GraniteClarityStrategy:
    """Strategy for CLARITY classification using Granite 3.2 with reasoning."""

    name = "granite-clarity"

    def build_prompt(self, question, answer):
        """
        Build prompt for CLARITY classification.
        
        Args:
            question: Interview question string
            answer: Interview answer string
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing political interview answers for clarity classification.

Question: {question}
Answer: {answer}

Analyze the answer step-by-step:
1. Does it directly address the question?
2. Is it evasive or indirect?
3. Does it decline to answer?

Provide your reasoning and then classify as one of:
- "Direct Reply": Directly answers the question
- "Direct Non-Reply": Explicitly declines or claims inability to answer
- "Indirect": Evasive, indirect, or partially answers

Respond in JSON format:
{{
  "reasoning": "Your step-by-step analysis...",
  "label": "Direct Reply|Direct Non-Reply|Indirect"
}}"""
        return prompt

    def extract_json(self, text: str):
        """Extract JSON from Granite model response."""
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Try to find JSON object
        try:
            # First, try direct parsing
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*"label"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Last resort: try to extract label and reasoning separately
            label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text, re.DOTALL)
            
            if label_match:
                result = {"label": label_match.group(1)}
                if reasoning_match:
                    result["reasoning"] = reasoning_match.group(1)
                else:
                    # Try to extract reasoning without quotes (might be multiline)
                    reasoning_match = re.search(r'"reasoning"\s*:\s*([^,}]+)', text, re.DOTALL)
                    if reasoning_match:
                        result["reasoning"] = reasoning_match.group(1).strip().strip('"')
                    else:
                        result["reasoning"] = "No reasoning provided"
                return result

            raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")

    def predict_single(self, question, answer, temperature=0.7):
        """
        Predict CLARITY label for a single question-answer pair.
        
        Args:
            question: Interview question
            answer: Interview answer
            temperature: Sampling temperature
        
        Returns:
            Dict with 'label' and 'reasoning' keys, or None if parsing fails
        """
        prompt = self.build_prompt(question, answer)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = call_granite_model(messages, temperature=temperature)
            parsed = self.extract_json(response)
            
            # Validate label
            valid_labels = ["Direct Reply", "Direct Non-Reply", "Indirect"]
            if parsed.get("label") not in valid_labels:
                # Try to normalize the label
                label_lower = parsed.get("label", "").lower()
                if "direct" in label_lower and "reply" in label_lower:
                    parsed["label"] = "Direct Reply"
                elif "direct" in label_lower and ("non" in label_lower or "decline" in label_lower):
                    parsed["label"] = "Direct Non-Reply"
                elif "indirect" in label_lower or "evasive" in label_lower:
                    parsed["label"] = "Indirect"
                else:
                    # Default fallback
                    parsed["label"] = "Indirect"
            
            return parsed
        except Exception as e:
            print(f"Error in predict_single: {e}")
            return None

    def predict_batch(self, examples):
        """
        Predict CLARITY labels for a batch of examples.
        
        Args:
            examples: List of dicts with 'question' and 'answer' keys
        
        Returns:
            List of dicts with 'label' and 'reasoning' keys
        """
        results = []
        for example in examples:
            question = example.get("question", "")
            answer = example.get("answer", "")
            
            if not question or not answer:
                results.append({"label": "Indirect", "reasoning": "Missing question or answer"})
                continue
            
            prediction = self.predict_single(question, answer, temperature=0.0)
            if prediction:
                results.append(prediction)
            else:
                results.append({"label": "Indirect", "reasoning": "Prediction failed"})
        
        return results
