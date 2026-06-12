"""
Evaluation metrics for MAS outputs.

Dispatches to the right evaluator based on scenario:
  - customer_service:      LLM-as-judge (approve/deny + actions)
  - software_engineering:  Sandboxed code runner (unit test pass/fail)
  - homogeneous_debate:    Exact string match (majority vote vs answer)

All evaluators return a scores dict with at least:
  {"decision_accuracy": 0|1, "action_correctness": ..., "reasoning_quality": ...}
"""

import json
import re
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"


def evaluate_task(resolution: str, task: dict, task_description: str,
                  scenario_name: str, model: str = "llama3.1:8b") -> dict:
    """Unified evaluation dispatcher — routes to the right evaluator by scenario.

    Args:
        resolution:      The final output text from the MAS.
        task:            The full task dict (includes ground_truth, mock_data, etc.)
        task_description: The task description string.
        scenario_name:   One of "customer_service", "software_engineering", "homogeneous_debate".
        model:           Model name (used by LLM-as-judge for customer_service).

    Returns:
        Scores dict with at minimum {"decision_accuracy": 0|1}.
    """
    if scenario_name == "software_engineering":
        from src.evaluation.code_runner import evaluate_resolution
        return evaluate_resolution(resolution, task)

    if scenario_name == "homogeneous_debate":
        return debate_judge(resolution, task)

    # Default: customer_service (LLM-as-judge)
    return llm_judge(resolution, task.get("ground_truth", {}),
                     task_description, model=model)


def debate_judge(resolution: str, task: dict) -> dict:
    """Exact-match evaluation for homogeneous debate (GSM8K / CommonsenseQA).

    Extracts the final ANSWER from the resolution and compares to ground truth.
    """
    ground_truth = task.get("ground_truth", {})
    correct_answer = str(ground_truth.get("answer", "")).strip()
    answer_type = ground_truth.get("answer_type", "integer")

    # Extract answer from resolution: look for "ANSWER: <value>"
    extracted = None
    match = re.search(r"ANSWER:\s*(.+?)(?:\s*$|\n)", resolution, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().rstrip(".")

    if extracted is None:
        # Fallback: look for the last number (GSM8K) or letter (CSQA) in text
        if answer_type == "integer":
            numbers = re.findall(r"[-+]?\d+", resolution)
            extracted = numbers[-1] if numbers else None
        elif answer_type == "multiple_choice":
            letters = re.findall(r"\b([A-E])\b", resolution)
            extracted = letters[-1] if letters else None

    if extracted is None:
        return {
            "decision_accuracy": 0,
            "action_correctness": 0,
            "reasoning_quality": -1,
            "explanation": "Could not extract answer from resolution",
            "extracted_answer": None,
            "correct_answer": correct_answer,
        }

    # Normalize comparison
    if answer_type == "integer":
        try:
            is_correct = int(extracted) == int(correct_answer)
        except (ValueError, TypeError):
            is_correct = extracted.strip() == correct_answer
    else:
        is_correct = extracted.strip().upper() == correct_answer.strip().upper()

    return {
        "decision_accuracy": 1 if is_correct else 0,
        "action_correctness": 1 if is_correct else 0,
        "reasoning_quality": -1,  # not assessed
        "explanation": f"Extracted '{extracted}', correct '{correct_answer}'",
        "extracted_answer": extracted,
        "correct_answer": correct_answer,
    }


def llm_judge(resolution: str, ground_truth: dict,
              task_description: str, model: str = "llama3.1:8b") -> dict:
    """Use LLM-as-judge to evaluate a resolution against ground truth.

    Returns dict with scores for decision_accuracy, action_correctness, reasoning_quality.
    """
    correct_actions = ground_truth.get("correct_actions", [])
    incorrect_actions = ground_truth.get("incorrect_actions", [])
    reason = ground_truth.get("reason", "")

    prompt = f"""You are an evaluator. Given a customer service resolution and the ground truth,
score the resolution on three dimensions.

TASK: {task_description}

GROUND TRUTH:
- Return eligible: {ground_truth.get('return_eligible', 'N/A')}
- Refund eligible: {ground_truth.get('refund_eligible', 'N/A')}
- Warranty repair eligible: {ground_truth.get('warranty_repair_eligible', 'N/A')}
- Reason: {reason}
- Correct actions: {json.dumps(correct_actions)}
- Incorrect actions (should NOT be taken): {json.dumps(incorrect_actions)}

RESOLUTION TO EVALUATE:
{resolution}

Score each dimension as 0 or 1:

1. decision_accuracy: Did the resolution make the correct approve/deny decision? (1=correct, 0=wrong)
2. action_correctness: Were the correct actions proposed and incorrect actions avoided? (1=all correct, 0=any error)
3. reasoning_quality: Was the reasoning factually accurate and consistent? (1=good, 0=flawed)

Respond in EXACTLY this JSON format, nothing else:
{{"decision_accuracy": 0 or 1, "action_correctness": 0 or 1, "reasoning_quality": 0 or 1, "explanation": "brief explanation"}}"""

    messages = [{"role": "user", "content": prompt}]
    is_thinking_model = any(k in model for k in ["qwen3.5", "qwen3-", "gemma4"])
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    if is_thinking_model:
        payload["think"] = False

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Try to parse JSON from the response
        # Handle cases where LLM wraps JSON in markdown
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        return result
    except (json.JSONDecodeError, KeyError, IndexError):
        # Fallback: try keyword matching
        return keyword_judge(resolution, ground_truth)
    except requests.exceptions.RequestException as e:
        return {"decision_accuracy": -1, "action_correctness": -1,
                "reasoning_quality": -1, "explanation": f"Judge error: {e}"}


def keyword_judge(resolution: str, ground_truth: dict) -> dict:
    """Simple keyword-based evaluation as fallback.

    Checks if correct action keywords appear and incorrect ones don't.
    """
    resolution_lower = resolution.lower()

    # Check decision accuracy
    is_eligible = ground_truth.get("return_eligible", False)
    deny_keywords = ["deny", "denied", "cannot", "not eligible", "decline", "refused", "non-refundable"]
    approve_keywords = ["approve", "accept", "process", "eligible", "authorized", "refund"]

    has_deny = any(k in resolution_lower for k in deny_keywords)
    has_approve = any(k in resolution_lower for k in approve_keywords)

    if is_eligible:
        decision_correct = 1 if has_approve and not has_deny else 0
    else:
        decision_correct = 1 if has_deny else 0

    # Check action correctness
    correct_actions = ground_truth.get("correct_actions", [])
    incorrect_actions = ground_truth.get("incorrect_actions", [])

    correct_found = sum(1 for a in correct_actions
                       if any(w.lower() in resolution_lower
                             for w in a.lower().split() if len(w) > 4))
    incorrect_found = sum(1 for a in incorrect_actions
                         if any(w.lower() in resolution_lower
                               for w in a.lower().split() if len(w) > 4))

    action_score = 1 if correct_found > 0 and incorrect_found == 0 else 0

    return {
        "decision_accuracy": decision_correct,
        "action_correctness": action_score,
        "reasoning_quality": -1,  # Can't assess reasoning with keywords
        "explanation": "keyword-based evaluation (LLM judge failed)",
    }
