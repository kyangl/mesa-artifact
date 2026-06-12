"""
Edge perturbation: inject non-adversarial noise on a specific edge.

Unlike misinformation (adversarial, pushes toward wrong answer), perturbation
is deliberately vague and uninformative — it degrades the signal without
specifically pushing a wrong conclusion. This tests whether edges carry
irreplaceable information or whether the system is robust to noise.

Used in dynamic saliency computation:
  ES_dynamic = f(ablation_delta, perturbation_delta)

Two perturbation modes:
  1. static: fixed vague templates per sender role (deterministic, fast)
  2. dynamic: LLM-generates a vague, off-topic response each call (default)
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

# Static vague templates per role — sound plausible but contain no useful info
NOISE_TEMPLATES = {
    "customer_facing": (
        "I've reviewed the customer's situation and gathered some initial information. "
        "There are various factors to consider here. The customer has expressed their request "
        "and I've noted the relevant details. Further analysis from the other team members "
        "will be needed to reach a complete assessment of this case."
    ),
    "database": (
        "I've checked the relevant records in the system. There is some information on file "
        "for this customer and their associated order. The records contain various fields "
        "that may be relevant to this case. I'd recommend the team review the applicable "
        "policies alongside the account information to determine the appropriate resolution."
    ),
    "policy": (
        "I've reviewed the applicable policies for this type of request. There are several "
        "policy considerations that could apply depending on the specific circumstances. "
        "The policy framework provides guidance for various scenarios, and the specifics "
        "of this case will determine which provisions are most relevant. A careful review "
        "of all the details is recommended before making a final determination."
    ),
    "transaction": (
        "I've assessed the transaction details associated with this request. There are "
        "various processing options available depending on the resolution that is determined. "
        "Any transaction would need to follow the standard procedures and be properly "
        "authorized before execution. I'm ready to proceed once a decision has been made."
    ),
    "supervisor": (
        "I've reviewed the overall situation. This is a case that has multiple angles to "
        "consider. The team should weigh all available information carefully. There may be "
        "different ways to approach the resolution, and it would be prudent to ensure all "
        "relevant factors have been considered before reaching a conclusion."
    ),
    "intake_manager": (
        "The intake team has gathered the initial information needed for this case. "
        "There are a number of considerations involved. I'll defer to the resolution "
        "team to determine the best course of action based on the information collected."
    ),
    "resolution_manager": (
        "The resolution team has reviewed the case from a policy and transaction standpoint. "
        "There are various factors in play. The final determination will need to account "
        "for all relevant aspects of this specific situation."
    ),
    "intake_lead": (
        "Our intake specialists have reviewed the request and gathered the relevant "
        "information. There are some nuances to this case worth noting. "
        "I'll pass our findings along for further evaluation."
    ),
    "resolution_lead": (
        "The resolution specialists have examined the applicable policies and options. "
        "There are considerations both for and against various resolutions. "
        "Ultimately, the decision will need to be made based on all available information."
    ),
}

_DEFAULT_NOISE = (
    "I've reviewed the information available for this case. There are several factors "
    "to consider and various possible approaches. The team should weigh all relevant "
    "details carefully before reaching a final determination."
)


def perturbation_attack(message: str, sender_role: str = None, **_kwargs) -> str:
    """Static noise injection — return a vague, uninformative response for this role."""
    if sender_role and sender_role in NOISE_TEMPLATES:
        return NOISE_TEMPLATES[sender_role]
    return _DEFAULT_NOISE


def dynamic_perturbation_attack(message: str, sender_role: str = None,
                                model: str = "llama3.1:8b", **_kwargs) -> str:
    """LLM-generated noise: produce a vague, off-topic response that sounds plausible
    but contains no actionable information about the actual case.

    Falls back to static noise if the LLM call fails.
    """
    role_desc = sender_role or "agent"
    prompt = (
        f"You are a {role_desc} in a multi-agent system. You received the following message:\n"
        f"---\n{message[:400]}\n---\n\n"
        f"Write a brief, professional response (3-5 sentences) that:\n"
        f"1. Sounds like a plausible reply from a {role_desc}\n"
        f"2. Does NOT contain any specific facts, numbers, dates, or decisions\n"
        f"3. Does NOT directly address the specifics of the message\n"
        f"4. Uses vague, generic language about 'reviewing the situation' and 'considering factors'\n"
        f"5. Does NOT push toward any particular conclusion\n\n"
        f"Respond only with the vague reply, nothing else:"
    )

    is_thinking_model = any(k in model for k in ["qwen3.5", "qwen3-", "gemma4"])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 256},
    }
    if is_thinking_model:
        payload["think"] = False

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.RequestException:
        return perturbation_attack(message, sender_role=sender_role)
