"""Ollama-backed MAS agent."""

import copy
import requests
import json
from typing import Optional

from src.logging.transcripts import ReceiverCall


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"


class Agent:
    """A single agent in the MAS with a role and system prompt."""

    def __init__(self, agent_id: str, role: str, system_prompt: str,
                 model: str = DEFAULT_MODEL, description: str = ""):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.description = description
        self.message_history = []
        self.last_call = None
        # Preserve every call for F2 replay.
        self.call_log = []

    def respond(self, message: str, context: Optional[dict] = None) -> str:
        """Generate a response given an incoming message.

        Args:
            message: The incoming message from another agent or the task.
            context: Optional dict with mock data the agent can reference.

        Returns:
            The agent's response string.
        """
        system = self.system_prompt
        if context:
            system += f"\n\nAvailable data:\n```json\n{json.dumps(context, indent=2)}\n```"

        messages = [{"role": "system", "content": system}]
        for h in self.message_history:
            messages.append(h)
        messages.append({"role": "user", "content": message})

        response = self._call_ollama(messages)

        # Snapshot the call before message_history changes.
        self.last_call = ReceiverCall(
            agent_id=self.agent_id,
            role=self.role,
            system=system,
            messages=copy.deepcopy(messages),
            response=response,
        )
        self.call_log.append(self.last_call)

        self.message_history.append({"role": "user", "content": message})
        self.message_history.append({"role": "assistant", "content": response})

        return response

    def _call_ollama(self, messages: list) -> str:
        """Call the Ollama API."""
        # Thinking models expose internal reasoning tokens; disable for speed.
        is_thinking_model = any(k in self.model for k in ["qwen3.5", "qwen3-", "gemma4"])
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1024,
            },
        }
        if is_thinking_model:
            payload["think"] = False
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"[ERROR: Agent {self.agent_id} failed to respond: {e}]"

    def reset(self):
        """Clear message history for a new trial."""
        self.message_history = []
        self.last_call = None
        self.call_log = []

    def __repr__(self):
        return f"Agent(id={self.agent_id}, role={self.role})"
