"""
Unified model interface with Anthropic prompt caching for Claude models.

AnthropicCachedModel  — uses Anthropic SDK directly, marks both the system
                         prompt and the large static content with cache_control
                         so repeated calls (N_SAMPLES) over the same input hit
                         the cache from the second call onward.

OpenAIModel           — thin wrapper around LangChain's init_chat_model,
                         same .generate() signature for drop-in interoperability.
"""

import anthropic
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Models list — edit here to add / remove models across all scripts
# ---------------------------------------------------------------------------

# MODELS = [
#     "anthropic:claude-sonnet-4-6",
#     "anthropic:claude-sonnet-4-5",
#     "anthropic:claude-haiku-4-5-20251001",
#     "openai:gpt-5-mini-2025-08-07",
#     "openai:gpt-5-nano-2025-08-07",
# ]
MODELS = [
    "openai:gpt-5-mini-2025-08-07",
    "openai:gpt-5-nano-2025-08-07",
    "openai:gpt-5",
]

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    @abstractmethod
    def generate(self, system: str, static_text: str, user_instruction: str) -> str:
        """
        Make a single generation call.

        Args:
            system:           System prompt (cached for Claude).
            static_text:      Large context that is the same across N_SAMPLES
                              for one paper/hypothesis (cached for Claude).
            user_instruction: Short dynamic instruction appended after static_text.
        """


# ---------------------------------------------------------------------------
# Claude — Anthropic SDK with prompt caching
# ---------------------------------------------------------------------------

class AnthropicCachedModel(BaseModel):
    def __init__(self, model_name: str) -> None:
        self._client = anthropic.Anthropic()
        self._model_name = model_name

    def generate(self, system: str, static_text: str, user_instruction: str) -> str:
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=4096,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": static_text,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                ],
            }],
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# OpenAI — LangChain wrapper
# ---------------------------------------------------------------------------

class OpenAIModel(BaseModel):
    def __init__(self, model_id: str) -> None:
        self._model = init_chat_model(model_id, temperature=1)

    def generate(self, system: str, static_text: str, user_instruction: str) -> str:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"{static_text}\n\n{user_instruction}"),
        ]
        return self._model.invoke(messages).content.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_id: str) -> BaseModel:
    """Return AnthropicCachedModel for claude-* models, OpenAIModel otherwise."""
    provider, model_name = model_id.split(":", 1)
    if provider == "anthropic":
        return AnthropicCachedModel(model_name)
    return OpenAIModel(model_id)


def build_all_models() -> dict[str, BaseModel]:
    """Instantiate every model in MODELS. Call once at startup."""
    return {model_id: build_model(model_id) for model_id in MODELS}
