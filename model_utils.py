"""
Unified model interface with Anthropic prompt caching for Claude models.

AnthropicCachedModel  — uses Anthropic SDK directly, marks both the system
                         prompt and the large static content with cache_control
                         so repeated calls (N_SAMPLES) over the same input hit
                         the cache from the second call onward.

OpenAIModel           — thin wrapper around LangChain's init_chat_model,
                         same .generate() signature for drop-in interoperability.

OpenRouterModel       — for models not natively supported by LangChain's
                         init_chat_model (Google Gemini/Gemma, Moonshot Kimi,
                         etc.). Uses the OpenAI-compatible client pointed at
                         OpenRouter, since OpenRouter fronts these providers
                         through one endpoint with "provider/model" slugs.
"""

import os

import anthropic
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI as OpenAIClient

load_dotenv()

# ---------------------------------------------------------------------------
# Models list — edit here to add / remove models across all scripts
#
# Two naming conventions coexist here, and build_model() below tells them
# apart by whether the string contains "/":
#   "provider:model"  -- routed through Anthropic's SDK directly (anthropic:)
#                         or LangChain's native provider integrations (openai:)
#   "provider/model"   -- OpenRouter slug format, routed through OpenRouter's
#                         OpenAI-compatible endpoint. Anything not natively
#                         wired into LangChain (Google, Moonshot, etc.) goes
#                         here rather than fighting an unsupported provider.
# ---------------------------------------------------------------------------

MODELS = [
    "anthropic:claude-haiku-4-5-20251001",
    "anthropic:claude-sonnet-4-5",
    "anthropic:claude-sonnet-4-6",
    "openai:gpt-5-nano-2025-08-07",
    "openai:gpt-5-mini-2025-08-07",
    "openai:gpt-5",
    # -- routed via OpenRouter (see OpenRouterModel below) --
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite",
    "google/gemma-4-31b-it:free",
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.7-code",
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
# OpenAI — LangChain wrapper (native provider integration)
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
# OpenRouter — for providers LangChain doesn't natively support
# (Google Gemini/Gemma, Moonshot Kimi, and anything else identified by a
# "provider/model" slug rather than a "provider:model" string)
# ---------------------------------------------------------------------------

class OpenRouterModel(BaseModel):
    def __init__(self, model_id: str) -> None:
        self._client = OpenAIClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self._model_id = model_id

    def generate(self, system: str, static_text: str, user_instruction: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_id,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{static_text}\n\n{user_instruction}"},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_id: str) -> BaseModel:
    """
    Return AnthropicCachedModel for "anthropic:*" models, OpenRouterModel
    for "provider/model" slugs, and OpenAIModel (native LangChain) otherwise.

    Order matters: the "/" check must come before any ":"-splitting logic,
    since "google/gemma-4-31b-it:free" contains BOTH characters -- checking
    "/" first routes it correctly as one OpenRouter slug rather than
    (incorrectly) splitting on the ":free" suffix as if it were a
    provider:model separator.
    """
    if "/" in model_id:
        return OpenRouterModel(model_id)

    provider, model_name = model_id.split(":", 1)
    if provider == "anthropic":
        return AnthropicCachedModel(model_name)
    return OpenAIModel(model_id)


def build_all_models() -> dict[str, BaseModel]:
    """Instantiate every model in MODELS. Call once at startup."""
    return {model_id: build_model(model_id) for model_id in MODELS}