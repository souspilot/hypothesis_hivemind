"""Model clients for OpenRouter.

Claude uses the native Anthropic Messages endpoint with prompt-cache markers.
Other models use the OpenAI-compatible endpoint. Set OPENROUTER_API_KEY
before creating clients. Run this module to make one test call per model."""

import os

import anthropic
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

import config

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Anthropic Skin: native Anthropic Messages protocol, cache_control honored.
OPENROUTER_ANTHROPIC_BASE_URL = "https://openrouter.ai/api"
# Standard OpenAI-compatible endpoint, for every non-Anthropic provider.
OPENROUTER_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

# Model order and legacy identifiers are defined in config.py.
MODELS = [m.slug for m in config.MODELS]

# Base interface

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


# Claude — via OpenRouter's Anthropic Skin, prompt caching preserved

class AnthropicCachedModel(BaseModel):
    def __init__(self, model_id: str) -> None:
        self._client = anthropic.Anthropic(
            base_url=OPENROUTER_ANTHROPIC_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        # Both endpoints require the provider-prefixed OpenRouter identifier.
        self._model_id = model_id

    def generate(self, system: str, static_text: str, user_instruction: str) -> str:
        response = self._client.messages.create(
            model=self._model_id,
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

        # Include provider response details when the content block is missing.
        if not response.content:
            raise RuntimeError(
                f"No content block from {self._model_id} "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r}, "
                f"raw_response={response!r})"
            )

        content = response.content[0].text.strip()
        if not content:
            raise RuntimeError(
                f"Empty content from {self._model_id} "
                f"(stop_reason={response.stop_reason!r})."
            )
        return content


# OpenAI, Google, and Moonshot use the OpenAI-compatible endpoint.

class OpenRouterModel(BaseModel):
    def __init__(self, model_id: str) -> None:
        self._client = OpenAIClient(
            base_url=OPENROUTER_OPENAI_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self._model_id = model_id

    def generate(self, system: str, static_text: str, user_instruction: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_id,
            # The output budget covers both reasoning and visible response tokens.
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{static_text}\n\n{user_instruction}"},
            ],
            # Some endpoints require reasoning; request low effort.
            extra_body={"reasoning": {"effort": "low"}},
        )

        choice = response.choices[0]
        content = (choice.message.content or "").strip()

        if not content:
            # The generation engine records failures as ERROR-prefixed responses.
            raise RuntimeError(
                f"Empty content from {self._model_id} "
                f"(finish_reason={choice.finish_reason!r}). "
                f"Likely reasoning-token budget exhaustion or a provider-side issue."
            )

        return content


# Factory

def build_model(model_id: str) -> BaseModel:
    """anthropic/* -> Anthropic Skin (cached). Everything else -> OpenAI-compat endpoint."""
    provider = model_id.split("/", 1)[0]
    if provider == "anthropic":
        return AnthropicCachedModel(model_id)
    return OpenRouterModel(model_id)


def build_all_models() -> dict[str, BaseModel]:
    """Instantiate every model in MODELS. Call once at startup."""
    return {model_id: build_model(model_id) for model_id in MODELS}


# Smoke test — run this file directly for a 1-call-per-model sanity check
# before committing to a full N_SAMPLES x papers x models batch.
#   python model_utils.py

if __name__ == "__main__":
    for model_id in MODELS:
        try:
            model = build_model(model_id)
            reply = model.generate(
                system="Reply with exactly one word.",
                static_text="N/A",
                user_instruction="Say hello.",
            )
            print(f"[ok]   {model_id:45} -> {reply[:60]!r}")
        except Exception as e:
            print(f"[FAIL] {model_id:45} -> {e}")
