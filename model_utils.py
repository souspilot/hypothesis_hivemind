"""
Unified model interface — all models now route through OpenRouter using
OPENROUTER_API_KEY, via two different OpenRouter endpoints:

AnthropicCachedModel  — Claude models, via OpenRouter's "Anthropic Skin"
                         (https://openrouter.ai/api). This speaks Anthropic's
                         native Messages protocol end-to-end, NOT the
                         OpenAI-translated one -- so cache_control blocks are
                         honored exactly as they would be calling Anthropic
                         directly. This matters: several other projects have
                         shipped bugs where routing Claude through OpenRouter's
                         OpenAI-compatible endpoint silently drops cache
                         markers (system prompt gets sent as a plain
                         {"role": "system"} message with no cache_control
                         support in that wire format), paying full price on
                         every one of the N_SAMPLES calls instead of getting
                         the 90% cache-hit discount from calls 2-10 onward.
                         Using the native-protocol endpoint sidesteps that
                         whole class of bug rather than working around it.

OpenRouterModel        — everything else (OpenAI, Google, Moonshot), via
                         OpenRouter's standard OpenAI-compatible endpoint
                         (https://openrouter.ai/api/v1). No prompt caching
                         applied here -- these weren't cached in the original
                         script either (OpenAIModel had no caching), so this
                         isn't a regression for them.

IMPORTANT: the exact OpenRouter model slugs below (e.g. whether Anthropic
version numbers use dots or dashes: "claude-sonnet-4.6" vs "claude-sonnet-4-6")
could not be verified with full certainty against OpenRouter's live catalog.
Before running a real batch, verify each slug at https://openrouter.ai/models
and run the smoke test at the bottom of this file -- a wrong slug fails
loudly (400 error) rather than silently, but better to catch it on 1 call
than discover it 6,840 calls in.
"""

import os

import anthropic
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Anthropic Skin: native Anthropic Messages protocol, cache_control honored.
OPENROUTER_ANTHROPIC_BASE_URL = "https://openrouter.ai/api"
# Standard OpenAI-compatible endpoint, for every non-Anthropic provider.
OPENROUTER_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Models list — edit here to add / remove models across all scripts.
# All entries are now OpenRouter "provider/model" slugs. build_model() routes
# anthropic/* to the Anthropic Skin (preserves caching), everything else to
# the OpenAI-compatible endpoint.
# ---------------------------------------------------------------------------

MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5",
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
# Claude — via OpenRouter's Anthropic Skin, prompt caching preserved
# ---------------------------------------------------------------------------

class AnthropicCachedModel(BaseModel):
    def __init__(self, model_id: str) -> None:
        self._client = anthropic.Anthropic(
            base_url=OPENROUTER_ANTHROPIC_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        # model_id is the full OpenRouter slug, e.g. "anthropic/claude-sonnet-4.6" --
        # OpenRouter needs the provider prefix even on the Anthropic-protocol
        # endpoint, since one endpoint still fronts multiple upstream
        # providers (Anthropic direct, Bedrock, Vertex) behind the scenes.
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
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Everything else — OpenAI, Google, Moonshot — via OpenRouter's
# OpenAI-compatible endpoint. No caching applied (matches prior behavior
# for OpenAI models; Google/Moonshot were never cached either).
# ---------------------------------------------------------------------------

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
    """anthropic/* -> Anthropic Skin (cached). Everything else -> OpenAI-compat endpoint."""
    provider = model_id.split("/", 1)[0]
    if provider == "anthropic":
        return AnthropicCachedModel(model_id)
    return OpenRouterModel(model_id)


def build_all_models() -> dict[str, BaseModel]:
    """Instantiate every model in MODELS. Call once at startup."""
    return {model_id: build_model(model_id) for model_id in MODELS}


# ---------------------------------------------------------------------------
# Smoke test — run this file directly for a 1-call-per-model sanity check
# before committing to a full N_SAMPLES x papers x models batch.
#   python model_utils.py
# ---------------------------------------------------------------------------

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