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
    "google/gemma-4-31b-it",
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

        # Guard added: previously this went straight to
        # response.content[0].text, which assumes the SDK call always
        # comes back with a normal Anthropic message shape. It doesn't
        # always -- OpenRouter's Anthropic Skin can return HTTP 200 with
        # response.content set to None instead of raising an SDK-level
        # error, which crashed here with a bare "'NoneType' object is
        # not subscriptable" and no way to tell what actually went wrong.
        # This surfaces the raw response instead, so the next failure
        # tells you something diagnosable (rate limit, provider routing
        # issue, a field OpenRouter's proxy dropped, etc.) rather than
        # just "NoneType".
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
            # Raised from 4096: some models on OpenRouter (e.g. Kimi
            # K2.6/K2.7) share ONE token budget between invisible
            # "thinking" and the visible answer -- like a student who
            # uses all their exam time on scratch paper and never writes
            # the final answer in the box. At 4096 tokens, a chunk of
            # samples came back with reasoning eating the whole budget
            # and finish_reason="length" before any answer text existed.
            # Doubling the ceiling gives real headroom for both, and
            # costs nothing extra for models that finish early --
            # max_tokens is a cap, you're only billed for tokens actually
            # generated.
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{static_text}\n\n{user_instruction}"},
            ],
            # NOTE: we tried extra_body={"reasoning": {"enabled": False}}
            # here previously. Some endpoints (at least one model behind
            # OpenRouter's Kimi routing) hard-reject that with a 400:
            # "Reasoning is mandatory for this endpoint and cannot be
            # disabled." So instead of forbidding reasoning outright, we
            # ask for the minimum amount via effort="low" -- this is a
            # request to spend LESS, not a demand to spend NONE, so it
            # doesn't hit the same wall. Models that don't support the
            # reasoning field at all just ignore it.
            extra_body={"reasoning": {"effort": "low"}},
        )

        choice = response.choices[0]
        content = (choice.message.content or "").strip()

        if not content:
            # Fail loudly instead of silently returning "" -- an empty
            # string looks exactly like a normal (if useless) sample once
            # it's sitting in the output JSON, with no trace of what went
            # wrong. Raising here means sample_model's existing try/except
            # catches it and records "ERROR: ..." instead, so failures are
            # visible in both the logs and the output file.
            raise RuntimeError(
                f"Empty content from {self._model_id} "
                f"(finish_reason={choice.finish_reason!r}). "
                f"Likely reasoning-token budget exhaustion or a provider-side issue."
            )

        return content


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