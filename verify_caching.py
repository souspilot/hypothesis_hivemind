#!/usr/bin/env python3
"""
Verifies prompt caching is actually active for Claude models routed through
OpenRouter's Anthropic Skin.

How the check works: we send the same long static_text block twice, both
times marked with cache_control. Anthropic's response includes a `usage`
object that separately reports:
  - cache_creation_input_tokens: tokens WRITTEN to the cache this call
  - cache_read_input_tokens:     tokens READ from an existing cache entry
  - input_tokens:                tokens that were neither written nor read
                                  from cache (charged at normal price)

Expected pattern if caching is working:
  Call 1 (first time seeing this static_text): cache_creation_input_tokens > 0,
                                                cache_read_input_tokens == 0
  Call 2 (same static_text again):             cache_creation_input_tokens == 0,
                                                cache_read_input_tokens > 0

If instead BOTH calls show cache_read_input_tokens == 0 and
cache_creation_input_tokens == 0 every time, caching isn't happening --
the model is treating each call as fully fresh, and you're paying full
input-token price on all N_SAMPLES calls, not just the first.
"""

from test_models import build_model

MODEL_ID = "anthropic/claude-sonnet-4.6"

# Needs to be long enough to be worth caching at all -- Anthropic's minimum
# cacheable block size is 1024 tokens for Sonnet/Opus (2048 for Haiku).
# Repeating a sentence is a cheap way to pad past that threshold for testing.
STATIC_TEXT = "This is a long static block used only to test caching. " * 200


def run_once(model, label: str):
    response = model._client.messages.create(
        model=model._model_id,
        max_tokens=10,
        system=[{
            "type": "text",
            "text": "Reply with one word.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": STATIC_TEXT,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Say hi."},
            ],
        }],
    )
    u = response.usage
    print(f"{label}:")
    print(f"  input_tokens               = {u.input_tokens}")
    print(f"  cache_creation_input_tokens = {getattr(u, 'cache_creation_input_tokens', 'N/A')}")
    print(f"  cache_read_input_tokens     = {getattr(u, 'cache_read_input_tokens', 'N/A')}")
    print()
    return u


def main():
    model = build_model(MODEL_ID)

    u1 = run_once(model, "Call 1 (first time seeing this text)")
    u2 = run_once(model, "Call 2 (same text again)")

    cache_write = getattr(u1, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(u2, "cache_read_input_tokens", 0) or 0

    print("=" * 60)
    if cache_write > 0 and cache_read > 0:
        print("PASS: caching is active -- call 1 wrote to cache, call 2 read from it.")
    elif cache_write == 0 and cache_read == 0:
        print("FAIL: no cache activity detected on either call. Caching is NOT working.")
        print("Check: is base_url really 'https://openrouter.ai/api' (not /v1)?")
        print("Check: is STATIC_TEXT actually long enough to exceed the cacheable minimum?")
    else:
        print("UNCLEAR: partial signal. Re-run -- OpenRouter provider routing can vary "
              "between calls, and a different upstream provider on call 2 would miss the cache.")


if __name__ == "__main__":
    main()