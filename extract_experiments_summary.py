#!/usr/bin/env python3
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant for summarizing key details of experiments "
    "and methodologies from scientific papers."
)

TASK_PROMPT = (
    "Summarize the following research paper, focusing ONLY on this question:\n"
    "Carefully analyze ONLY the experiments performed or methods used.\n"
    "Do NOT include results, abstract, introduction, or discussion.\n\n"
    "Output MUST be valid JSON of the form:\n"
    '{\n'
    '  "title": "<paper title>",\n'
    '  "experiments_summary": "<concise summary>"\n'
    '}\n'
    "Do NOT wrap the JSON in markdown code fences.\n\n"
    "Paper text:\n"
)

TRAIN_DIR = Path("data2/train")
OUTPUT_DIR = Path("data2/experiments_summary")

# OpenRouter speaks the OpenAI Chat Completions format, not Anthropic's native
# Messages format. Model names on OpenRouter are "provider/model-name" --
# e.g. "anthropic/claude-sonnet-4-6" -- because one endpoint fronts many
# providers, so the provider prefix disambiguates which company's model
# you mean (this is different from calling Anthropic directly, where the
# plain "claude-sonnet-4-6" string was enough since there's only one provider).
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"


def extract_paper_text(data: dict) -> str:
    title = data.get("title", "")
    paragraphs = data.get("pdf_parse", {}).get("body_text", [])
    body = "\n\n".join(p["text"] for p in paragraphs)
    return f"Title: {title}\n\n{body}"


def parse_json_response(text: str) -> dict:
    """Extract JSON from response, stripping any markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def process_paper(client: OpenAI, paper_path: Path) -> dict | None:
    paper_id = paper_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    if output_path.exists():
        print(f"  [skip] {paper_id} already processed")
        return None

    with open(paper_path) as f:
        data = json.load(f)

    paper_text = extract_paper_text(data)
    prompt = TASK_PROMPT + paper_text

    # OpenAI-style chat completions: the system prompt is just another
    # message in the list (role="system"), rather than a separate
    # top-level "system" parameter like Anthropic's native API uses.
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text_content = response.choices[0].message.content or ""

    result = parse_json_response(text_content)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_files = sorted(TRAIN_DIR.glob("*.json"))
    print(f"Found {len(paper_files)} papers. Saving to {OUTPUT_DIR}/\n")

    success, errors = 0, 0
    for i, paper_path in enumerate(paper_files, 1):
        print(f"[{i}/{len(paper_files)}] {paper_path.stem}")
        try:
            result = process_paper(client, paper_path)
            if result:
                print(f"  -> {result['title'][:70]}")
                success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            # Save error record so we know which ones failed
            error_path = OUTPUT_DIR / f"{paper_path.stem}.json"
            with open(error_path, "w") as f:
                json.dump({"error": str(e), "paper_id": paper_path.stem}, f)
            errors += 1

        time.sleep(0.3)  # light rate-limit buffer

    print(f"\nDone. {success} processed, {errors} errors.")


if __name__ == "__main__":
    main()