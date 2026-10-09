#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

from config import DATASETS, parse_dataset_args

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

# OpenRouter uses provider-prefixed model identifiers.
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"

# Allow space for the methods summary and JSON wrapper.
MAX_OUTPUT_TOKENS = 4096


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


def process_paper(client: OpenAI, paper_path: Path, output_dir: Path) -> dict | None:
    paper_id = paper_path.stem
    output_path = output_dir / f"{paper_id}.json"

    if output_path.exists():
        print(f"  [skip] {paper_id} already processed")
        return None

    with open(paper_path) as f:
        data = json.load(f)

    paper_text = extract_paper_text(data)
    prompt = TASK_PROMPT + paper_text

    # Send the system prompt through the Chat Completions message list.
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    choice = response.choices[0]
    text_content = choice.message.content or ""

    # Report truncation before attempting JSON parsing.
    if choice.finish_reason == "length":
        raise RuntimeError(
            f"Response truncated at max_tokens={MAX_OUTPUT_TOKENS} "
            f"(finish_reason='length'). Raw output was {len(text_content)} chars. "
            f"Consider raising MAX_OUTPUT_TOKENS."
        )

    try:
        result = parse_json_response(text_content)
    except json.JSONDecodeError as e:
        # Retain the raw response for diagnosing malformed JSON.
        raise RuntimeError(
            f"JSON parse failed ({e}). Raw response was: {text_content!r}"
        ) from e

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*", choices=list(DATASETS), help="default: all")
    args = parser.parse_args()
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    for dataset in parse_dataset_args(args.dataset):
        output_dir = dataset.summaries_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        paper_files = sorted(dataset.papers_dir.glob("*.json"))
        print(f"Found {len(paper_files)} papers. Saving to {output_dir}/\n")

        success, errors = 0, 0
        for i, paper_path in enumerate(paper_files, 1):
            print(f"[{i}/{len(paper_files)}] {paper_path.stem}")
            try:
                result = process_paper(client, paper_path, output_dir)
                if result:
                    print(f"  -> {result['title'][:70]}")
                    success += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                # Save error record so we know which ones failed
                error_path = output_dir / f"{paper_path.stem}.json"
                with open(error_path, "w") as f:
                    json.dump({"error": str(e), "paper_id": paper_path.stem}, f)
                errors += 1

            time.sleep(0.3)  # light rate-limit buffer

        print(f"\nDone. {success} processed, {errors} errors.")


if __name__ == "__main__":
    main()
