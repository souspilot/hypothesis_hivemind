import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

import anthropic

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

TRAIN_DIR = Path("data/train")
OUTPUT_DIR = Path("data/experiments_summary")


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


def process_paper(client: anthropic.Anthropic, paper_path: Path) -> dict | None:
    paper_id = paper_path.stem
    output_path = OUTPUT_DIR / f"{paper_id}.json"

    if output_path.exists():
        print(f"  [skip] {paper_id} already processed")
        return None

    with open(paper_path) as f:
        data = json.load(f)

    paper_text = extract_paper_text(data)
    prompt = TASK_PROMPT + paper_text

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    text_content = next(
        (b.text for b in response.content if b.type == "text"), ""
    )

    result = parse_json_response(text_content)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    load_dotenv()

    client = anthropic.Anthropic()
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
