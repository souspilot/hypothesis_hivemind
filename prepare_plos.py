"""Select the study's PLOS articles from processed JSON for generation.

Existing training files are preserved. A conflicting file requires manual
review rather than an automatic overwrite.
"""

import argparse
import json
import shutil
from pathlib import Path

from config import DATASETS


def prepare(source: Path, destination: Path, check: bool = False) -> int:
    dataset = DATASETS["plos"]
    rows = json.loads(dataset.metadata_path.read_text())
    papers = [source / f"{row['id']}.json" for row in rows if row["id"] not in dataset.skip_ids()]
    pending = []
    for path in papers:
        record = json.loads(path.read_text())
        if not record.get("title") or not isinstance(record.get("pdf_parse", {}).get("body_text"), list):
            raise ValueError(f"Incomplete article JSON: {path}")
        target = destination / path.name
        if target.exists():
            if json.loads(target.read_text()) != record:
                raise ValueError(f"Existing training file differs: {target}")
        else:
            pending.append(path)
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
        for path in pending:
            shutil.copy2(path, destination / path.name)
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATASETS["plos"].papers_dir.parent / "processed")
    parser.add_argument("--out", type=Path, default=DATASETS["plos"].papers_dir)
    parser.add_argument("--check", action="store_true", help="validate inputs without copying files")
    args = parser.parse_args()
    count = prepare(args.source, args.out, args.check)
    print(f"{count} article files {'to copy' if args.check else 'copied'}")


if __name__ == "__main__":
    main()
