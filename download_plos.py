#!/usr/bin/env python3
"""Download PLOS Biology PDFs and JATS XML for the DOIs in plos_2026_apr_to_jun.txt."""

import time
import requests
from pathlib import Path

# ---- Config -----------------------------------------------------------
DOI_LIST_FILE = "plos_2026_apr_to_jun.txt"          # one DOI per line
DELAY_SECONDS = 1.5                 # be polite to the server between requests
TIMEOUT_SECONDS = 30

HEADERS = {
    # Use a browser user agent for publisher downloads.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Download locations, URL templates, and expected content types.
FILE_TYPES = {
    "pdf": {
        "output_dir": Path("data2/raw"),
        "url_template": "https://journals.plos.org/plosbiology/article/file?id={doi}&type=printable",
        "extension": ".pdf",
        "expected_content_type": "pdf",
    },
    "xml": {
        "output_dir": Path("data2/xml"),
        "url_template": "https://journals.plos.org/plosbiology/article/file?id={doi}&type=manuscript",
        "extension": ".xml",
        "expected_content_type": "xml",
    },
}


def doi_to_filename(doi: str, extension: str) -> str:
    """Turn a DOI like 10.1371/journal.pbio.3003762 into a safe filename."""
    # Use the DOI suffix as the filename.
    return doi.split("/")[-1] + extension


def download_one(doi: str, file_type: dict) -> bool:
    """Download one article file, checking the response content type."""
    url = file_type["url_template"].format(doi=doi)
    out_dir = file_type["output_dir"]
    out_path = out_dir / doi_to_filename(doi, file_type["extension"])

    if out_path.exists():
        print(f"  [skip] {doi} -> already downloaded")
        return True

    try:
        # Stream large responses in chunks.
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, stream=True) as resp:
            resp.raise_for_status()  # raises an error if we got a 404/500/etc.

            content_type = resp.headers.get("Content-Type", "")
            expected = file_type["expected_content_type"]
            if expected not in content_type.lower():
                print(f"  [warn] {doi} -> response wasn't {expected} (Content-Type: {content_type}); skipping")
                return False

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"  [ok]   {doi} -> {out_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  [fail] {doi} -> {e}")
        return False


def main():
    for file_type in FILE_TYPES.values():
        file_type["output_dir"].mkdir(exist_ok=True)

    dois = [
        line.strip()
        for line in Path(DOI_LIST_FILE).read_text().splitlines()
        if line.strip()
    ]

    print(f"Found {len(dois)} DOIs. Downloading PDFs into '{FILE_TYPES['pdf']['output_dir']}/' "
          f"and XML into '{FILE_TYPES['xml']['output_dir']}/'...\n")

    successes, failures = 0, 0
    for i, doi in enumerate(dois, start=1):
        print(f"[{i}/{len(dois)}] {doi}")
        for kind, file_type in FILE_TYPES.items():
            print(f"  ({kind})")
            ok = download_one(doi, file_type)
            successes += ok
            failures += not ok
            time.sleep(DELAY_SECONDS)  # avoid hammering the server rapid-fire

    print(f"\nDone. {successes} succeeded, {failures} failed.")


if __name__ == "__main__":
    main()
