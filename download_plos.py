#!/usr/bin/env python3
"""
Downloads PLOS Biology article PDFs and manuscript XML (JATS) files given a
list of DOIs.

Think of this like a mail-order form: for every DOI (the article's "shipping
label"), we build the PLOS download-URL envelope, put it in the mail
(an HTTP GET request), and save whatever comes back to disk. We do this
twice per DOI now -- once asking for the PDF, once asking for the XML --
using the same underlying delivery logic for both, just with a different
address label and box each time.
"""

import time
import requests
from pathlib import Path

# ---- Config -----------------------------------------------------------
DOI_LIST_FILE = "plos_2026_apr_to_jun.txt"          # one DOI per line
DELAY_SECONDS = 1.5                 # be polite to the server between requests
TIMEOUT_SECONDS = 30

HEADERS = {
    # Some servers refuse requests that don't look like they came from a browser.
    # This is like putting a return address on the envelope so it doesn't get
    # tossed as junk mail.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Each entry describes one file type to fetch per DOI: where it lands,
# what URL shape to request, what extension to save it as, and what
# Content-Type we expect back (used as a sanity check that we didn't
# just download an HTML error page by mistake).
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
    # The part after the last slash (e.g. journal.pbio.3003762) is unique
    # and filesystem-safe, so we just use that.
    return doi.split("/")[-1] + extension


def download_one(doi: str, file_type: dict) -> bool:
    """
    Fetch a single DOI's file for one file type (pdf or xml). Same delivery
    mechanism regardless of type -- only the URL, extension, and expected
    Content-Type change based on what's in file_type.
    """
    url = file_type["url_template"].format(doi=doi)
    out_dir = file_type["output_dir"]
    out_path = out_dir / doi_to_filename(doi, file_type["extension"])

    if out_path.exists():
        print(f"  [skip] {doi} -> already downloaded")
        return True

    try:
        # stream=True means: don't load the whole file into memory at once,
        # read it in chunks -- like pouring water through a hose instead
        # of trying to catch it all in one bucket.
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