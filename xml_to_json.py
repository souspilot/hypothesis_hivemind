#!/usr/bin/env python3
"""Convert PLOS JATS XML into the article JSON schema used by the generation scripts."""

import json
from pathlib import Path
from lxml import etree

INPUT_DIR = Path("data2/xml")     # folder of source .xml (JATS) files
OUTPUT_DIR = Path("data2/processed")  # matches TRAIN_DIR in the summarizer script


def get_text(el) -> str:
    """Return normalized text from an XML element, including nested elements."""
    return " ".join("".join(el.itertext()).split())


def extract_title(root, ns) -> str:
    node = root.find(".//article-title", ns)
    return get_text(node) if node is not None else ""


def extract_doi(root, ns) -> str:
    node = root.find('.//article-id[@pub-id-type="doi"]', ns)
    return node.text.strip() if node is not None and node.text else ""


def extract_authors(root, ns) -> list[dict]:
    """Read the first author group in the article front matter.

    Exclude reviewer and production credits elsewhere in the document."""
    authors = []
    front = root.find(".//front", ns)
    if front is None:
        return authors

    # Use the first group containing an author; ignore later credit groups.
    first_author_group = None
    for group in front.findall(".//contrib-group", ns):
        if group.find('contrib[@contrib-type="author"]', ns) is not None:
            first_author_group = group
            break
    if first_author_group is None:
        return authors

    for contrib in first_author_group.findall('contrib[@contrib-type="author"]', ns):
        surname_el = contrib.find(".//surname", ns)
        given_el = contrib.find(".//given-names", ns)
        authors.append({
            "first": get_text(given_el) if given_el is not None else "",
            "last": get_text(surname_el) if surname_el is not None else "",
        })
    return authors


def extract_abstract(root, ns) -> str:
    # Exclude the table-of-contents abstract.
    for abstract_el in root.findall(".//abstract", ns):
        if abstract_el.get("abstract-type") != "toc":
            paras = [get_text(p) for p in abstract_el.findall(".//p", ns)]
            return " ".join(paras)
    return ""


def extract_body_paragraphs(root, ns) -> list[dict]:
    """Extract direct paragraph children of each body section, with section titles.

    Nested sections are visited separately to avoid duplicate paragraphs."""
    paragraphs = []
    body = root.find(".//body", ns)
    if body is None:
        return paragraphs

    for sec in body.iter("sec"):
        title_el = sec.find("title", ns)
        section_title = get_text(title_el) if title_el is not None else None

        # Nested sections are handled by the outer loop.
        for p in sec.findall("p", ns):
            text = get_text(p)
            if text:
                paragraphs.append({
                    "text": text,
                    "cite_spans": [],
                    "ref_spans": [],
                    "eq_spans": [],
                    "section": section_title,
                    "sec_num": None,
                })

    return paragraphs


def convert_one(xml_path: Path, out_dir: Path) -> Path:
    paper_id = xml_path.stem
    out_path = out_dir / f"{paper_id}.json"

    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    ns = {}  # Article and paragraph tags are not namespaced.

    record = {
        "paper_id": paper_id,
        "header": {
            "generated_with": "jats-xml-extractor",
            "source_file": xml_path.name,
        },
        "title": extract_title(root, ns),
        "doi": extract_doi(root, ns),
        "authors": extract_authors(root, ns),
        "abstract_text": extract_abstract(root, ns),
        "pdf_parse": {
            "paper_id": paper_id,
            "body_text": extract_body_paragraphs(root, ns),
        },
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {INPUT_DIR}/. Writing JSON to {OUTPUT_DIR}/\n")

    success, errors = 0, 0
    for i, xml_path in enumerate(xml_files, 1):
        out_path = OUTPUT_DIR / f"{xml_path.stem}.json"
        if out_path.exists():
            print(f"[{i}/{len(xml_files)}] [skip] {xml_path.name} already converted")
            continue
        try:
            convert_one(xml_path, OUTPUT_DIR)
            print(f"[{i}/{len(xml_files)}] [ok]   {xml_path.name}")
            success += 1
        except Exception as e:
            print(f"[{i}/{len(xml_files)}] [fail] {xml_path.name} -> {e}")
            errors += 1

    print(f"\nDone. {success} converted, {errors} errors.")


if __name__ == "__main__":
    main()
