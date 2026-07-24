#!/usr/bin/env python3
"""
Converts a folder of JATS-XML papers (the format PLOS, PMC, etc. publish in)
into S2ORC-style JSON files matching the schema your summarizer script reads.

Why this beats PDF extraction: JATS XML is the publisher's actual structured
source, not a rendered page image with no memory of paragraph/section
boundaries. Every paragraph is already wrapped in <p>, every section already
has a <title>, and reference markers are already tagged with <xref>. We're
just walking a tree that's already correctly shaped, not reconstructing one.
"""

import json
from pathlib import Path
from lxml import etree

INPUT_DIR = Path("data2/xml")     # folder of source .xml (JATS) files
OUTPUT_DIR = Path("data2/processed")  # matches TRAIN_DIR in the summarizer script


def get_text(el) -> str:
    """
    Flatten an XML element's text (including nested tags like <italic> or
    <xref>) into one clean string. itertext() walks every text node inside
    the element, like reading a sentence aloud even when some words are
    italicized or superscripted -- the formatting tags disappear, the
    words remain, in order.
    """
    return " ".join("".join(el.itertext()).split())


def extract_title(root, ns) -> str:
    node = root.find(".//article-title", ns)
    return get_text(node) if node is not None else ""


def extract_doi(root, ns) -> str:
    node = root.find('.//article-id[@pub-id-type="doi"]', ns)
    return node.text.strip() if node is not None and node.text else ""


def extract_authors(root, ns) -> list[dict]:
    """
    Only pull contribs from <front>. This matters because some JATS files
    embed extra material -- peer-review history, production/copyediting
    metadata -- elsewhere in the document (e.g. in a <sub-article> or
    <back> section), and those sections can contain their OWN
    contrib-group/contrib[@contrib-type="author"] elements for reviewers,
    editors, or production staff. Searching the whole tree with ".//"
    would sweep those in too. Scoping to <front> restricts us to the
    actual byline, the same way you'd search a company's front-desk
    sign-in sheet rather than every sign-in sheet in the building.
    """
    authors = []
    front = root.find(".//front", ns)
    if front is None:
        return authors

    # Take only the FIRST <contrib-group> that contains an author contrib --
    # this is the original byline. If the document has additional
    # contrib-groups later in <front> (e.g. attached to a correction
    # notice, production credit, or later article version), those are
    # skipped rather than merged in.
    # (Note: lxml's find()/findall() use a restricted path syntax that
    # doesn't support nested predicates like "group[child[@attr]]", so we
    # do the "does this group contain an author" check in plain Python
    # instead of trying to express it as one XPath query.)
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
    # PLOS papers often have two <abstract> tags: the real one and a
    # shorter "toc" (table-of-contents) blurb. We want the real one --
    # i.e. the one WITHOUT abstract-type="toc".
    for abstract_el in root.findall(".//abstract", ns):
        if abstract_el.get("abstract-type") != "toc":
            paras = [get_text(p) for p in abstract_el.findall(".//p", ns)]
            return " ".join(paras)
    return ""


def extract_body_paragraphs(root, ns) -> list[dict]:
    """
    Walk every <sec> in <body> and pull out its <p> paragraphs, tagging
    each with the section title it belongs to (e.g. "Introduction",
    "Methods" -> "Study sites and sampling"). We only look at direct
    child <p> of each <sec> (via ".//p" scoped per-section further down)
    so figure captions and table content don't get swept in as if they
    were narrative paragraphs.
    """
    paragraphs = []
    body = root.find(".//body", ns)
    if body is None:
        return paragraphs

    for sec in body.iter("sec"):
        title_el = sec.find("title", ns)
        section_title = get_text(title_el) if title_el is not None else None

        # Only direct <p> children of this <sec> -- nested <sec>s will be
        # visited separately by the outer iter("sec") loop, so we don't
        # want to double-count their paragraphs here.
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
    ns = {}  # JATS elements we're using here (article-title, sec, p, etc.)
             # aren't in a namespace in this file, so an empty map is correct;
             # only mml:/xlink: prefixed tags are namespaced, and we don't
             # need to address those directly since itertext() ignores tags.

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