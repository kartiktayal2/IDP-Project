"""
Approval-gated RAG indexing for one verified IDP purchase order.
Run from the IDP project root with the project's venv activated.
Accepts an extracted JSON file OR a saved console log containing
"MISTRAL RAW ANNOTATION".
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path




def load_extraction(source: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Extraction/log file not found: {source}")

    content = source.read_text(encoding="utf-8-sig")

    def parse_json_object(block: str) -> dict:
        block = block.strip()

        obj = json.loads(block)

        # The Mistral log stores JSON as an escaped JSON string.
        if isinstance(obj, str):
            obj = json.loads(obj)

        if not isinstance(obj, dict):
            raise ValueError("Extracted PO data must be a JSON object.")

        return obj

    # 1. Check known log formats FIRST.
    # This prevents accidentally parsing the schema JSON.
    log_formats = [
        (
            "========== MISTRAL RAW ANNOTATION ==========",
            "============================================",
        ),
        (
            "========== RAW MODEL RESPONSE ==========",
            "========== END RAW MODEL RESPONSE ==========",
        ),
    ]

    for start_marker, end_marker in log_formats:
        if start_marker not in content:
            continue

        block = content.split(start_marker, 1)[1]

        if end_marker in block:
            block = block.split(end_marker, 1)[0]

        try:
            data = parse_json_object(block)

            if "line_items" not in data or not isinstance(data["line_items"], list):
                raise ValueError("Extraction does not contain a valid line_items list.")

            print("[index] Correct extraction block loaded.")
            print("[index] Header fields:", len(data) - 1)
            print("[index] Line items:", len(data["line_items"]))

            return data

        except (json.JSONDecodeError, ValueError):
            continue

    # 2. Fall back to standalone JSON files.
    try:
        data = parse_json_object(content)

        if "line_items" in data and isinstance(data["line_items"], list):
            return data

    except (json.JSONDecodeError, ValueError):
        pass

    raise ValueError(
        "Could not find valid PO extraction JSON. "
        "Check the Mistral/Cohere annotation block or standalone JSON file."
    )



def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and manually approve one PO before RAG indexing.")
    ap.add_argument("--po", required=True, help="Original PO PDF path")
    ap.add_argument("--template", required=True, help="Exact Excel template path used for extraction")
    ap.add_argument("--extraction", required=True, help="Extraction JSON or saved console log path")
    ap.add_argument("--customer-key", default=None, help="Optional customer grouping key")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    po = Path(args.po).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    extraction_file = Path(args.extraction).expanduser().resolve()
    for label, path in (("PO PDF", po), ("template", template), ("extraction/log", extraction_file)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    # Uses the same project config and template scanner as main.py.
    from main import CFG
    from core.template import scan_template
    from core.rag import validate_current_po, index_successful_po, rag_health

    data = load_extraction(extraction_file)
    positions = scan_template(str(template), CFG)

    print("\\n[1/4] Validating extraction against the PDF text and template...")
    report = validate_current_po(str(po), data, positions)
    for line in report.get("summary", []):
        print(" -", line)
    print("Automated checks passed:", report.get("automated_checks_passed", False))
    print("Association review required:", report.get("association_review_required", False))
    print("Issue count:", report.get("issue_count", len(report.get("issues", []))))

    for issue in report.get("issues", []):
        print(
            f" [{str(issue.get('severity', 'info')).upper()}] "
            f"{issue.get('field', '-')}: {issue.get('message', issue)}"
        )

    if not report.get("automated_checks_passed", False):
        print("\\nSTOPPED: hard validation checks failed. Nothing was indexed.")
        return 2

    print("\\n[2/4] Checking ChromaDB/RAG store...")
    health = rag_health()
    if not health.get("available"):
        print("STOPPED: RAG store unavailable:", health.get("error", "unknown error"))
        return 3
    print("Collection:", health.get("collection"))
    print("Existing chunks:", health.get("count"))
    print("Store:", health.get("store_path"))

    print("\\n[3/4] Human approval")
    print("Only approve if you personally compared the extracted fields and line items with the original PO.")
    answer = input('Type INDEX to approve this PO for RAG: ').strip()
    if answer != "INDEX":
        print("Approval not given. Nothing was indexed.")
        return 0

    print("\\n[4/4] Indexing...")
    count = index_successful_po(
        pdf_path=str(po),
        extracted_data=data,
        template_path=str(template),
        customer_key=args.customer_key,
        validation_report=report,
        human_approved=True,
    )
    if count <= 0:
        print("No chunks were indexed. Check the RAG logs.")
        return 4
    print(f"SUCCESS: indexed {count} chunk(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
