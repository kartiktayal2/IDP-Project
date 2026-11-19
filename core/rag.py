"""
core/rag.py — IDP Purchase Order RAG + Validation
=================================================

Single-file module for:
  1. Reading text from the CURRENT PO PDF (text layer; no OCR is performed).
  2. Validating extracted JSON against the current PO and scanned template.
  3. Retrieving relevant, approved historical PO examples from ChromaDB.
  4. Indexing only explicitly human-approved extraction results.

Design principles:
  - The current PO is always the source of truth.
  - Historical values are examples, never values to copy into a new PO.
  - No customer-specific field names or template layouts are hard-coded.
  - Template field definitions come from core.template.scan_template().
  - Validation is evidence-based and deliberately reports uncertainty.
  - A record is never indexed unless human_approved=True is explicitly passed.
  - RAG failures can be caught by the caller so the existing extraction flow
    can continue without RAG.

Expected extraction shape:
  {
      "some_header_key": {"value": "ABC", "confidence": 0.95},
      "line_items": [
          {"description": "Part A", "quantity": "2", "confidence": 0.9}
      ]
  }

Dependencies (install in the project's active venv):
  pip install chromadb sentence-transformers pdfplumber

Optional OCR is intentionally NOT included. If a scanned PDF has no selectable
text, this module reports that fact instead of pretending OCR was performed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("idp.rag")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PROJECT_ROOT / "rag_store"
COLLECTION_NAME = "idp_purchase_orders"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_TOP_K = 4
DEFAULT_MAX_DISTANCE = 0.55  # cosine distance; lower is more similar
MAX_PDF_TEXT_CHARS = 100_000

# Keep this false by default. A PO for one customer must not casually retrieve
# another customer's records just because the text looks semantically similar.
ALLOW_CROSS_TEMPLATE_FALLBACK = False


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _normalise(value: Any) -> str:
    """Normalise text for conservative evidence comparisons."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _normalise_words(value: Any) -> list[str]:
    if value is None:
        return []
    return re.findall(r"[a-z0-9]+", str(value).casefold())


def _unwrap_value(field: Any) -> Any:
    """Support the IDP's {value, confidence} format and plain scalar values."""
    if isinstance(field, dict) and "value" in field:
        return field.get("value")
    return field


def _field_confidence(field: Any) -> float | None:
    if not isinstance(field, dict):
        return None
    value = field.get("confidence")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _safe_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", str(value).casefold()).strip("_") or "unknown"


def _template_key(template_path: str | Path | None, customer_key: str | None = None) -> str:
    """Return a stable customer/template key, independent of download filename.

    Explicit customer keys and customer-folder names take precedence. For
    templates stored directly in a generic folder (for example po-templates),
    use a fingerprint of the workbook's actual cell structure/content so that
    SharePoint files named sp_download_<timestamp>.xlsx still match when they
    represent the same template.
    """
    if customer_key:
        return _safe_key(customer_key)

    if not template_path:
        return "unspecified"

    path = Path(template_path)
    parent = path.parent.name.strip()
    if parent and parent.casefold() not in {
        "templates", "po-templates", "po_templates", "template"
    }:
        return _safe_key(parent)

    try:
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=False)
        digest = hashlib.sha256()
        for worksheet in workbook.worksheets:
            digest.update(worksheet.title.strip().casefold().encode("utf-8"))
            for row in worksheet.iter_rows():
                values = [" ".join(str(cell.value).split()) if cell.value is not None else "" for cell in row]
                if any(values):
                    digest.update("\x1f".join(values).casefold().encode("utf-8"))
                    digest.update(b"\n")
        workbook.close()
        return "template_" + digest.hexdigest()[:24]
    except Exception as exc:
        # Keep the fallback explicit in logs; filename keys are not stable
        # across repeated SharePoint downloads.
        logger.warning(
            "Could not fingerprint template '%s'; falling back to filename key: %s",
            path.name, exc,
        )
        return _safe_key(path.stem)


def _record_id(pdf_path: str | Path, extracted_data: dict[str, Any]) -> str:
    """
    Stable content-aware ID. Re-indexing the same file/result updates rather
    than duplicates the same chunks.
    """
    path = Path(pdf_path)
    try:
        stat_part = f"{path.resolve()}::{path.stat().st_size}::{path.stat().st_mtime_ns}"
    except OSError:
        stat_part = str(path)
    digest = hashlib.sha256(
        (stat_part + "::" + _safe_json(extracted_data)).encode("utf-8")
    ).hexdigest()
    return digest[:24]


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def read_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract selectable text from a PDF using pdfplumber.

    This is not OCR. For scanned/image-only PDFs the returned text may be
    empty; caller should route those documents through the project's OCR
    capability if/when one is added.
    """
    import pdfplumber

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PO PDF not found: {path}")

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text(layout=True) or page.extract_text() or ""
            except Exception as exc:
                logger.warning("Could not read text from PDF page %s: %s", page_number, exc)
                page_text = ""
            if page_text.strip():
                pages.append(f"[PAGE {page_number}]\n{page_text.strip()}")

    return "\n\n".join(pages).strip()[:MAX_PDF_TEXT_CHARS]


# Backward-compatible private alias used by the earlier rag.py draft.
_read_pdf_text = read_pdf_text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _value_occurs_in_text(document_text: str, value: Any) -> bool:
    """
    Check whether an extracted value has textual evidence in the PDF.

    Short values are matched as complete tokens to avoid false substring
    matches (for example, matching '6' inside '20661').
    Longer values use normalised substring matching and token coverage.
    """
    candidate = _normalise(value)
    if not candidate or not document_text:
        return False

    # Short values must match complete PDF text tokens.
    if len(candidate) < 3:
        value_tokens = _normalise_words(value)
        document_tokens = set(_normalise_words(document_text))

        return (
            len(value_tokens) == 1
            and value_tokens[0] in document_tokens
        )

    haystack = _normalise(document_text)

    if candidate in haystack:
        return True

    value_tokens = _normalise_words(value)
    document_tokens = set(_normalise_words(document_text))

    if len(value_tokens) < 4:
        return False

    unique_tokens = set(value_tokens)

    if len(unique_tokens) < 4:
        return False

    matched = sum(
        1 for token in unique_tokens
        if token in document_tokens
    )

    coverage = matched / len(unique_tokens)

    return coverage >= 0.85



def _find_label_value_proximity(
    document_text: str,
    key: str,
    label: str,
    value: Any,
    window: int = 500,
) -> str:
    """
    Optional association signal, not a hard truth test.

    Customer PO labels often differ from Excel template labels. Therefore,
    absence of the template label must not automatically fail a field.
    Return values:
      - "near_label": a label/key token appears near the extracted value
      - "value_only": value is present, but no usable label proximity found
      - "not_found": value itself is not present
    """
    value_n = _normalise(value)
    text_n = _normalise(document_text)
    if len(value_n) < 3 or value_n not in text_n:
        return "not_found"

    aliases = []
    for raw in (label, key, str(key).replace("_", " ")):
        normalised = _normalise(raw)
        if len(normalised) >= 4 and normalised not in aliases:
            aliases.append(normalised)

    value_positions = [m.start() for m in re.finditer(re.escape(value_n), text_n)]
    for alias in aliases:
        label_positions = [m.start() for m in re.finditer(re.escape(alias), text_n)]
        if any(abs(vp - lp) <= window for vp in value_positions for lp in label_positions):
            return "near_label"

    return "value_only"


def _definitions(positions: dict[str, Any] | None, section: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    if not isinstance(positions, dict):
        return output

    for definition in positions.get(section, []) or []:
        if not isinstance(definition, (list, tuple)) or len(definition) < 2:
            continue
        key, label = str(definition[0]), str(definition[1])
        if section == "li_fields" and key.casefold() == "confidence":
            continue
        output.append((key, label))
    return output


def validate_extraction(
    extracted_data: dict[str, Any],
    document_text: str,
    positions: dict[str, Any] | None = None,
    *,
    human_approved: bool = False,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """
    Validate the extracted PO against the CURRENT PO's text and active
    template structure.

    Important:
      - "evidence_found" means the value text appears in the PDF text layer.
      - "association" is only a heuristic; template labels may not match PO
        labels, so "value_only" is review-needed, not automatically invalid.
      - A null field is not considered an extraction error.
      - Validation never claims that a value is semantically correct.
      - trusted_for_rag requires explicit human approval AND no hard failures.

    human_approved must only be True after a person checks the extraction
    against the original PO. Never set it automatically after model output.
    """
    report: dict[str, Any] = {
        "version": 1,
        "structure_ok": True,
        "document_text_available": bool(document_text and document_text.strip()),
        "evidence_ok": True,
        "association_review_required": False,
        "human_approved": bool(human_approved),
        "trusted_for_rag": False,
        "header_fields": {},
        "line_items": [],
        "issues": [],
        "summary": [],
    }

    if not isinstance(extracted_data, dict):
        report["structure_ok"] = False
        report["evidence_ok"] = False
        report["issues"].append({
            "severity": "error",
            "code": "invalid_root",
            "message": "Extraction result must be a JSON object/dictionary.",
        })
        return report

    if not document_text or not document_text.strip():
        report["evidence_ok"] = False
        report["issues"].append({
            "severity": "error",
            "code": "no_pdf_text",
            "message": (
                "No selectable PDF text was found. This may be a scanned PDF; "
                "OCR or manual review is required before trusting/indexing it."
            ),
        })

    header_defs = _definitions(positions, "header_fields")
    line_defs = _definitions(positions, "li_fields")

    # Header structure and evidence.
    for key, label in header_defs:
        if key not in extracted_data:
            report["structure_ok"] = False
            report["header_fields"][key] = {
                "status": "missing_key",
                "value": None,
                "template_label": label,
            }
            report["issues"].append({
                "severity": "error",
                "code": "missing_header_key",
                "field": key,
                "message": f"Required output key '{key}' is missing.",
            })
            continue

        raw_field = extracted_data.get(key)
        value = _unwrap_value(raw_field)
        confidence = _field_confidence(raw_field)

        if _is_blank(value):
            report["header_fields"][key] = {
                "status": "null",
                "value": None,
                "confidence": confidence,
                "evidence_found": None,
                "association": "not_applicable",
                "template_label": label,
            }
            continue

        evidence = _value_occurs_in_text(document_text, value)
        association = _find_label_value_proximity(
            document_text, key, label, value
        )
        status = "supported" if evidence and association == "near_label" else (
            "evidence_only" if evidence else "not_found"
        )

        report["header_fields"][key] = {
            "status": status,
            "value": str(value),
            "confidence": confidence,
            "evidence_found": evidence,
            "association": association,
            "template_label": label,
        }

        if not evidence:
            report["evidence_ok"] = False
            report["issues"].append({
                "severity": "error",
                "code": "header_value_not_found",
                "field": key,
                "message": f"'{key}' value was not found in the current PO text.",
            })
        elif association != "near_label":
            report["association_review_required"] = True
            report["issues"].append({
                "severity": "review",
                "code": "header_association_uncertain",
                "field": key,
                "message": (
                    f"'{key}' appears in the PO, but its relationship to the "
                    "field label could not be established automatically."
                ),
            })

        if min_confidence is not None and confidence is not None and confidence < min_confidence:
            report["association_review_required"] = True
            report["issues"].append({
                "severity": "review",
                "code": "low_model_confidence",
                "field": key,
                "message": f"'{key}' model confidence is below {min_confidence:.2f}.",
            })

    # Line-item structure and evidence.
    line_items = extracted_data.get("line_items")
    if not isinstance(line_items, list):
        report["structure_ok"] = False
        report["evidence_ok"] = False
        line_items = []
        report["issues"].append({
            "severity": "error",
            "code": "invalid_line_items",
            "field": "line_items",
            "message": "'line_items' must be a list.",
        })

    expected_line_keys = [key for key, _ in line_defs]
    for row_number, row in enumerate(line_items, start=1):
        row_result: dict[str, Any] = {
            "row": row_number,
            "status": "supported",
            "fields": {},
        }

        if not isinstance(row, dict):
            report["structure_ok"] = False
            report["evidence_ok"] = False
            row_result["status"] = "invalid_row"
            row_result["reason"] = "Line item must be an object/dictionary."
            report["line_items"].append(row_result)
            report["issues"].append({
                "severity": "error",
                "code": "invalid_line_item_row",
                "field": f"line_items[{row_number - 1}]",
                "message": f"Line item row {row_number} is not an object.",
            })
            continue

        for key in expected_line_keys:
            value = _unwrap_value(row.get(key))
            if _is_blank(value):
                row_result["fields"][key] = {
                    "value": None,
                    "status": "null",
                    "evidence_found": None,
                }
                continue

            evidence = _value_occurs_in_text(document_text, value)
            row_result["fields"][key] = {
                "value": str(value),
                "status": "evidence_found" if evidence else "not_found",
                "evidence_found": evidence,
            }

            if not evidence:
                row_result["status"] = "needs_review"
                report["evidence_ok"] = False
                report["issues"].append({
                    "severity": "error",
                    "code": "line_item_value_not_found",
                    "field": f"line_items[{row_number - 1}].{key}",
                    "message": (
                        f"Line item {row_number}, field '{key}' was not found "
                        "in the current PO text."
                    ),
                })

        # Preserve unexpected keys in the report as a schema warning.
        unexpected = sorted(set(row.keys()) - set(expected_line_keys) - {"confidence"})
        if unexpected:
            row_result["unexpected_keys"] = unexpected
            row_result["status"] = "needs_review"
            report["issues"].append({
                "severity": "review",
                "code": "unexpected_line_item_keys",
                "field": f"line_items[{row_number - 1}]",
                "message": f"Unexpected line-item keys: {', '.join(unexpected)}.",
            })

        row_confidence = _field_confidence(row)
        row_result["confidence"] = row_confidence
        if min_confidence is not None and row_confidence is not None and row_confidence < min_confidence:
            row_result["status"] = "needs_review"
            report["association_review_required"] = True
            report["issues"].append({
                "severity": "review",
                "code": "low_line_item_confidence",
                "field": f"line_items[{row_number - 1}]",
                "message": f"Line item {row_number} confidence is below {min_confidence:.2f}.",
            })

        report["line_items"].append(row_result)

    # Template schema check: all expected header keys and line_items exist.
    if not isinstance(extracted_data.get("line_items"), list):
        report["structure_ok"] = False

    hard_checks_passed = (
        report["structure_ok"]
        and report["evidence_ok"]
        and report["document_text_available"]
    )

    # Association warnings require human review, but are not labelled as
    # definitive extraction failures. Human approval remains mandatory.
    report["automated_checks_passed"] = bool(hard_checks_passed)
    report["trusted_for_rag"] = bool(hard_checks_passed and human_approved)

    if hard_checks_passed and not human_approved:
        report["summary"].append(
            "Evidence/structure checks passed. Human approval is still required before indexing."
        )
    if not hard_checks_passed:
        report["summary"].append(
            "Validation found one or more hard issues. Review the original PO before approval."
        )
    if report["association_review_required"]:
        report["summary"].append(
            "Some field associations or confidence scores need manual review."
        )
    if human_approved and hard_checks_passed:
        report["summary"].append(
            "Human approval recorded. This result is eligible for RAG indexing."
        )
    if not report["summary"]:
        report["summary"].append("No validation warnings were detected.")

    report["issue_count"] = len(report["issues"])
    return report


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Paragraph-aware chunking with safe overlap for oversized paragraphs."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    cleaned = re.sub(r"[ \t]+", " ", text or "").strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunks.append(paragraph[start:end])
                if end >= len(paragraph):
                    break
                start = end - overlap
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------

def _get_collection(store_path: str | Path = DEFAULT_STORE):
    """Lazy-load RAG dependencies so the base IDP app can run without them."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    store = Path(store_path)
    store.mkdir(parents=True, exist_ok=True)

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(store))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def rag_health(store_path: str | Path = DEFAULT_STORE) -> dict[str, Any]:
    """Return store status without failing the caller."""
    try:
        collection = _get_collection(store_path)
        return {
            "available": True,
            "collection": COLLECTION_NAME,
            "count": collection.count(),
            "store_path": str(Path(store_path).resolve()),
            "embedding_model": EMBEDDING_MODEL,
        }
    except Exception as exc:
        logger.exception("RAG health check failed")
        return {
            "available": False,
            "collection": COLLECTION_NAME,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_extraction_context(
    pdf_path: str | Path,
    template_path: str | Path | None = None,
    *,
    customer_key: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
    store_path: str | Path = DEFAULT_STORE,
    allow_cross_template: bool = ALLOW_CROSS_TEMPLATE_FALLBACK,
) -> str:
    """
    Retrieve approved historical examples relevant to the current PO.

    Only records indexed through index_successful_po(..., human_approved=True)
    are present in the collection. Retrieval is template/customer scoped by
    default. Cross-customer fallback is opt-in and disabled by default.
    """
    if top_k < 1:
        return ""

    query_text = read_pdf_text(pdf_path)
    if not query_text:
        logger.info("RAG retrieval skipped: no selectable PDF text.")
        return ""

    collection = _get_collection(store_path)
    total = collection.count()
    if total == 0:
        return ""

    key = _template_key(template_path, customer_key)
    where: dict[str, Any] | None = {"template_key": key}

    # ChromaDB Collection.count() does not accept a `where` argument.
    # Collection.get() applies the metadata filter and always returns IDs;
    # request only metadata to avoid loading chunk text/embeddings just to count.
    scoped_result = collection.get(where=where, include=["metadatas"])
    scoped_count = len(scoped_result.get("ids") or [])
    if scoped_count == 0:
        if not allow_cross_template:
            logger.info("No historical RAG records for template key '%s'.", key)
            return ""
        where = None
        available_count = total
    else:
        available_count = scoped_count

    result = collection.query(
        query_texts=[query_text[:12000]],
        n_results=min(top_k * 3, available_count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    # Multiple chunks can belong to the same historical PO. Return each
    # historical PO once, choosing its best matching chunk.
    selected: dict[str, dict[str, Any]] = {}
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        if max_distance is not None and distance is not None and float(distance) > max_distance:
            continue

        record_id = str(metadata.get("record_id") or f"result_{index}")
        candidate = {
            "document": document,
            "metadata": metadata,
            "distance": distance,
        }
        previous = selected.get(record_id)
        if previous is None or (
            distance is not None
            and (previous["distance"] is None or float(distance) < float(previous["distance"]))
        ):
            selected[record_id] = candidate

        if len(selected) >= top_k:
            break

    if not selected:
        return ""

    sections: list[str] = []
    for number, item in enumerate(selected.values(), start=1):
        metadata = item["metadata"]
        extraction_json = metadata.get("extraction_json", "{}")
        try:
            historical_extraction = json.loads(extraction_json)
        except (TypeError, json.JSONDecodeError):
            historical_extraction = {}

        sections.append(
            f"[HISTORICAL EXAMPLE {number}]\n"
            f"Customer/template group: {metadata.get('template_key', 'unknown')}\n"
            f"Source: {metadata.get('source_file', 'historical PO')}\n"
            f"Similarity distance: {item['distance']}\n"
            f"Relevant historical PO text excerpt:\n{item['document']}\n"
            f"Historical structured extraction (reference only):\n"
            f"{_safe_json(historical_extraction)}"
        )

    return (
        "RETRIEVED HISTORICAL PURCHASE-ORDER EXAMPLES — REFERENCE ONLY\n"
        "Use these examples only to understand recurring document structure, "
        "supplier terminology, label variations, section boundaries and table "
        "patterns. They are NOT the current purchase order.\n\n"
        "STRICT SOURCE-OF-TRUTH RULES:\n"
        "1. Extract values only from the CURRENT PO being processed.\n"
        "2. NEVER copy a PO number, date, address, phone, email, item, quantity, "
        "price, tax, subtotal, total or any other value from a historical PO.\n"
        "3. Never fill a blank or missing current field using historical data.\n"
        "4. Historical examples may guide interpretation of layout/labels only.\n"
        "5. If the current PO does not support a value, follow the existing "
        "extraction prompt and return null.\n"
        "6. Do not treat a similar-looking template as proof that values match.\n\n"
        + "\n\n".join(sections)
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_successful_po(
    pdf_path: str | Path,
    extracted_data: dict[str, Any],
    template_path: str | Path | None = None,
    *,
    customer_key: str | None = None,
    validation_report: dict[str, Any] | None = None,
    human_approved: bool = False,
    store_path: str | Path = DEFAULT_STORE,
    chunk_size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """
    Index a successfully processed PO.

    IMPORTANT: human_approved must come from a real user review/approval
    action. This function refuses to index when it is False. A caller must
    not pass True immediately after model extraction.

    validation_report should be the output of validate_extraction() for the
    same PDF and extracted_data. It is required as an additional safety gate.
    Returns the number of chunks indexed; returns 0 when safely skipped.
    """
    if not human_approved:
        logger.warning("RAG indexing skipped: explicit human approval is missing.")
        return 0

    if not isinstance(validation_report, dict):
        logger.warning("RAG indexing skipped: validation report is missing.")
        return 0

    if not validation_report.get("automated_checks_passed", False):
        logger.warning("RAG indexing skipped: automated validation did not pass.")
        return 0

    text = read_pdf_text(pdf_path)
    if not text:
        logger.warning("RAG indexing skipped: no selectable PDF text.")
        return 0

    if not isinstance(extracted_data, dict):
        logger.warning("RAG indexing skipped: extraction is not a dictionary.")
        return 0

    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    collection = _get_collection(store_path)
    key = _template_key(template_path, customer_key)
    source_path = Path(pdf_path)
    rid = _record_id(source_path, extracted_data)
    extraction_json = _safe_json(extracted_data)

    ids = [f"{rid}_{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "record_id": rid,
            "source_file": source_path.name[:500],
            "template_key": key,
            "chunk_index": index,
            "human_approved": True,
            "validation_version": str(validation_report.get("version", 1)),
            # Chroma metadata values must be scalar. JSON string is supported.
            "extraction_json": extraction_json,
        }
        for index in range(len(chunks))
    ]

    # Upsert makes retries idempotent for the same source/result.
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    logger.info(
        "Indexed approved PO '%s': %s chunk(s), template_key=%s",
        source_path.name, len(chunks), key
    )
    return len(chunks)


def delete_indexed_po(
    record_id: str,
    store_path: str | Path = DEFAULT_STORE,
) -> None:
    """Delete all chunks belonging to one record ID."""
    if not record_id or not re.fullmatch(r"[a-f0-9]{24}", record_id):
        raise ValueError("record_id must be a 24-character hexadecimal ID")
    collection = _get_collection(store_path)
    collection.delete(where={"record_id": record_id})


# ---------------------------------------------------------------------------
# Convenience pipeline helpers
# ---------------------------------------------------------------------------

def prepare_rag_context(
    pdf_path: str | Path,
    template_path: str | Path | None = None,
    *,
    customer_key: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    store_path: str | Path = DEFAULT_STORE,
) -> str:
    """
    Safe convenience wrapper for retrieval.

    RAG is optional: on retrieval/dependency/store errors it logs the issue
    and returns an empty string, allowing the existing extraction prompt to
    run unchanged.
    """
    try:
        return retrieve_extraction_context(
            pdf_path,
            template_path,
            customer_key=customer_key,
            top_k=top_k,
            store_path=store_path,
        )
    except Exception:
        logger.exception("RAG retrieval failed; continuing without RAG context.")
        return ""


def validate_current_po(
    pdf_path: str | Path,
    extracted_data: dict[str, Any],
    positions: dict[str, Any] | None = None,
    *,
    human_approved: bool = False,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Read the current PDF and validate its extraction in one call."""
    try:
        document_text = read_pdf_text(pdf_path)
    except Exception as exc:
        logger.exception("Could not read current PO for validation.")
        return {
            "version": 1,
            "structure_ok": isinstance(extracted_data, dict),
            "document_text_available": False,
            "evidence_ok": False,
            "association_review_required": True,
            "human_approved": bool(human_approved),
            "automated_checks_passed": False,
            "trusted_for_rag": False,
            "header_fields": {},
            "line_items": [],
            "issues": [{
                "severity": "error",
                "code": "pdf_read_error",
                "message": str(exc),
            }],
            "summary": ["Could not read the current PO; manual review is required."],
            "issue_count": 1,
        }

    return validate_extraction(
        extracted_data,
        document_text,
        positions,
        human_approved=human_approved,
        min_confidence=min_confidence,
    )


# Public names for import stability.
__all__ = [
    "read_pdf_text",
    "validate_extraction",
    "validate_current_po",
    "retrieve_extraction_context",
    "prepare_rag_context",
    "index_successful_po",
    "delete_indexed_po",
    "rag_health",
]
