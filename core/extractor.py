"""
core/extractor.py — Azure Edition
Supports: Mistral Azure OCR, OpenAI Azure, Phi Azure, Llama Azure.
Adding a new Azure provider: implement _extract_<name> and add to PROVIDERS.
"""

from __future__ import annotations

import base64
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests


def _extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.
    Used by LLM extraction and duplicate-header fallback.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pages_text = []

        for page in pdf.pages:
            page_text = page.extract_text(layout=True) or ""
            if page_text.strip():
                pages_text.append(page_text)

    return "\n\n".join(pages_text)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _retry(fn, retries: int, timeout_codes: tuple = (408, 503)):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = fn()
            raw, status = result[0], result[1]
            if status == 429:
                wait = 30 * attempt
                print(f"  [429] Rate limited, attempt {attempt}/{retries} — waiting {wait}s")
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Rate limited after {retries} attempts.")
            if status in timeout_codes:
                wait = 10 * attempt
                print(f"  [{status}] attempt {attempt}/{retries} — retrying in {wait}s")
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API returned {status} after {retries} attempts.")
            if status != 200:
                raise RuntimeError(f"API call failed [{status}]: {raw}")
            return raw, status
        except requests.exceptions.Timeout as e:
            last_exc = e
            wait = 10 * attempt
            print(f"  Timeout on attempt {attempt}/{retries} — retrying in {wait}s")
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"All {retries} attempts timed out.") from last_exc


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        clean = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return json.loads(clean)


def _loose(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", k.lower())


def _unwrap(data: dict) -> dict:
    for _ in range(3):
        if not isinstance(data, dict):
            break
        for v in data.values():
            if isinstance(v, dict) and len(v) >= 3:
                data = v
                break
        else:
            break
    return data


def _iter_response_dicts(data):
    """Yield dictionaries recursively, keeping the original response intact."""
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _iter_response_dicts(value)

    elif isinstance(data, list):
        for item in data:
            yield from _iter_response_dicts(item)


def _find_line_items(data):
    """Find line_items even when nested inside another response object."""
    for current_dict in _iter_response_dicts(data):
        for key, value in current_dict.items():
            if _loose(key) == "lineitems":
                if isinstance(value, list):
                    return value

                print(
                    f"[warning] line_items was found but is not a list: "
                    f"{type(value).__name__}"
                )
                return []

    return []




def _normalize_response(
    data: dict,
    positions: dict,
    estimate_confidence: bool = False
) -> dict:

    if not positions:
        return data

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected model response to be a dictionary, "
            f"received {type(data).__name__}"
        )

    response_dicts = list(_iter_response_dicts(data))
    normalized = {}

    for key, label, *_ in positions["header_fields"]:

        candidates = [_loose(key), _loose(label)]
        found = False
        raw_value = None

        # Search the original response and nested dictionaries.
        # Prefer the expected field key over its label.
        for candidate in candidates:
            for current_dict in response_dicts:
                matching_key = next(
                    (
                        existing_key
                        for existing_key in current_dict
                        if _loose(existing_key) == candidate
                    ),
                    None
                )

                if matching_key is not None:
                    raw_value = current_dict[matching_key]
                    found = True
                    break

            if found:
                break

        if not found:
            normalized[key] = {
                "value": None,
                "confidence": 0.0 if estimate_confidence else None
            }
            continue

        # Preserve the model's value and confidence when supplied.
        if isinstance(raw_value, dict) and "value" in raw_value:
            field_result = raw_value.copy()

            if "confidence" not in field_result:
                field_result["confidence"] = (
                    _estimate_confidence(field_result["value"])
                    if estimate_confidence
                    else None
                )

            normalized[key] = field_result

        else:
            normalized[key] = {
                "value": raw_value,
                "confidence": (
                    _estimate_confidence(raw_value)
                    if estimate_confidence
                    else None
                )
            }

    # Find line_items independently so nested header data
    # cannot cause line items to be lost.
    normalized["line_items"] = _find_line_items(data)

    return normalized


def _estimate_confidence(val) -> float | None:
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return 0.0
    if len(s) < 2:
        return 0.4
    return 0.85


def _is_missing_header_value(value) -> bool:
    """Return True when an extracted header value is effectively empty."""
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or text in {"null", "none", "n/a", "na", "-", "—"}

def _apply_mistral_summary_fallback(
    result: dict,
    pdf_path: str,
    positions: dict = None,
) -> dict:
    """Recover missing financial summary values using PDF label alignment."""
    if not isinstance(result, dict) or not positions or not pdf_path:
        return result

    summary_pattern = re.compile(
        r"\b(amount|tax|freight|subtotal|total|discount|"
        r"net amount|grand total|total due|sales tax|vat|gst|hst)\b",
        re.IGNORECASE,
    )

    # Use only financial fields defined in the active template.
    fields = [
        (key, str(label).strip())
        for key, label, *_ in positions.get("header_fields", [])
        if summary_pattern.search(f"{key} {label}")
    ]

    if not fields:
        return result

    number_pattern = re.compile(
        r"^(?:\.\d{1,2}|\d[\d,]*(?:\.\d{1,2})?)$"
    )

    def normalize(text):
        return re.findall(r"[a-z0-9]+", str(text).lower())

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for key, label in fields:
                current = result.get(key)
                current_value = (
                    current.get("value")
                    if isinstance(current, dict)
                    else current
                )

                # Never overwrite an existing model extraction.
                if not _is_missing_header_value(current_value):
                    continue

                label_tokens = normalize(label)
                if not label_tokens:
                    continue

                matches = []

                for page in pdf.pages:
                    words = page.extract_words(
                        x_tolerance=2,
                        y_tolerance=3,
                        keep_blank_chars=False,
                    )

                    # Group words into visual lines.
                    lines = []
                    for word in sorted(
                        words, key=lambda w: (w["top"], w["x0"])
                    ):
                        line = next(
                            (
                                item for item in lines
                                if abs(item["top"] - word["top"]) <= 3
                            ),
                            None,
                        )
                        if line is None:
                            line = {"top": word["top"], "words": []}
                            lines.append(line)
                        line["words"].append(word)

                    # Locate the exact template label.
                    for line in lines:
                        line_words = sorted(
                            line["words"], key=lambda w: w["x0"]
                        )
                        tokens = [
                            token
                            for word in line_words
                            for token in normalize(word["text"])
                        ]

                        for start in range(
                            len(tokens) - len(label_tokens) + 1
                        ):
                            if tokens[start:start + len(label_tokens)] != label_tokens:
                                continue

                            # Find the word positions occupied by the label.
                            label_words = line_words[start:start + len(label_tokens)]
                            label_right = max(w["x1"] for w in label_words)

                            # Find a numeric value below and aligned with this label.
                            candidates = [
                                word for word in words
                                if 4 <= word["top"] - line["top"] <= 35
                                and abs(word["x1"] - label_right) <= 18
                                and number_pattern.fullmatch(
                                    word["text"].strip().replace(",", "")
                                )
                            ]

                            if len(candidates) == 1:
                                matches.append(candidates[0]["text"].strip())

                # Only use a single unambiguous label/value match.
                if len(matches) != 1:
                    continue

                value = matches[0]

                if isinstance(current, dict):
                    updated = current.copy()
                    updated["value"] = value
                    updated["confidence"] = 1.0
                    result[key] = updated
                else:
                    result[key] = {
                        "value": value,
                        "confidence": 1.0,
                    }

                print(
                    f"  [mistral_azure] Recovered summary field "
                    f"'{label}': {value}"
                )

    except Exception as exc:
        print(f"  [mistral_azure] Summary fallback skipped: {exc}")

    return result


def _value_search_text(value: Any) -> str:
    """Normalize a value for duplicate-value matching in OCR/PDF text."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _duplicate_value_present_in_document(value: Any, document_text: str) -> bool:
    """
    Check whether a non-empty extracted value occurs at least twice in the
    document. Two occurrences are important: one can belong to the source
    field and the second can be the duplicated value for the missing field.
    """
    if _is_missing_header_value(value):
        return False

    value_text = _value_search_text(value)
    document_text = _value_search_text(document_text)
    if not value_text or not document_text:
        return False

    # Exact text match first (preserves formatted values such as 877-624-5757).
    if document_text.count(value_text) >= 2:
        return True

    # Phone numbers are often formatted differently by OCR, e.g.
    # 877-624-5757 vs 8776245757. Match their digits instead.
    digits = re.sub(r"\D", "", value_text)
    if len(digits) >= 7:
        digit_text = re.sub(r"\D", "", document_text)
        return digit_text.count(digits) >= 2

    return False


def _field_family(key: str, label: str) -> str:
    """
    Classify header fields so a duplicated value is only copied between
    semantically compatible fields. Line items are deliberately excluded.
    """
    text = f"{key} {label}".lower()
    if re.search(r"\b(phone|telephone|tel)\b", text):
        return "phone"
    if re.search(r"\bfax\b", text):
        return "fax"
    if re.search(r"\b(email|e-mail)\b", text):
        return "email"
    if re.search(r"\b(address|addr)\b", text):
        return "address"
    if re.search(r"\b(date|day)\b", text):
        return "date"
    if re.search(r"\b(id|identifier|number|no\.?|#)\b", text):
        return "id_or_number"
    if re.search(r"\bcurrency\b", text):
        return "currency"
    if re.search(r"\b(terms|payment)\b", text):
        return "terms"
    return "other"

def _apply_duplicate_header_fallback(
    result: dict,
    pdf_path: str,
    positions: dict = None,
) -> dict:
    """
    Correct TO / SHIP TO phone and fax fields using PDF text positions.
    Only runs when the active template contains all three target fields
    and the PDF has clearly identifiable TO and SHIP TO sections.
    """
    if not positions or not isinstance(result, dict):
        return result

    try:
        import pdfplumber

        # Identify the relevant fields from the active template.
        field_map = {}

        for key, label, *_ in positions.get("header_fields", []):
            name = f"{key} {label}".lower()

            # Normalize accented characters, e.g. Tél -> Tel
            name = unicodedata.normalize("NFKD", name)
            name = "".join(
                char for char in name
                if not unicodedata.combining(char)
            )

            name = re.sub(r"[^a-z0-9]+", " ", name).strip()
            tokens = set(name.split())

            if "fax" in tokens:
                field_map["fax"] = key
            elif (
                tokens.intersection({"phone", "telephone", "tel"})
                and "ship" in tokens
                and "to" in tokens
            ):
                field_map["ship_phone"] = key
            elif (
                tokens.intersection({"phone", "telephone", "tel"})
                and "to" in tokens
                and "ship" not in tokens
            ):
                field_map["to_phone"] = key

        # Do not make partial or uncertain corrections.
        if not all(k in field_map for k in ("to_phone", "fax", "ship_phone")):
            return result

        phone_pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:\+?\(?\d[\d().\-\s]{5,}\d)(?![A-Za-z0-9])"
        )

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(
                    x_tolerance=2,
                    y_tolerance=3,
                    keep_blank_chars=False
                )

                # Find the TO and SHIP TO headings on the same page.
                to_heading = next(
                    (
                        w for w in words
                        if w["text"].strip().upper() == "TO:"
                    ),
                    None
                )
                ship_heading = next(
                    (
                        w for w in words
                        if w["text"].strip().upper() == "SHIP"
                        and any(
                            x["text"].strip().upper() == "TO:"
                            and abs(x["top"] - w["top"]) <= 3
                            and x["x0"] > w["x0"]
                            for x in words
                        )
                    ),
                    None
                )

                if not to_heading or not ship_heading:
                    continue

                left_center = (to_heading["x0"] + to_heading["x1"]) / 2
                right_center = ship_heading["x0"]
                if right_center <= left_center:
                    continue

                column_split = (left_center + right_center) / 2
                heading_y = max(to_heading["bottom"], ship_heading["bottom"])

                # Group words into visual text lines.
                lines = []
                for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
                    line = next(
                        (
                            item for item in lines
                            if abs(item["top"] - word["top"]) <= 2.5
                        ),
                        None
                    )
                    if line is None:
                        line = {"top": word["top"], "words": []}
                        lines.append(line)
                    line["words"].append(word)

                # Stop at the next section/table heading.
                stop_words = {
                    "reference", "buyer", "incoterms", "terms",
                    "description", "part", "quantity", "qty"
                }
                stop_y = float("inf")

                for line in lines:
                    if line["top"] <= heading_y:
                        continue
                    text = " ".join(
                        w["text"] for w in sorted(
                            line["words"], key=lambda w: w["x0"]
                        )
                    ).lower()
                    tokens = set(re.findall(r"[a-z]+", text))
                    if len(tokens.intersection(stop_words)) >= 1:
                        stop_y = line["top"]
                        break

                found = {}

                for line in lines:
                    if not (heading_y < line["top"] < stop_y):
                        continue

                    left_words = sorted(
                        [w for w in line["words"] if w["x0"] < column_split],
                        key=lambda w: w["x0"]
                    )
                    right_words = sorted(
                        [w for w in line["words"] if w["x0"] >= column_split],
                        key=lambda w: w["x0"]
                    )

                    for side, side_words in (
                        ("left", left_words),
                        ("right", right_words)
                    ):
                        if not side_words:
                            continue

                        text = " ".join(w["text"] for w in side_words)
                        lower_text = text.lower()

                        if not re.search(r"\b(phone|telephone|tel|fax)\b", lower_text):
                            continue

                        matches = phone_pattern.findall(text)
                        if len(matches) != 1:
                            continue

                        value = re.sub(r"\s+", " ", matches[0]).strip()
                        if len(re.sub(r"\D", "", value)) < 7:
                            continue

                        if re.search(r"\bfax\b", lower_text) and side == "left":
                            found["fax"] = value
                        elif re.search(r"\b(phone|telephone|tel)\b", lower_text):
                            if side == "left":
                                found["to_phone"] = value
                            else:
                                found["ship_phone"] = value

                # Apply only if all three values were found unambiguously.
                if all(k in found for k in ("to_phone", "fax", "ship_phone")):
                    for field_name, value_name in (
                        ("to_phone", "to_phone"),
                        ("fax", "fax"),
                        ("ship_phone", "ship_phone")
                    ):
                        result[field_map[field_name]] = {
                            "value": found[value_name],
                            "confidence": 1.0
                        }

                    print(
                        "  [mistral_azure] Corrected contact fields from "
                        "TO / SHIP TO PDF layout."
                    )
                    return result

        return result

    except Exception as e:
        # Never fail the extraction because this optional correction failed.
        print(f"  [mistral_azure] Contact-field fallback skipped: {e}")
        return result


def _build_schema_instruction(positions: dict) -> str:
    if not positions:
        return ""

    field_list = "\n".join(
        f'  "{k}": {{"value": "<extracted text or null>", "confidence": <0.0-1.0>}}'
        for k, l, *_ in positions["header_fields"]
    )
    li_fields = "\n".join(
        f'    "{k}": "<{l}>"'
        for k, l, _ in positions["li_fields"]
        if k != "confidence"
    )

    return (
        "Return a JSON object with this EXACT structure:\n"
        "{\n"
        f"{field_list},\n"
        '  "line_items": [\n'
        "    {\n"
        f"{li_fields},\n"
        '      "confidence": <0.0-1.0 score for this row>\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "HEADER FIELD EXTRACTION RULES:\n"
        "1. Treat every requested field as an independent extraction target.\n"
        "2. Identify the field label and its corresponding value using the "
        "document's visual layout, section boundaries, and surrounding labels.\n"
        "3. Keep Vendor, Supplier, Ship To, Delivery To, Bill To, Invoice To, "
        "Sold To, Remit To, Buyer and Issuer information separate.\n"
        "4. Never assign a value to a field merely because it is nearby, "
        "similar, or belongs to the same company.\n"
        "5. Never copy a value from one section into another unless the "
        "document independently establishes that it belongs to both.\n"
        "6. Preserve complete company names, addresses and contact details.\n"
        "7. Associate phone numbers, fax numbers and email addresses with "
        "their own labelled sections and contacts.\n"
        "8. A blank field must remain null. Do not fill it using another "
        "field, previous documents, assumptions or external knowledge.\n"
        "9. Do not mistake a section heading or field label for its value.\n"
        "10. If a value cannot be reliably associated with a requested "
        "field, return null.\n\n"

        "PURCHASE ORDER IDENTIFICATION RULES:\n"
        "1. Extract the actual purchase order identifier, not the name "
        "of the company issuing the purchase order.\n"
        "2. Use labels such as Purchase Order No, PO Number, PO No, "
        "Order Number or equivalent document-specific labels as evidence.\n"
        "3. If a company name and PO identifier appear on the same line, "
        "extract only the identifier for the purchase order field.\n"
        "4. Do not include the issuer's name, document heading, page number, "
        "date or unrelated identifiers in the PO number.\n"
        "5. Do not assume that the first number in the document is the PO number.\n"
        "6. If the PO identifier cannot be identified reliably, return null.\n\n"

        "LINE ITEM RULES:\n"
        "1. Extract every actual product row separately.\n"
        "2. Keep item number, description, quantity, UOM, unit price, "
        "delivery date and amount in their respective fields.\n"
        "3. Do not merge adjacent columns or combine multiple rows.\n"
        "4. Include continuation lines in the appropriate description "
        "when they clearly belong to that item.\n"
        "5. Do not treat table headers, subtotals or footer text as product rows.\n\n"

        "CONFIDENCE RULES:\n"
        "1.0 = The value is explicitly present and correctly associated "
        "with the requested field.\n"
        "1.0 = The field is genuinely absent or blank in the PO; return null.\n"
        "0.7 = The value is present, but its association is somewhat ambiguous.\n"
        "0.4 = The value is uncertain or requires inference.\n"
        "0.0 = The field or its value is unreadable and cannot be reliably "
        "identified; return null.\n"
        "Never assign 0.0 confidence merely because a field is absent.\n"
        "Confidence must reflect extraction certainty, not field availability.\n\n"

        "FINAL VALIDATION:\n"
        "Before returning JSON, verify that every populated value belongs "
        "to its requested field and correct document section. "
        "Preserve all requested schema keys. Return JSON only."
    )

def _build_gpt_alignment_instruction(positions: dict = None) -> str:
    """
    Align other models with GPT extraction behaviour.
    Confidence scores are handled separately.
    """

    return """
GPT EXTRACTION BEHAVIOUR — FOLLOW THESE RULES:

1. FIELD MAPPING
   Use the actual document layout and template column headings.
   Do not assign a value to a field merely because its format
   looks appropriate for that field.

2. LINE ITEM EXTRACTION
   Preserve the content of each column independently.
   Do not move values between columns based on assumptions.

3. ITEM / DESCRIPTION
   If an item code and product description both appear inside
   the Item / Description column, preserve both in that field.
   Do not move the item code into Supplier Item No merely
   because it looks like a product identifier.

4. SUPPLIER ITEM NUMBER
   Extract a supplier item number only when it is actually
   associated with the Supplier Item No column.
   If that column is blank, return null.
   Never borrow a value from Item / Description.

5. EXTERNAL ITEM NUMBER
   Treat a value explicitly labelled 'Ext Item No' as a
   separate external item number.
   Do not confuse it with Supplier Item No.

6. CONTINUATION LINES
   Include product-description continuation lines in the
   description field when they belong to that product.
   Do not include unrelated labels or metadata.

7. MISSING VALUES
   Return null when a field is absent.
   Never invent, infer, or borrow values from nearby fields.

8. HEADER FIELDS
   Keep Supplier, Delivery Address, Invoice To, Contact,
   Phone, E-mail and other similarly named fields separate.
   Do not transfer values between these fields.

9. HEADER SUMMARY TABLE RULES
   Identify every summary field using its printed label.
   Associate each value with the correct label using horizontal
   alignment, column positions and surrounding labels.
   Preserve the original left-to-right column relationships.
   Do not shift values into neighbouring fields when a field is blank.
   Do not assign a value based only on its format or position
   relative to another value.
   When multiple values appear on the same row, identify their
   individual column boundaries before assigning them.
   If a value cannot be associated reliably with its label,
   return null rather than guessing.
   Never copy a value from another summary field, even when
   both fields contain identical amounts.

10. OUTPUT
    Follow the supplied schema and return every requested field.
    Do not add fields that are not part of the schema.

11. CONFIDENCE
    Confidence scoring must not change the extracted value.
    Extract the information first; calculate confidence separately.
"""


def _build_mistral_alignment_instruction(positions: dict = None) -> str:
    """
    Strict, template-driven line-item extraction instructions for Mistral.

    Handles irregular tables, split headers, wrapped rows, ambiguous
    alignment and extraction exceptions without inventing values.
    """
    if not positions:
        return ""

    header_map = "\n".join(
        f'- Output key "{key}" corresponds to header field "{label}".'
        for key, label, *_ in positions.get("header_fields", [])
    )


        # Identify financial summary fields from the active template.
    summary_fields = []

    summary_pattern = (
        r"\b(amount|tax|freight|subtotal|total|discount|"
        r"net amount|grand total|total due|sales tax|"
        r"vat|gst|hst)\b"
    )

    for key, label, *_ in positions.get("header_fields", []):
        field_text = f"{key} {label}"

        if re.search(summary_pattern, field_text, re.IGNORECASE):
            summary_fields.append(
                f'- JSON key "{key}" maps to the summary field "{label}".'
            )

    summary_map = "\n".join(summary_fields)

    # Build explicit ownership rules from the active template.
    contact_fields = []

    contact_pattern = r"\b(phone|telephone|tel|t\s*l|fax|email|e mail)\b"

    for key, label, *_ in positions.get("header_fields", []):
        key_text = re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()
        label_text = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()

        combined = f"{key_text} {label_text}"

        if not re.search(contact_pattern, combined):
            continue

        if re.search(r"\bfax\b", combined):
            contact_type = "fax"
        elif re.search(r"\b(email|e mail)\b", combined):
            contact_type = "email"
        else:
            contact_type = "phone"

        # Prefer the template label to identify the owning section.
        label_owner = re.sub(contact_pattern, "", label_text).strip()
        key_owner = re.sub(contact_pattern, "", key_text).strip()

        owner = label_owner or key_owner

        if owner:
            contact_fields.append(
                f'- Field "{key}" ({label}) belongs to the '
                f'"{owner.upper()}" section. Extract its {contact_type} '
                f'only from that section.'
            )
        else:
            contact_fields.append(
                f'- Field "{key}" ({label}) has no explicit section name. '
                f'Use document headings, visual grouping and nearby section '
                f'boundaries to identify its owner. Do not borrow values '
                f'from another section.'
            )

    contact_map = "\n".join(contact_fields)
    
    column_map = "\n".join(
        f'- Output key "{key}" corresponds ONLY to template column "{label}".'
        for key, label, *_ in positions.get("li_fields", [])
        if key != "confidence"
    )

    return """
        STRICT MISTRAL PURCHASE ORDER EXTRACTION INSTRUCTIONS

        You are extracting structured purchase order data from the ORIGINAL
        DOCUMENT. Accuracy and correct field association are more important
        than filling every field.

        A. TEMPLATE AND FIELD MAPPING

        1. Follow the supplied JSON schema and active template mappings exactly.
        2. Use the output keys and labels supplied below. Do not create,
        rename, remove or substitute fields.
        3. Map values using document headings, visual positions, cell boundaries,
        row structure and surrounding context together.
        4. Never assign a value to a field merely because its format looks suitable.
        5. Keep every header field independent. Never transfer values between
        Vendor, Supplier, Ship To, Bill To, Remit To, Buyer, Delivery To,
        Invoice To or other distinct sections.
        6. Never copy a value from one field into another unless the document
        independently shows that the value belongs to both.
        
    
        """ + "HEADER FIELD MAP:\n" + header_map + """

        FOOTER SUMMARY FIELD MAP:
        """ + (
            summary_map
            if summary_map
            else "No financial summary fields are defined in the active template."
        ) + """

        STRICT FOOTER SUMMARY EXTRACTION RULES:

        1. Extract every summary field listed in the FOOTER SUMMARY
           FIELD MAP independently.

        2. Inspect the entire original document, including the bottom
           of every page, for the corresponding summary values.

        3. Locate the actual printed label first. Then identify the
           value visually associated with that exact label.

        4. Use the label, horizontal alignment, vertical alignment,
           surrounding labels and table boundaries together.
           Never assign a value based on numerical appearance alone.

        5. Extract a value only when the document provides clear
           evidence that it belongs to the requested field.

        6. Keep Amount, Tax, Freight, Subtotal, Discount and Total
           as separate fields. Never interchange them.

        7. If two or more summary fields contain identical values,
           extract each independently only when the document
           explicitly supports each assignment.

        8. Preserve the original numeric value, decimal precision,
           negative sign and currency notation where applicable.

        9. Treat explicitly printed zero values, including 0, 0.00
           and .00, as valid extracted values. Never convert them
           to null or an empty string.

        10. Never calculate Amount, Tax, Freight, Subtotal,
            Discount or Total using other extracted values.

        11. Never calculate tax using a percentage or derive Total
            by adding Amount, Tax and Freight.

        12. Never copy Total into Amount, Amount into Total,
            or any other summary field into a neighbouring field.

        13. Never use a line-item unit price, quantity, extended
            amount or individual product value as a substitute
            for a document-level summary value.

        14. Never extract a summary value from an unrelated page,
            another purchase order, a previous extraction or
            external knowledge.

        15. If a field has multiple candidate values, use only
            the value clearly associated with its printed label.
            If the association remains ambiguous, return null.

        16. If the field label or its corresponding value is absent
            or unreadable, return null. Do not guess.

        17. Summary values must be assigned only to their
            corresponding header JSON keys. Never put them
            inside line_items.

        18. Before returning JSON, verify every summary field
            against the original document and its printed label.
                19. A summary label and its corresponding amount may appear
            on the same horizontal row. Match the amount immediately
            to the right of that exact label.

        20. Never associate an amount with a label from the row
            above or below it.

        21. Treat GST, HST, PST, QST, VAT and Freight as
            independent fields. Never shift amounts between them.

        22. If a label has no amount on its row, return null.
            Never borrow the next available numeric value.

        23. Verify every summary value against its original
            label and horizontal position before returning JSON.

        CONTACT FIELD OWNERSHIP RULES:
        """ + contact_map + """

STRICT CONTACT ASSOCIATION:

1. Identify the section heading before assigning a phone, fax or email.
2. Associate each contact value with its correct section.
3. Do not assign a contact value based only on its format or proximity.
4. Keep different contact sections independent.
5. A phone number belonging to one section must not be transferred
   to another section.
6. Preserve multiple contact values when explicitly shown.
7. If ownership cannot be established, return null.
8. A value may appear in multiple fields only when the document
   independently supports each assignment.

LINE-ITEM COLUMN MAP:
""" + column_map + """

B. STRICT TABLE RECONSTRUCTION

1. First identify the actual line-item table and its boundaries.
2. Identify the logical columns using the template labels and the
   document's visual layout.
3. A logical column heading may be split across multiple lines.
   Reconstruct the heading only when the fragments and their positions
   clearly establish that they form one heading.
4. Do not assume that each physical text line represents a separate row.
5. Do not assume that all values belonging to a row appear on one line.
6. Use horizontal alignment, column boundaries and vertical sequence
   to associate values with their correct columns.
7. Preserve the original left-to-right column relationships.
8. Do not shift values into adjacent columns simply to fill missing fields.
9. Do not merge multiple product rows into one object.
10. Do not split one product row into multiple objects merely because
    its description wraps onto additional lines.

C. ONE PRODUCT ROW = ONE JSON OBJECT

1. Every distinct purchased product or service must be represented
   by its own line_items object.
2. Keep the values of each product within that product's object.
3. A product identifier, description, quantity, unit, price or amount
   must remain associated with the row and column supported by the
   document.
4. If a description continues on the next physical line, append it to
   the same product's description only when the layout or context
   supports that relationship.
5. Do not attach shipping instructions, general notes, contact details,
   terms or unrelated text to a product description.
6. Do not combine values from different products into a space-separated
   string in a single field.
7. Do not combine several quantities, units or prices into one value.
8. Preserve the source spelling, identifiers, decimal precision and
   meaningful punctuation.

D. FIELD ASSOCIATION RULES

1. An item identifier appearing in one column must not automatically
   be copied into another identifier column.
2. A blank template column must remain null unless the document
   explicitly provides a corresponding value.
3. Do not confuse a product description with an item identifier.
4. Do not confuse supplier/vendor item numbers with other identifiers.
5. Do not confuse a unit price with an extended amount.
6. Do not infer a value from the order total, subtotal or another row.
7. Do not use information from another purchase order or external
   knowledge.

E. TABLE BOUNDARIES AND EXCLUSIONS

1. Exclude introductory notes that appear before the product rows.
2. Exclude shipping instructions, freight conditions, delivery notes,
   general terms and contact information.
3. Exclude subtotals, taxes, discounts, order totals and summary rows
   from line_items unless the active template explicitly defines
   them as product/service rows.
4. Exclude page numbers, repeated table headings and footer content.
5. If the table continues onto another page, continue extracting
   its actual product rows from that page.
6. Do not mistake a repeated header on a new page for a product row.

F. EXCEPTION HANDLING — DO NOT FORCE AN EXTRACTION

Handle uncertain layouts conservatively:

1. Split or incomplete headings:
   Reconstruct a heading only when its fragments can be reliably
   associated using their wording and visual positions. Otherwise,
   do not guess the column mapping.

2. Wrapped descriptions:
   Attach continuation text only when it clearly belongs to the
   preceding product. Otherwise, leave the uncertain text unassigned.

3. Missing or blank cells:
   Return null for the affected field. Never borrow a neighbouring
   value or shift other values to fill the gap.

4. Ambiguous row boundaries:
   Create separate objects only when there is evidence of distinct
   products. Do not invent extra rows to account for unassigned text.

5. Multiple tables:
   Extract product/service rows from all relevant purchase-order
   line-item tables. Do not combine unrelated tables or summaries.

6. Incomplete or unreadable rows:
   Preserve fields that are clearly supported by the document.
   Return null for individual fields that cannot be reliably read.
   Do not discard an otherwise identifiable product row just because
   one of its fields is missing.

7. No identifiable line-item table:
   Return an empty line_items array. Do not interpret arbitrary
   document text as purchased products.

8. Conflicting evidence:
   Prefer explicit column headings and clear cell alignment over
   assumptions based on value format. If the conflict cannot be
   resolved, return null for the uncertain field.

G. CONFIDENCE

1. Confidence must reflect both the readability of the value and
   the reliability of its association with the requested field.
2. Explicitly present and correctly aligned: high confidence.
3. Present but somewhat ambiguous: moderate confidence.
4. Inferred or weakly associated: low confidence.
5. Absent, blank or unreadable: return null and zero confidence
   where the schema permits it.
6. Never increase confidence merely because a value looks plausible.
7. Confidence scoring must never change, complete or invent a value.

H. FINAL SELF-CHECK BEFORE RETURNING JSON

Before producing the response, verify:

1. Each actual product row has its own object.
2. No two product rows have been merged.
3. No row has been split because of wrapped description text.
4. Each value belongs to its correct template column.
5. Multiple quantities, units or prices have not been concatenated.
6. No shipping notes, footer text or contact information appear
   as product data.
7. Missing and ambiguous values have not been fabricated.
8. Every requested template key is preserved.
9. The output conforms to the supplied JSON schema.

Return only valid JSON. Do not return markdown or additional fields.
"""


# ── Company detection ──────────────────────────────────────────────────────────

def detect_company_from_pdf(pdf_path: str, model_cfg: dict = None) -> str:
    """
    Extract the buying/issuing company name from a PO PDF using text parsing.
    Prioritizes company names near the top of the first page.
    """
    import pdfplumber
    import re

    skip_words = {
        "purchase order", "invoice", "quotation", "quote", "sales order",
        "date", "page", "cad funds", "canada", "phone", "fax", "attn",
        "purchase order number", "purchase order date page", "ship to",
        "vendor address", "vendor", "supplier", "bill to", "sold to",
        "reference", "terms", "ship via", "currency", "buyer name",
        "send to", "sold by",
    }

    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text() or ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    def is_company_line(line):
        lower = line.lower().strip().rstrip(":")

        if len(line) < 4:
            return False

        if lower in skip_words:
            return False

        # Reject lines containing only numbers and address punctuation.
        if re.fullmatch(r"[\d\s/.,#-]+", line):
            return False

        # Reject phone numbers, URLs, and common address labels.
        if re.match(
            r"^(phone|fax|attn|http|www|po box|p\.o\. box|"
            r"gst|hst|pst)\b",
            lower,
        ):
            return False

        # Reject street-address patterns such as "235 W 2950 S".
        if re.match(
            r"^\d+\s+[NSEW]\s+\d+\s+[NSEW]$",
            line,
            re.IGNORECASE,
        ):
            return False

        # Reject common street addresses.
        if re.search(
            r"\b(street|st\.|avenue|ave\.|road|rd\.|"
            r"boulevard|blvd\.|drive|dr\.|lane|ln\.|way)\b",
            lower,
        ):
            return False

        return True

    # Strategy 1: Look for a company name near the top of the document.
    for line in lines[:12]:
        if not is_company_line(line):
            continue

        # Prefer names containing clear company identifiers.
        if re.search(
            r"\b(inc\.?|ltd\.?|llc|corp\.?|corporation|"
            r"limited|company|associates)\b",
            line,
            re.IGNORECASE,
        ):
            cleaned = re.sub(
                r"\s+(purchase\s+order|invoice|quotation|order|po|page|date).*$",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()

            if cleaned:
                return cleaned

    # Strategy 2: Check the first few meaningful lines for a company name.
    for line in lines[:6]:
        if is_company_line(line):
            # Avoid returning address fragments or document headings.
            if re.search(r"[A-Za-z]", line) and not re.search(
                r"\b(inc\.?|ltd\.?|llc|corp\.?|corporation|"
                r"limited|company|associates)\b",
                line,
                re.IGNORECASE,
            ):
                if len(line.split()) >= 2 and not re.search(
                    r"\d{3,}", line
                ):
                    return line

    # Strategy 3: Find the Ship To section as a fallback.
    for i, line in enumerate(lines):
        if re.search(r"\bship\s*to\b", line, re.IGNORECASE):
            for candidate in lines[i + 1:i + 5]:
                if is_company_line(candidate):
                    if re.search(r"[A-Za-z]", candidate):
                        return candidate

    # Strategy 4: Return the first meaningful text line as a last resort.
    for line in lines:
        if is_company_line(line) and re.search(r"[A-Za-z]", line):
            return line

    return ""


def download_from_sharepoint(url: str, output_dir: str = None) -> str:
    import tempfile, re, time, urllib.parse
    import requests as req

    download_url = url + ("&download=1" if "?" in url else "?download=1")

    # Ensure output dir exists
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_dir = output_dir or tempfile.gettempdir()

    print(f"  Downloading from SharePoint: {url[:50]}...")
    r = req.get(download_url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download from SharePoint [{r.status_code}]")

    content_type = r.headers.get("Content-Type", "")
    print(f"  [sharepoint] Content-Type: {content_type!r}")

    # ── Step 1: Try Content-Disposition header ────────────────────────────────
    cd    = r.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.IGNORECASE)
    filename = match.group(1).strip() if match else ""

    # ── Step 2: Try extracting filename from URL path ─────────────────────────
    if not filename or filename in ("download", ""):
        parsed     = urllib.parse.urlparse(url)
        path_parts = parsed.path.split("/")
        for part in reversed(path_parts):
            decoded = urllib.parse.unquote(part)
            if decoded.endswith((".xlsx", ".pdf", ".xlsm")):
                filename = decoded
                break

    # ── Step 3: Final fallback — use content type ─────────────────────────────
    if not filename or filename in ("download", ""):
        if "spreadsheet" in content_type or "excel" in content_type:
            ext = ".xlsx"
        elif "pdf" in content_type:
            ext = ".pdf"
        else:
            ext = ".xlsx" if ":x:" in url else ".pdf"
        filename = f"sp_download_{int(time.time())}{ext}"

    # ── Step 4: Force correct extension based on Content-Type ─────────────────
    if "spreadsheet" in content_type or "excel" in content_type:
        if not filename.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            filename = Path(filename).stem + ".xlsx"

    # ── Step 5: Force .xlsx if saving to templates folder ────────────────────
    if output_dir and "template" in output_dir.lower():
        if not filename.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            filename = Path(filename).stem + ".xlsx"
    # ── Validate content is actually a file, not HTML ─────────────────────────
    if r.content[:5] in (b'<!DOC', b'<html', b'<HTML'):
        raise RuntimeError(
            "SharePoint returned a login/redirect page instead of the file. "
            "Open the sharing link in a private browser window — if it asks you "
            "to sign in, go to SharePoint → right-click file → Share → "
            "change to 'Anyone with the link can view' → copy the new link."
        )

    if filename.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        if r.content[:2] != b'PK':
            raise RuntimeError(
                f"Downloaded content is not a valid Excel file. "
                f"First bytes: {r.content[:20]!r}. "
                "The SharePoint link may require authentication."
            )

    save_path = str(Path(out_dir) / filename)
    with open(save_path, "wb") as f:
        f.write(r.content)

    print(f"  [sharepoint] Saved as: {save_path}")
    return save_path

# ── Mistral Azure — Document AI with chat completions fallback ─────────────────

def _extract_mistral_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Mistral on Azure — tries Document AI annotation first, falls back to chat completions."""
    pdf_b64 = base64.standard_b64encode(Path(pdf_path).read_bytes()).decode("utf-8")
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}",
    }

    # ── Attempt 1: Document AI annotation API ─────────────────────────────────
    schema = _build_mistral_schema(positions)
    print("\n========== MISTRAL SCHEMA BEING SENT ==========")
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    print("===============================================\n")

    schema_instruction = _build_schema_instruction(positions)
    alignment_instruction = _build_mistral_alignment_instruction(positions)
    pdf_text = _extract_pdf_text(pdf_path)

    text_evidence = (
        "\n\nPDF TEXT EVIDENCE:\n"
        "Use this text as supporting evidence alongside "
        "the original document layout. Do not rely on text "
        "order alone for field association.\n\n"
        f"{pdf_text}"
    )

    annotation_prompt = (
        f"{prompt}\n\n"
        f"{schema_instruction}\n\n"
        f"{alignment_instruction}\n\n"
        f"{text_evidence}"
    )

    payload = {
        "model": model_cfg["model_name"],
        "document": {
            "type":         "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        "document_annotation_format": schema,
        "document_annotation_prompt": annotation_prompt,   # may be rejected on older Azure deployments
        "include_image_base64":       False,
    }

    ocr_text = None

    def call_annotation():
        resp = requests.post(
            model_cfg["endpoint"].rstrip("/"),
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 300),
        )
        return resp.text, resp.status_code

    try:
        raw, status = call_annotation()

        if status == 422 and "extra_forbidden" in raw:
            # Older Azure deployment — strip the prompt field and retry
            print("  [mistral_azure] document_annotation_prompt rejected — retrying without it")
            payload.pop("document_annotation_prompt")
            raw, status = call_annotation()

        if status != 200:
            raise RuntimeError(f"API call failed [{status}]: {raw}")

        data = json.loads(raw)
        ann  = data.get("document_annotation")
        print("\n========== MISTRAL RAW ANNOTATION ==========")
        print(json.dumps(ann, indent=2) if ann else "NO ANNOTATION")
        print("============================================\n")

        if ann:
            result = json.loads(ann) if isinstance(ann, str) else ann
            result = _apply_layout_based_header_fallback(
                result,
                pdf_path,
                positions
            )

            # Check whether Mistral merged multiple product rows into one.
            line_items = result.get("line_items", [])
            merged_rows_detected = False

            if isinstance(line_items, list) and len(line_items) == 1 and positions:
                item = line_items[0]

                if isinstance(item, dict):
                    for key, label, *_ in positions.get("li_fields", []):
                        field_label = f"{key} {label}".lower()

                        # Ignore combined Unit Price / Extended Price columns.
                        # These legitimately contain two numeric values.
                        if "unit price" in field_label and "extended price" in field_label:
                            continue

                        # Check quantity and other price/amount columns.
                        if not re.search(
                            r"\b(qty|quantity|price|amount|cost)\b",
                            field_label
                        ):
                            continue

                        value = item.get(key)

                        if value is None:
                            continue

                        if isinstance(value, dict):
                            value = value.get("value")

                        if not isinstance(value, str):
                            continue

                        numbers = re.findall(
                            r"(?<![A-Za-z])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?",
                            value
                        )

                        if len(numbers) > 1:
                            merged_rows_detected = True
                            print(
                                f"  [mistral_azure] Possible merged rows detected "
                                f"in '{label}': {value}"
                            )
                            break

            # Also detect suspiciously long product descriptions.
            # These may indicate that shipping instructions or footer text
            # have been incorrectly assigned to a product row.

            if isinstance(line_items, list) and positions:
                description_keys = [
                    key
                    for key, label, *_ in positions.get("li_fields", [])
                    if re.search(r"\b(description|item\s+desc)\b", f"{key} {label}", re.I)
                ]

                for item in line_items:
                    if not isinstance(item, dict):
                        continue

                    for key in description_keys:
                        value = item.get(key)

                        if isinstance(value, dict):
                            value = value.get("value")

                        if isinstance(value, str) and len(value.strip()) > 250:
                            merged_rows_detected = True
                            print(
                                f"  [mistral_azure] Suspicious description detected "
                                f"in '{key}' ({len(value)} characters)."
                            )
                            break

                    if merged_rows_detected:
                        break

            if not merged_rows_detected:
                result = _apply_mistral_summary_fallback(
                    result,
                    pdf_path,
                    positions
                )

                return _apply_duplicate_header_fallback(
                    result,
                    pdf_path,
                    positions
                )
            raise RuntimeError(
                "Mistral annotation rejected because potentially merged "
                "line items or suspicious descriptions were detected. "
                "No chat-completions fallback is configured."
            )

        raise RuntimeError(
            "Mistral returned no document_annotation. "
            "No chat-completions fallback is configured."
        )

    except Exception:
        raise

    # # ── Attempt 2: Chat completions fallback ──────────────────────────────────
    # if not ocr_text:
    #     ocr_text = _extract_pdf_text(pdf_path)

    # schema_instruction = _build_schema_instruction(positions)
    # alignment_instruction = _build_mistral_alignment_instruction(positions)
    # chat_payload = {
    #     "model": model_cfg["model_name"],
    #     "messages": [
    #         {
    #             "role":    "system",
    #             "content": "You are a purchase order data extraction assistant. Return valid JSON only.",
    #         },
    #         {
    #             "role":    "user",
    #             "content": (
    #                 f"{prompt}\n\n"
    #                 f"{schema_instruction}\n\n"
    #                 f"{alignment_instruction}\n\n"
    #                 "Return ONLY valid JSON — no explanation, no markdown.\n\n"
    #                 f"PURCHASE ORDER TEXT:\n{ocr_text}"
    #             ),
    #         },
    #     ],
    #     "response_format": {"type": "json_object"},
    #     "temperature":     0,
    # }

    # # Derive chat endpoint from OCR endpoint
    # chat_endpoint = model_cfg["endpoint"].rstrip("/")
    # if "/ocr" in chat_endpoint:
    #     chat_endpoint = chat_endpoint.replace("/ocr", "/chat/completions")
    # elif not chat_endpoint.endswith("/chat/completions"):
    #     chat_endpoint = chat_endpoint + "/chat/completions"

    # def call_chat():
    #     resp = requests.post(
    #         chat_endpoint,
    #         headers=headers,
    #         json=chat_payload,
    #         timeout=model_cfg.get("timeout", 120),
    #     )
    #     return resp.text, resp.status_code

    # raw, status = _retry(call_chat, model_cfg.get("retries", 3))

    # # Check whether the API returned an empty response
    # if not raw or not raw.strip():
    #     raise RuntimeError(
    #         f"Azure API returned an empty response (HTTP {status}). "
    #         "Check the model deployment, endpoint, and API response."
    #     )

    # # Parse the API response and expose the actual error if it isn't JSON
    # try:
    #     data = json.loads(raw)
    # except json.JSONDecodeError as exc:
    #     raise RuntimeError(
    #         f"Azure API returned invalid JSON (HTTP {status}). "
    #         f"Response preview: {raw[:500]!r}"
    #     ) from exc

    # # Validate the expected chat-completions response structure
    # try:
    #     content = data["choices"][0]["message"]["content"]
    # except (KeyError, IndexError, TypeError) as exc:
    #     raise RuntimeError(
    #         "Unexpected Azure API response structure. "
    #         f"Response preview: {raw[:500]!r}"
    #     ) from exc

    # if not isinstance(content, str) or not content.strip():
    #     raise RuntimeError(
    #         "Azure model returned empty message content. "
    #         f"Response preview: {raw[:500]!r}"
    #     )

    # result = _parse_json(content)
    # result = _normalize_response(result, positions, estimate_confidence=False)
    # return _apply_duplicate_header_fallback(result, pdf_path, positions)


def _build_mistral_schema(positions: dict) -> dict:
    """Build JSON schema for Mistral Azure annotation API."""
    if not positions:
        return {}
    print("\n========== MISTRAL SCHEMA DEBUG ==========")
    print("HEADER FIELDS:")
    for field in positions["header_fields"]:
        print(field)

    print("\nLINE ITEM FIELDS:")
    for field in positions["li_fields"]:
        print(field)

    print("==========================================\n")

    props = {}
    
    for key, label, *_ in positions["header_fields"]:
        field_name = re.sub(
            r"[^a-z0-9]+", " ", f"{key} {label}".lower()
        ).strip()

        
        field_description = (
            f"{label}. Locate the corresponding field in the original document "
            "and extract only the value that belongs to it. First identify the "
            "field label or column heading, then establish the value's association "
            "using the document layout. Consider horizontal and vertical alignment, "
            "column boundaries, row structure, section headings, nearby labels "
            "and the relationship between headings and values. "
            "For tabular or summary sections, match each value to its own heading "
            "or column rather than selecting the nearest number or text. "
            "Keep adjacent and similarly formatted fields independent, even when "
            "their values are identical. Never copy, calculate, infer or substitute "
            "a value from another field. Preserve explicitly printed zero values. "
            "Return null only when the value is absent, unreadable or cannot be "
            "reliably associated with this field."
        )


        # Dynamically identify contact fields from the active template.
        contact_pattern = r"\b(phone|telephone|tel|t\s*l|fax|email)\b"

        if re.search(r"\battention\b", field_name):
            field_description = (
                f"{label}. Extract the complete text printed inside the "
                "ATTENTION section or boxed ATTENTION area of the original "
                "purchase order. Inspect the entire box, including wrapped "
                "lines and text below the ATTENTION heading. "
                "Do not confuse the ATTENTION heading with its actual content. "
                "Preserve the complete wording, email addresses, punctuation "
                "and original text. Do not summarize or shorten the content. "
                "If the box contains readable instructions, extract them "
                "as the field value. Return null only if the section is "
                "genuinely empty or unreadable."
            )

        elif re.search(contact_pattern, field_name):
            if re.search(r"\bfax\b", field_name):
                contact_type = "fax"
                remove_pattern = r"\bfax\b"
            elif re.search(r"\bemail\b", field_name):
                contact_type = "email"
                remove_pattern = r"\bemail\b"
            else:
                contact_type = "phone"
                remove_pattern = r"\b(phone|telephone|tel|t\s*l)\b"

            key_text = re.sub(
                r"[^a-z0-9]+", " ", str(key).lower()
            ).strip()

            label_text = re.sub(
                r"[^a-z0-9]+", " ", str(label).lower()
            ).strip()

            label_owner = re.sub(
                remove_pattern, "", label_text
            ).strip()

            key_owner = re.sub(
                remove_pattern, "", key_text
            ).strip()

            owner = label_owner or key_owner

            if owner:
                field_description = (
                    f"{label}. Extract ONLY the {contact_type} "
                    f"belonging to the '{owner.upper()}' section. "
                    f"Look for the {contact_type} label within that "
                    "section, including labels such as Phone, "
                    "Telephone or Tel where applicable. "
                    "Never use a contact value from another section. "
                    "Return null if no matching value exists in "
                    "the specified section."
                )
            else:
                field_description = (
                    f"{label}. Extract ONLY the value explicitly "
                    f"labelled as {contact_type}. Use the document's "
                    "section headings and visual grouping to establish "
                    "ownership. Never borrow a value from another "
                    "contact field or section. Return null if "
                    "ownership cannot be established."
                )

        elif re.search(r"\bship\s*via\b", field_name):
            field_description = (
                f"{label}. Extract the shipping method or carrier "
                "explicitly associated with this Ship Via field. "
                "Locate the exact 'SHIP VIA' or equivalent bilingual "
                "field heading in the original document. "
                "Inspect the cell or section directly associated with "
                "that heading, including the value printed below it. "
                "The value may be in a separate row beneath the heading, "
                "rather than on the same horizontal line. "
                "Keep Ship Via separate from Ship To, F.O.B. and Terms. "
                "Do not extract the value from the adjacent Terms cell. "
                "Preserve the original wording and language. "
                "Return null only if the corresponding value is genuinely "
                "blank, unreadable or cannot be reliably associated."
            )

        elif "freight h s t" in field_name:
            field_description = (
                f"{label}. Extract ONLY the amount associated with "
                "Freight H.S.T. Do not use Net Amount or Total Due. "
                "Return null if no separate amount is shown."
            )


        props[key] = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
            },
            "description": field_description,
        }

    
    li_props = {}

    for key, label, _ in positions["li_fields"]:
        if key == "confidence":
            li_props["confidence"] = {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            }
            continue

        field_name = f"{key} {label}".lower()

        if "description" in field_name:
            field_description = (
                f"{label}. Extract the actual product name and description. "
                "Include meaningful description continuation lines belonging "
                "to this item. Do not include item numbers, supplier item "
                "numbers, labels such as 'Ext Item No', or unrelated metadata."
            )

        elif re.search(r"\b(our\s+ref|ref\s*#|reference)\b", label, re.I):
            field_description = (
                f"{label}. Extract the value from this exact reference column. "
                "If the same identifier also appears in another column, "
                "extract it here only when it is visibly present in this column. "
                "Do not infer or copy it from another field. "
                "Return null if this column is blank."
    )
        elif any(term in field_name for term in (
            "supplitemno", "supplieritemno", "vendoritemno"
        )):
            field_description = (
                f"{label}. Extract only the supplier/vendor item number "
                "when it is explicitly associated with this column. "
                "Do not copy the item number or infer a value from another "
                "column. Return null if this column is blank."
            )

        else:
            field_description = label

        li_props[key] = {
            "type": ["string", "null"],
            "description": field_description
        }

    if "confidence" not in li_props:
        li_props["confidence"] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        }

    props["line_items"] = {
        "type":  "array",
        "items": {"type": "object", "properties": li_props},
    }

    required = [f[0] for f in positions["header_fields"]] + ["line_items"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "purchase_order_template",
            "schema": {
                "type":       "object",
                "properties": props,
                "required":   required,
            },
        },
    }



def _apply_layout_based_header_fallback(
    result: dict,
    pdf_path: str,
    positions: dict = None,
) -> dict:
    """
    Recover missing header values using PDF visual layout.

    Safety rules:
    1. Never overwrite a field explicitly returned as null with high confidence.
    2. Never assign non-date text to date-related fields.
    3. Never use generic layout fallback for contact fields.
    4. Recover a value only when exactly one candidate is found.
    """

    if not isinstance(result, dict) or not positions or not pdf_path:
        return result

    header_fields = [
        (key, str(label).strip())
        for key, label, *_ in positions.get("header_fields", [])
        if key and str(label).strip()
    ]

    if not header_fields:
        return result

    try:
        import pdfplumber
        import unicodedata

        def normalize_text(text):
            text = unicodedata.normalize("NFKD", str(text))
            text = "".join(
                char for char in text
                if not unicodedata.combining(char)
            )
            return text.lower()

        def normalize_tokens(text):
            return re.findall(r"[a-z0-9]+", normalize_text(text))

        def label_variants(label):
            return [
                tokens
                for part in str(label).split("/")
                if (tokens := normalize_tokens(part))
            ]

        def is_contact_field(key, label):
            field_text = normalize_text(f"{key} {label}")
            field_text = re.sub(r"[^a-z0-9]+", " ", field_text).strip()

            return bool(
                re.search(
                    r"\b(phone|telephone|tel|fax|email)\b",
                    field_text
                )
                or re.search(r"\bt\s+l\b", field_text)
                or re.search(r"\be\s+mail\b", field_text)
            )

        def is_date_field(key, label):
            field_text = normalize_text(f"{key} {label}")
            field_text = re.sub(r"[^a-z0-9]+", " ", field_text)

            return bool(re.search(
                r"\b("
                r"expected arrival|expected date|arrival date|"
                r"delivery date|requested date|req date|"
                r"required date|ship date|shipping date|"
                r"due date|date expected"
                r")\b",
                field_text
            ))

        def contains_date(value):
            """
            Accept common numeric and written date formats.
            Reject arbitrary text such as 'Ship'.
            """
            value = str(value).strip()

            patterns = [
                # 2025-09-12, 2025/09/12
                r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",

                # 09/12/2025, 12-09-25
                r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",

                # Sep 12, 2025 / September 12 2025
                r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
                r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
                r"aug(?:ust)?|sep(?:t(?:ember)?)?|"
                r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
                r"\s+\d{1,2}(?:,)?\s+\d{2,4}\b",

                # 12 Sep 2025
                r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|"
                r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
                r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
                r"\s+\d{2,4}\b",
            ]

            return any(
                re.search(pattern, value, re.I)
                for pattern in patterns
            )

        def group_visual_lines(words):
            if not words:
                return []

            heights = [
                max(1, word["bottom"] - word["top"])
                for word in words
            ]
            tolerance = sorted(heights)[len(heights) // 2] * 0.5

            lines = []

            for word in sorted(
                words,
                key=lambda w: (w["top"], w["x0"])
            ):
                matching_line = next(
                    (
                        line for line in lines
                        if abs(line["top"] - word["top"]) <= tolerance
                    ),
                    None,
                )

                if matching_line is None:
                    lines.append({
                        "top": word["top"],
                        "words": [word],
                    })
                else:
                    matching_line["words"].append(word)
                    matching_line["top"] = sum(
                        item["top"] for item in matching_line["words"]
                    ) / len(matching_line["words"])

            for line in lines:
                line["words"].sort(key=lambda w: w["x0"])

            return sorted(lines, key=lambda line: line["top"])

        def find_label(line_words, variants):
            flattened = []
            word_map = []

            for word_index, word in enumerate(line_words):
                for token in normalize_tokens(word["text"]):
                    flattened.append(token)
                    word_map.append(word_index)

            for variant in variants:
                size = len(variant)

                if not size:
                    continue

                for start in range(len(flattened) - size + 1):
                    if flattened[start:start + size] == variant:
                        first_word = word_map[start]
                        last_word = word_map[start + size - 1]

                        return (
                            line_words[first_word]["x0"],
                            line_words[last_word]["x1"],
                        )

            return None

        recovered_candidates = {
            key: [] for key, _ in header_fields
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(
                    x_tolerance=2,
                    y_tolerance=3,
                )

                lines = group_visual_lines(words)
                page_labels = []

                # Locate template labels on this page.
                for line_index, line in enumerate(lines):
                    for key, label in header_fields:
                        match = find_label(
                            line["words"],
                            label_variants(label),
                        )

                        if match:
                            page_labels.append({
                                "key": key,
                                "line_index": line_index,
                                "top": line["top"],
                                "x0": match[0],
                                "x1": match[1],
                            })

                # Associate labels with the next visual row.
                for label_info in page_labels:
                    key = label_info["key"]
                    label_line_index = label_info["line_index"]

                    same_line_labels = [
                        item for item in page_labels
                        if item["line_index"] == label_line_index
                    ]

                    left_labels = [
                        item for item in same_line_labels
                        if item["x1"] <= label_info["x0"]
                    ]

                    right_labels = [
                        item for item in same_line_labels
                        if item["x0"] >= label_info["x1"]
                    ]

                    left_boundary = (
                        max(item["x1"] for item in left_labels)
                        if left_labels else 0
                    )

                    right_boundary = (
                        min(item["x0"] for item in right_labels)
                        if right_labels else page.width
                    )

                    for next_index in range(
                        label_line_index + 1,
                        len(lines)
                    ):
                        candidate_line = lines[next_index]

                        # Do not cross another recognized header row.
                        if any(
                            item["line_index"] == next_index
                            for item in page_labels
                        ):
                            break

                        candidate_words = [
                            word for word in candidate_line["words"]
                            if word["x0"] >= left_boundary
                            and word["x1"] <= right_boundary
                        ]

                        value = " ".join(
                            word["text"] for word in candidate_words
                        ).strip()

                        if value:
                            recovered_candidates[key].append(value)

                        break

        # Apply recovery only when safe.
        for key, label in header_fields:
            current = result.get(key)

            current_value = (
                current.get("value")
                if isinstance(current, dict)
                else current
            )

            # Do not overwrite an explicitly confirmed empty value.
            if isinstance(current, dict):
                confidence = current.get("confidence")

                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = None

                if (
                    "value" in current
                    and current_value is None
                    and confidence is not None
                    and confidence >= 0.95
                ):
                    print(
                        f"  [mistral_azure] Preserving confirmed empty "
                        f"field '{label}'"
                    )
                    continue

            # Do not use generic fallback for contact information.
            if is_contact_field(key, label):
                continue

            # Existing non-empty model values are preserved.
            if not _is_missing_header_value(current_value):
                continue

            candidates = list(dict.fromkeys(
                recovered_candidates.get(key, [])
            ))

            # Require exactly one candidate.
            if len(candidates) != 1:
                continue

            recovered_value = candidates[0]

            # Date fields must contain an actual date.
            if is_date_field(key, label) and not contains_date(
                recovered_value
            ):
                print(
                    f"  [mistral_azure] Rejected invalid date fallback "
                    f"for '{label}': {recovered_value}"
                )
                continue

            if isinstance(current, dict):
                updated = current.copy()
                updated["value"] = recovered_value

                # Fallback-derived values should not receive high confidence.
                updated["confidence"] = min(
                    float(updated.get("confidence") or 0.7),
                    0.7
                )

                result[key] = updated
            else:
                result[key] = {
                    "value": recovered_value,
                    "confidence": 0.7,
                }

            print(
                f"  [mistral_azure] Header fallback recovered "
                f"'{label}': {recovered_value}"
            )

    except Exception as exc:
        print(
            f"  [mistral_azure] Layout-based header fallback "
            f"skipped: {exc}"
        )

    return result

# ── OpenAI Azure — chat completions ───────────────────────────────────────────

def _extract_openai_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """OpenAI GPT-4o on Azure — extracts text then uses chat completions."""
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    headers = {
        "api-key":      model_cfg["api_key"],
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {
                "role":    "system",
                "content": "You are a purchase order data extraction assistant. Always return valid JSON only.",
            },
            {
                "role":    "user",
                "content": (
                    f"{prompt}\n\n{schema_instruction}\n\n"
                    "Return ONLY valid JSON — no explanation, no markdown.\n\n"
                    f"PURCHASE ORDER TEXT:\n{text}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature":     0,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    
    raw, status_code = _retry(call, model_cfg.get("retries", 3))

    # DEBUG 1: Inspect API response
    data = json.loads(raw)

    print("\n========== MODEL RESPONSE DEBUG ==========")
    print("Response type:", type(data).__name__)
    print("Response keys:", list(data.keys()) if isinstance(data, dict) else "Not a dictionary")

    if isinstance(data, dict):
        choices = data.get("choices", [])
        print("Choices count:", len(choices))

        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
        else:
            content = None
    else:
        content = None

    print("Content type:", type(content).__name__)
    print("Content length:", len(content) if isinstance(content, str) else "Not a string")
    print("Raw content preview:", repr(content)[:2000])

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty or unexpected content")

    # DEBUG 2: Inspect parsed response
    result = _parse_json(content)
    print("\n========== GPT DEBUG: PARSED JSON ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nGPT line-item count:",
        len(result.get("line_items", []))
        if isinstance(result.get("line_items"), list)
        else "Invalid line_items structure")

    for index, item in enumerate(result.get("line_items", []), start=1):
        print(f"\nGPT Line Item {index}:")
        print(json.dumps(item, indent=2, ensure_ascii=False))

    print("========== END PARSED JSON ==========\n")

    print("Parsed JSON keys:", list(result.keys()))
    print("Raw line_items count:",
        len(result.get("line_items", []))
        if isinstance(result.get("line_items"), list)
        else "Not a list")

    # DEBUG 3: Inspect normalized response
    normalized = _normalize_response(
        result, positions, estimate_confidence=True
    )

    print("Normalized line_items count:",
        len(normalized.get("line_items", [])))

    print("========== END DEBUG ==========\n")

    return _apply_duplicate_header_fallback(
        normalized, pdf_path, positions
    )



# ── Phi Azure — Azure AI chat completions ─────────────────────────────────────

def _extract_phi_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Microsoft Phi on Azure AI — uses Azure AI inference chat completions."""
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    headers = {
        "Authorization": f"Bearer {model_cfg['api_key']}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model_cfg["model_name"],
        "messages": [
            {
                "role":    "system",
                "content": "You are a purchase order data extraction assistant. Always return valid JSON only.",
            },
            {
                "role":    "user",
                "content": (
                    f"{prompt}\n\n{schema_instruction}\n\n"
                    "Return ONLY valid JSON — no explanation, no markdown.\n\n"
                    f"PURCHASE ORDER TEXT:\n{text}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature":     0,
        "max_tokens":      4096,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, _  = _retry(call, model_cfg.get("retries", 3))
    data    = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    result  = _parse_json(content)
    result = _normalize_response(result, positions, estimate_confidence=False)
    return _apply_duplicate_header_fallback(result, pdf_path, positions)


def _build_llama_alignment_instruction(positions: dict = None) -> str:
    """Create field-mapping guidance from the active template."""
    if not positions:
        return ""

    header_map = "\n".join(
        f'- JSON key "{key}": value associated with header label "{label}".'
        for key, label, *_ in positions.get("header_fields", [])
    )
    column_map = "\n".join(
        f'- JSON key "{key}": value from table column "{label}" only.'
        for key, label, *_ in positions.get("li_fields", [])
        if key != "confidence"
    )

    return (
        "TEMPLATE-BASED FIELD ASSIGNMENT:\n"
        "Use document labels, table headings, cell positions, and row "
        "structure to assign values. Do not classify a value by appearance "
        "alone. Preserve table columns independently; never move a value "
        "to another column because it looks suitable there. Blank cells "
        "must be null. Keep continuation text with a row only when the "
        "document layout supports that association. Preserve identifiers "
        "that belong to a cell; do not automatically reclassify them. "
        "Associate header values with their own labels and sections. "
        "Do not copy or infer missing values. Confidence must not change "
        "the extracted value.\n\n"
        "HEADER FIELD MAP:\n" + header_map +
        "\n\nLINE-ITEM COLUMN MAP:\n" + column_map
    )



def _build_aligned_line_item_text(document_text: str, positions: dict = None) -> str:
    """
    Preserve the PDF's original table spacing and expose heading offsets.
    Avoids guessing cell boundaries from text alignment.
    """
    if not document_text or not positions:
        return ""

    fields = [
        (key, str(label).strip())
        for key, label, *_ in positions.get("li_fields", [])
        if key != "confidence" and str(label).strip()
    ]
    if len(fields) < 2:
        return ""

    lines = document_text.splitlines()
    header_idx = None
    heading_positions = []

    for idx, line in enumerate(lines):
        found = []
        for key, label in fields:
            match = re.search(re.escape(label), line, flags=re.IGNORECASE)
            if not match:
                found = []
                break
            found.append((key, label, match.start()))
        if len(found) == len(fields):
            found.sort(key=lambda item: item[2])
            if all(found[i][2] < found[i + 1][2] for i in range(len(found) - 1)):
                header_idx = idx
                heading_positions = found
                break

    if header_idx is None:
        return ""

    table_lines = []
    blank_count = 0
    for line in lines[header_idx:]:
        if not line.strip():
            blank_count += 1
            if blank_count >= 3 and table_lines:
                break
            continue
        blank_count = 0
        if re.search(
            r'\b(order\s+total|subtotal|terms\s+and\s+conditions|invoice\s+to)\b',
            line,
            re.IGNORECASE,
        ):
            break
        table_lines.append(line)

    if len(table_lines) < 2:
        return ""

    offsets = ", ".join(
        f'{label} (JSON key "{key}") starts at character {column}'
        for key, label, column in heading_positions
    )
    raw_rows = "\n".join(
        f"row {i}: {line!r}" for i, line in enumerate(table_lines)
    )
    header_labels = [
        str(label).strip()
        for _, label, *_ in positions.get("header_fields", [])
        if str(label).strip()
    ]

    return (
        "PDF TABLE LAYOUT EVIDENCE (spaces are preserved exactly; "
        "do not treat this as a pre-parsed table):\n"
        f"Detected heading offsets: {offsets}\n"
        f"Original table lines:\n{raw_rows}\n\n"
        "Header-field labels from the active template: "
        + ", ".join(header_labels)
        + ". If a continuation line starts with one of these header labels, "
        "do not reassign its labelled value to a line-item column. "
        "Assign line-item values using the original table heading positions "
        "and the preserved spaces. A blank cell remains null."
    )


def _apply_llama_template_label_continuation(
    result: dict,
    pdf_path: str,
    positions: dict = None,
) -> dict:
    """
    Correct a narrow class of Llama line-item column shifts.

    A template header field can also appear as a labelled continuation line
    inside the PDF line-item table (for example, "Ext Item No ..."). When
    that continuation line is horizontally aligned with the item-description
    column, move its value from a conflicting line-item field into the
    description. The rule is driven by the active template and PDF geometry;
    it contains no supplier- or document-specific labels.
    """
    if not isinstance(result, dict) or not positions or not pdf_path:
        return result

    li_fields = positions.get("li_fields", [])
    header_fields = positions.get("header_fields", [])

    def norm(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    def field_value(field):
        return field.get("value") if isinstance(field, dict) else field

    def set_field(item, key, value):
        current = item.get(key)
        if isinstance(current, dict) and "value" in current:
            updated = current.copy()
            updated["value"] = value
            item[key] = updated
        else:
            item[key] = value

    desc = next(
        (
            (key, str(label))
            for key, label, *_ in li_fields
            if re.search(r"\b(description|item\s+desc)\b", norm(key + " " + str(label)))
        ),
        None,
    )
    if not desc:
        return result

    desc_key, desc_label = desc
    # Only consider a supplier-item-number field as a possible mistaken target.
    supplier_item = next(
        (
            (key, str(label))
            for key, label, *_ in li_fields
            if key != desc_key
            and re.search(
                r"\b(supplier|suppl|vendor)\b", norm(key + " " + str(label))
            )
            and re.search(r"\b(item|part|product)\b", norm(key + " " + str(label)))
        ),
        None,
    )
    if not supplier_item:
        return result

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            page_lines = []
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=2, y_tolerance=3)
                grouped = []
                for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
                    if not grouped or abs(grouped[-1][0]["top"] - word["top"]) > 2.5:
                        grouped.append([word])
                    else:
                        grouped[-1].append(word)
                page_lines.extend(
                    {
                        "top": min(w["top"] for w in group),
                        "words": sorted(group, key=lambda w: w["x0"]),
                        "text": " ".join(w["text"] for w in sorted(group, key=lambda w: w["x0"])),
                    }
                    for group in grouped
                )
    except Exception as exc:
        print(f"[Llama layout correction] PDF geometry unavailable: {exc}")
        return result

    # Find the table heading by matching all active template line-item labels
    # on one PDF text line. This avoids relying on fixed page coordinates.
    labels = [
        (key, str(label))
        for key, label, *_ in li_fields
        if key != "confidence" and str(label).strip()
    ]
    header_line = None
    heading_x = {}
    for line in page_lines:
        line_norm = norm(line["text"])
        found = {}
        for key, label in labels:
            label_norm = norm(label)
            if label_norm and label_norm in line_norm:
                # Use the first matching word position as the column anchor.
                label_tokens = label_norm.split()
                normalized_words = []
                for word in line["words"]:
                    word_tokens = norm(word["text"]).split()
                    normalized_words.extend(
                        (token, word) for token in word_tokens
                    )
                words_norm = [entry[0] for entry in normalized_words]
                for i in range(len(words_norm) - len(label_tokens) + 1):
                    if words_norm[i:i + len(label_tokens)] == label_tokens:
                        found[key] = normalized_words[i][1]["x0"]
                        break
        if len(found) == len(labels):
            header_line = line
            heading_x = found
            break

    if header_line is None or desc_key not in heading_x:
        return result

    desc_x = heading_x[desc_key]
    # Allow modest extraction/layout variation, but require the labelled
    # continuation to begin at the description column's horizontal anchor.
    header_label_candidates = [
        str(label).strip()
        for _, label, *_ in header_fields
        if str(label).strip()
    ]
    continuations = []
    for line in page_lines:
        if line["top"] <= header_line["top"]:
            continue
        first_x = line["words"][0]["x0"] if line["words"] else None
        if first_x is None or abs(first_x - desc_x) > 14:
            continue

        text = line["text"].strip()
        text_norm = norm(text)
        for header_label in sorted(header_label_candidates, key=len, reverse=True):
            label_norm = norm(header_label)
            if not label_norm:
                continue
            if text_norm.startswith(label_norm + " "):
                # Remove the label using token count, retaining the actual
                # source spelling and punctuation of the value.
                tokens = text.split()
                remainder = " ".join(tokens[len(label_norm.split()):]).strip(" :-–—")
                if remainder:
                    continuations.append(remainder)
                break

    if not continuations:
        return result

    supplier_key, _ = supplier_item
    for item in result.get("line_items", []):
        if not isinstance(item, dict):
            continue
        supplier_value = field_value(item.get(supplier_key))
        if supplier_value is None or not str(supplier_value).strip():
            continue

        supplier_norm = norm(supplier_value)
        for continuation in continuations:
            continuation_norm = norm(continuation)
            # Only move a value when the model's supplier-item field contains
            # the labelled continuation value (or vice versa).
            if (
                supplier_norm == continuation_norm
                or supplier_norm in continuation_norm
                or continuation_norm in supplier_norm
            ):
                current_desc = field_value(item.get(desc_key))
                current_desc_text = str(current_desc or "").strip()
                if continuation_norm not in norm(current_desc_text):
                    merged = " ".join(
                        part for part in (continuation, current_desc_text) if part
                    ).strip()
                    set_field(item, desc_key, merged)
                set_field(item, supplier_key, None)
                print(
                    "[Llama layout correction] Moved a template-labelled "
                    "continuation from the supplier-item field to the "
                    "description field."
                )
                break

    return result


def _correct_llama_received_bo_alignment(
    result: dict,
    pdf_path: str,
    positions: dict = None,
) -> dict:
    """
    Conservative Llama-only correction for RECEIVED / B.O. quantity shifts.

    A row is corrected only when:
      1. Its item number matches between PDF and extracted JSON.
      2. PDF RECEIVED column has no quantity.
      3. PDF B.O. column has a quantity.
      4. Llama put that exact B.O. quantity in RECEIVED.
      5. Llama left B.O. empty.

    Ambiguous rows are never modified.
    """
    from decimal import Decimal, InvalidOperation

    if not isinstance(result, dict) or not isinstance(positions, dict):
        return result

    line_items = result.get("line_items")
    li_fields = positions.get("li_fields", [])

    if not isinstance(line_items, list) or not line_items:
        return result

    # Identify the required fields from the active Excel template.
    received_key = None
    bo_key = None
    item_number_key = None

    for key, label, *_ in li_fields:
        key_text = str(key).lower().replace("_", " ")
        label_text = str(label).lower()

        if "received" in key_text or "received" in label_text:
            received_key = key

        if (
            "b o" in key_text
            or "b o" in label_text.replace(".", "")
            or "souff" in key_text
            or "souff" in label_text
        ):
            bo_key = key

        if (
            key_text.strip() in {"no", "item no", "item number"}
            or label_text.strip().replace(".", "") in {
                "no", "item no", "item number"
            }
        ):
            item_number_key = key

    if not all([received_key, bo_key, item_number_key]):
        print(
            "[LLAMA RECEIVED/BO] Required template fields not found. "
            "No changes made."
        )
        return result

    def get_value(field):
        if isinstance(field, dict):
            return field.get("value")
        return field

    def normalize_item_code(value):
        if value is None:
            return None
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())

    def numeric_value(value):
        if value is None:
            return None

        text = str(value).strip().replace(",", "").replace("$", "")

        if not text:
            return None

        try:
            return Decimal(text)
        except (InvalidOperation, ValueError):
            return None

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            # Avoid applying assumptions to multi-page documents.
            if len(pdf.pages) != 1:
                print(
                    "[LLAMA RECEIVED/BO] Multi-page PDF. "
                    "No changes made."
                )
                return result

            page = pdf.pages[0]
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
            )

        # Locate the actual quantity-column headings.
        ordered_heading = next(
            (w for w in words if w["text"].upper() == "ORDERED"),
            None,
        )
        received_heading = next(
            (w for w in words if w["text"].upper() == "RECEIVED"),
            None,
        )
        bo_heading = next(
            (w for w in words if w["text"].upper() == "B.O."),
            None,
        )

        if not all([
            ordered_heading,
            received_heading,
            bo_heading,
        ]):
            print(
                "[LLAMA RECEIVED/BO] PDF headings not recognized. "
                "No changes made."
            )
            return result

        ordered_x = ordered_heading["x0"]
        received_x = received_heading["x0"]
        bo_x = bo_heading["x0"]

        if not ordered_x < received_x < bo_x:
            print(
                "[LLAMA RECEIVED/BO] Unexpected column order. "
                "No changes made."
            )
            return result

        header_y = max(
            ordered_heading["top"],
            received_heading["top"],
            bo_heading["top"],
        )

        ordered_received_boundary = (
            ordered_x + received_x
        ) / 2

        received_bo_boundary = (
            received_x + bo_x
        ) / 2

        # Identify PDF item rows using item codes in the first column.
        pdf_rows = {}

        for word in words:
            code = normalize_item_code(word["text"])

            if (
                word["top"] > header_y
                and word["x0"] < 70
                and code
                and re.fullmatch(r"(?=.*\d)[A-Z0-9]+", code)
            ):
                row_y = round(word["top"], 1)

                # Ignore duplicate item-code detections on the same row.
                pdf_rows[row_y] = {
                    "item_code": code,
                    "received": None,
                    "bo": None,
                }

        if not pdf_rows or len(pdf_rows) != len(line_items):
            print(
                "[LLAMA RECEIVED/BO] PDF/extracted row count mismatch:",
                len(pdf_rows),
                "PDF rows;",
                len(line_items),
                "extracted rows. No changes made."
            )
            return result

        # Read only numeric values within the two relevant PDF columns.
        for word in words:
            text = word["text"].strip().replace(",", "")

            if not re.fullmatch(r"\d+(?:\.\d+)?", text):
                continue

            x = word["x0"]
            y = word["top"]

            if not (
                received_bo_boundary - 60 <= x <= bo_x + 45
            ):
                continue

            closest_row_y = min(
                pdf_rows,
                key=lambda row_y: abs(row_y - y),
            )

            if abs(closest_row_y - y) > 3:
                continue

            if (
                ordered_received_boundary <= x
                < received_bo_boundary
            ):
                pdf_rows[closest_row_y]["received"] = text

            elif received_bo_boundary <= x <= bo_x + 45:
                pdf_rows[closest_row_y]["bo"] = text

        # Match extracted rows by item number, never by list position alone.
        extracted_by_code = {}

        for item in line_items:
            if not isinstance(item, dict):
                print(
                    "[LLAMA RECEIVED/BO] Invalid extracted row. "
                    "No changes made."
                )
                return result

            code = normalize_item_code(
                get_value(item.get(item_number_key))
            )

            if not code or code in extracted_by_code:
                print(
                    "[LLAMA RECEIVED/BO] Missing or duplicate extracted "
                    "item number. No changes made."
                )
                return result

            extracted_by_code[code] = item

        if set(extracted_by_code) != {
            row["item_code"] for row in pdf_rows.values()
        }:
            print(
                "[LLAMA RECEIVED/BO] PDF and extracted item numbers "
                "do not match exactly. No changes made."
            )
            return result

        corrected_count = 0

        # Correct only rows with direct, matching evidence.
        for pdf_row in pdf_rows.values():
            code = pdf_row["item_code"]
            item = extracted_by_code[code]

            pdf_received = numeric_value(pdf_row["received"])
            pdf_bo = numeric_value(pdf_row["bo"])

            model_received = numeric_value(
                get_value(item.get(received_key))
            )
            model_bo = numeric_value(
                get_value(item.get(bo_key))
            )

            # PDF must show Received blank and B.O. populated.
            if pdf_received is not None or pdf_bo is None:
                continue

            # Llama must have put the exact B.O. quantity in Received
            # and left B.O. empty. Otherwise do not interfere.
            if model_received != pdf_bo or model_bo is not None:
                continue

            # Preserve the existing field structure and confidence.
            for key, new_value in (
                (received_key, None),
                (bo_key, str(pdf_row["bo"])),
            ):
                existing = item.get(key)

                if isinstance(existing, dict):
                    existing["value"] = new_value
                    existing["confidence"] = 1.0
                else:
                    item[key] = {
                        "value": new_value,
                        "confidence": 1.0,
                    }

            corrected_count += 1

        print(
            "[LLAMA RECEIVED/BO] Corrected",
            corrected_count,
            "verified row(s). Other rows were left unchanged."
        )

    except Exception as exc:
        print(
            "[LLAMA RECEIVED/BO] Correction safely skipped:",
            str(exc)
        )

    return result


# ── Llama Azure — Azure AI chat completions ───────────────────────────────────

def _extract_llama_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    print("\n========== LLAMA HANDLER ENTERED ==========")

    text = _extract_pdf_text(pdf_path)

    # DEBUG: Check whether PDF text extraction is working
    print("\n========== PDF TEXT DEBUG ==========")
    print("PDF path:", pdf_path)
    print("Extracted text length:", len(text) if text else 0)
    print("Extracted text preview:")
    print(repr(text) if text else "EMPTY TEXT")
    print("========== END PDF TEXT DEBUG ==========\n")

    schema_instruction = _build_schema_instruction(positions)
    alignment_instruction = _build_llama_alignment_instruction(positions)
    aligned_table_text = _build_aligned_line_item_text(text, positions)
    gpt_alignment_instruction = _build_gpt_alignment_instruction(positions)

    li_list = ""
    if positions:
        li_list = "\n".join(
            f'- "{k}": {l}'
            for k, l, _ in positions["li_fields"]
            if k != "confidence"
        )
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key']}",
        "Content-Type":  "application/json",
    }

    full_prompt = (
        f"{prompt}\n\n"
        f"{schema_instruction}\n\n"
        f"{alignment_instruction}\n\n"
        f"{gpt_alignment_instruction}\n\n"
        f"{aligned_table_text}\n\n"
        
        "STRICT LINE ITEM COLUMN ALIGNMENT RULES:\n"
        "1. Read the table headings first and identify the exact "
        "column associated with each output field.\n"
        "2. Assign every row value to its original column using "
        "the horizontal position of the value relative to the "
        "column headings.\n"
        "3. Numeric values may be right-aligned. Do not assume "
        "that a value belongs to a column simply because its "
        "text starts below or near the heading's starting position.\n"
        "4. Preserve the complete left-to-right column relationship. "
        "Never shift numeric values into neighbouring columns.\n"
        "5. A small integer such as 1, 2, 5 or 10 can be a valid "
        "quantity. Never discard it because another numeric value "
        "looks more like a quantity.\n"
        "6. Keep Qty, Rate, Unit Price and Amount independent. "
        "Do not interchange quantity, unit price and extended amount.\n"
        "7. If the PDF shows Qty=5, Rate=143.00 and Amount=715.00, "
        "the output must preserve those exact assignments.\n"
        "8. Do not calculate missing values using multiplication "
        "or other arithmetic. Extract only values supported by "
        "the document.\n"
        "9. If a cell is genuinely blank, return null for that "
        "field without moving another cell's value into it.\n"
        "10. Before returning JSON, compare every extracted numeric "
        "value against its original table column.\n\n"
        "Extract every actual product row. Keep rows separate. "
        "Return null for genuinely blank cells.\n\n"
        "Return ONLY valid JSON — no explanation, no markdown.\n\n"
        f"PURCHASE ORDER TEXT:\n{text}"
    )

    payload = {
        "model":    model_cfg["model_name"],
        "messages": [
            {
                "role":    "system",
                "content": "You are a precise purchase order data extraction assistant. Always extract ALL line items — never skip any rows. Return valid JSON only.",
            },
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0,
        "max_tokens":  8192,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    
    raw, status = _retry(call, model_cfg.get("retries", 3))

    data = json.loads(raw)

    print("\n========== LLAMA/COHERE RESPONSE DEBUG ==========")
    print("Response keys:", list(data.keys()) if isinstance(data, dict) else type(data).__name__)

    choices = data.get("choices", []) if isinstance(data, dict) else []

    print("Choices count:", len(choices))

    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
    else:
        content = None

    print("Content type:", type(content).__name__)
    print("Content length:", len(content) if isinstance(content, str) else "Not a string")
    print("Content preview:", repr(content)[:2000])

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty or unexpected content")

    result = _parse_json(content)
    print("\n========== RAW MODEL RESPONSE ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("========== END RAW MODEL RESPONSE ==========\n")

    print("Parsed JSON keys:", list(result.keys()))
    print(
        "Raw line items:",
        len(result.get("line_items", []))
        if isinstance(result.get("line_items"), list)
        else "Not a list"
    )

    normalized = _normalize_response(
        result, positions, estimate_confidence=True
    )

    # Apply only to the Llama/Cohere chat-completions path. Mistral and GPT
    # extraction handlers are not changed.
    normalized = _apply_llama_template_label_continuation(
        normalized, pdf_path, positions
    )
    normalized = _correct_llama_received_bo_alignment(
        normalized, pdf_path, positions
    )

    print(
        "Normalized line items:",
        len(normalized.get("line_items", []))
    )

    print("========== END RESPONSE DEBUG ==========\n")

    return _apply_duplicate_header_fallback(
        normalized, pdf_path, positions
    )



# ── OpenAI Azure — Responses API (/v1/responses endpoint) ────────────────────

def _extract_openai_azure_responses(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """GPT-5.x on Azure using the new /v1/responses API (not chat completions)."""
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)
    gpt_description_instruction = """
    GPT-ONLY LINE ITEM DESCRIPTION RULES:

    - Product descriptions may span multiple physical lines.
    - Combine continuation lines belonging to the same product.
    - Before appending a continuation line, check whether its
      beginning repeats text already captured from the previous
      description line.
    - Do not include overlapping text twice.
    - If a previous line ends with a truncated word and the next
      line contains its complete form, retain the complete form once.
    - Preserve all unique product details, specifications,
      materials, dimensions and model numbers.
    - Do not remove text that provides additional information.
    - Apply these rules only to line-item descriptions.
    """

    gpt_line_item_confidence_instruction = """
    GPT-ONLY LINE ITEM CONFIDENCE RULES:

    Calculate confidence for each line-item row independently.

    1.0:
    - The product row is clearly identifiable.
    - Extracted values are explicitly supported by the PDF.
    - Values are correctly associated with their respective columns.
    - Description continuation lines are correctly associated with the product.

    0.7:
    - The product row is identifiable, but at least one important
    field has genuine ambiguity in its value or column association.

    0.4:
    - The row or one or more important values require inference.

    0.0:
    - The row cannot be reliably identified or its values are unreadable.

    IMPORTANT:
    - Do not lower confidence merely because a description spans
    multiple physical lines.
    - Do not lower confidence merely because PDF text extraction
    repeats overlapping description text, if the product and
    its actual details remain identifiable.
    - Do not lower confidence because a field is genuinely blank.
    - Do not assign 1.0 simply to increase the score.
    - Evaluate the actual evidence and row association.
    - Return one numeric confidence score for each line item.
    """


    
    print("\n========== GPT DEBUG: INPUT PDF TEXT ==========")
    print(text)
    print("========== END INPUT PDF TEXT ==========\n")

    headers = {
        "api-key":      model_cfg["api_key"],
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_cfg["model_name"],
        
        "input": (
            f"{prompt}\n\n{schema_instruction}\n\n"
            f"{gpt_description_instruction}\n\n"
            f"{gpt_line_item_confidence_instruction}\n\n"
            "Return ONLY valid JSON — no explanation, no markdown.\n\n"
            f"PURCHASE ORDER TEXT:\n{text}"
        ),

        "instructions": "You are a purchase order data extraction assistant. Always return valid JSON only.",
        "text": {
            "format": {
                "type": "json_object",
            }
        },
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, status_code = _retry(call, model_cfg.get("retries", 3))
    print("\n========== GPT DEBUG: API RESPONSE ==========")
    print("HTTP Status:", status_code)
    print("Raw API Response:")
    print(raw)
    print("========== END RAW API RESPONSE ==========\n")
    data    = json.loads(raw)

    # Responses API returns output as a list of content blocks
    try:
        content = data["output"][0]["content"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Responses API response structure: {raw[:300]}")
    print("\n========== GPT DEBUG: MODEL JSON CONTENT ==========")
    print(content)
    print("========== END MODEL JSON CONTENT ==========\n")

    result = _parse_json(content)
    result = _normalize_response(
    result,
    positions,
    estimate_confidence=False
    )

    print("\n========== GPT DEBUG: AFTER NORMALIZATION ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("========== END AFTER NORMALIZATION ==========\n")

    result = _apply_duplicate_header_fallback(
        result, pdf_path, positions
    )

    print("\n========== GPT DEBUG: FINAL OUTPUT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("========== END FINAL OUTPUT ==========\n")

    return result


def _extract_cohere_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None):
    print("\n========== COHERE FUNCTION ENTERED ==========")
    print("Model:", model_cfg.get("model_name"))
    print("Endpoint:", model_cfg.get("endpoint"))

    # Build contact-field mappings from the active Excel template.
    # This avoids customer-specific or hardcoded JSON field names.
    contact_keywords = (
        "address", "phone", "telephone", "tel", "fax",
        "email", "website", "web site", "url",
        "adresse", "tél", "site web",
        "supplier", "vendor", "fournisseur",
        "ship to", "ship_to", "expédier",
        "remit", "bill to", "buyer"
    )

    contact_fields = []
    for field in (positions or {}).get("header_fields", []):
        key, label, *_ = field
        key_text = str(key).strip()
        label_text = str(label).strip()
        searchable = f"{key_text} {label_text}".lower()

        if any(word in searchable for word in contact_keywords):
            contact_fields.append(
                f'- JSON key "{key_text}" | Template label: "{label_text}"'
            )

    contact_field_map = (
        "\n".join(contact_fields)
        if contact_fields
        else "No contact-specific fields identified in the template."
    )

    cohere_prompt = (
        f"{prompt}\n\n"

        "ADDITIONAL CONTACT AND HEADER FIELD EXTRACTION RULES:\n"
        "1. Inspect the ENTIRE purchase order, including its header, "
        "middle sections, footer, and contact-information blocks.\n"
        "2. Do not restrict extraction to the first occurrence of a field "
        "or to information immediately below its heading.\n"
        "3. Use the active template field map below. Extract each value "
        "into its corresponding JSON key.\n"
        "4. A contact detail may appear in a separate company-information "
        "block or footer. Associate it using the company name, section "
        "heading, labels, and surrounding context.\n"
        "5. Keep Supplier/Vendor, Ship To, Bill To, Remit To, and Buyer "
        "details separate. Never transfer contact details between them.\n"
        "6. Keep phone numbers separate from fax numbers. Do not mistake "
        "a fax number for a phone number.\n"
        "7. Preserve complete addresses, including street, city, "
        "state/province, postal code, and country when present.\n"
        "8. Extract website URLs and email addresses when explicitly "
        "present, even if they appear in the footer.\n"
        "9. Do not return null merely because a value is not adjacent "
        "to its template label. Search the full document first.\n"
        "10. Do not invent, infer, or externally look up missing values. "
        "Return null only when the value cannot be found or reliably "
        "associated after inspecting the full document.\n\n"

        "CONTACT-RELATED TEMPLATE FIELDS:\n"
        f"{contact_field_map}\n\n"

        "SHIP TO ADDRESS EXTRACTION RULES:\n"
        "1. Extract Ship To and Ship To Address as separate fields.\n"
        "2. Locate the explicit Ship To heading and identify the company "
        "name and its corresponding address.\n"
        "3. Combine wrapped address lines while preserving their content.\n"
        "4. Never use Bill To or Vendor Address as a substitute.\n"
        "5. If the Ship To section genuinely contains no address, return null.\n"
        "ADDITIONAL ADDRESS EXTRACTION RULE:\n"
        "Inspect the complete document footer and contact-information blocks "
        "for standalone addresses.\n"
        "If a separately printed address appears alongside phone numbers, "
        "fax numbers, or website information, extract it into the matching "
        "template address field.\n"
        "Do not automatically leave this field null because a similar address "
        "was already extracted as Ship To Address.\n"
        "An address independently printed in the footer may be extracted "
        "again into the corresponding address field.\n"
        "Do not fabricate or copy an address that is not independently present.\n"
    )

    print("Contact fields included in Cohere prompt:")
    for field in contact_fields:
        print(" ", field)

    print("Calling Llama handler with Cohere-specific instructions...")

    result = _extract_llama_azure(
        pdf_path, model_cfg, cohere_prompt, positions
    )

    print("\nCohere Ship To:", result.get("ship_to"))
    print("Cohere Ship To Address:", result.get("ship_to_address"))

    print("========== COHERE FUNCTION FINISHED ==========")
    print("Returned line items:", len(result.get("line_items", [])))

    return result


# ── Provider registry ──────────────────────────────────────────────────────────

PROVIDERS: dict[str, Any] = {
    "mistral_azure":          _extract_mistral_azure,
    "openai_azure":           _extract_openai_azure,
    "openai_azure_responses": _extract_openai_azure_responses,
    "phi_azure":              _extract_phi_azure,
    "llama_azure":            _extract_llama_azure,
    "cohere_azure":           _extract_cohere_azure,
}


def extract(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None, schema: dict = None) -> dict:
    """Route extraction to the correct Azure provider."""
    provider = model_cfg.get("provider")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(PROVIDERS.keys())}"
        )
    print(f"  [{provider}] Extracting with model: {model_cfg.get('model_name')}")
    return PROVIDERS[provider](pdf_path, model_cfg, prompt, positions)