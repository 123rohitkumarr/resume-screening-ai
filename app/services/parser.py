import io
import re
import os
import hashlib
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from docx import Document
from app.config import settings
from app.utils.helpers import clean_text, setup_logger

log = setup_logger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and remove dangerous characters."""
    if not filename:
        return "unnamed_resume"
    # Take basename only to strip directory path
    base = Path(filename).name
    # Replace any characters other than alphanumeric, dots, dashes, and underscores
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    # Prevent hidden files
    sanitized = sanitized.lstrip(".")
    return sanitized or "resume"


def compute_content_hash(text: str) -> str:
    """Generate SHA256 hash of cleaned text for duplicate resume detection."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF with corruption detection."""
    text_chunks = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise ValueError("PDF document has 0 pages.")
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text_chunks.append(page_text)
    except Exception as e:
        log.error(f"Error parsing PDF: {e}")
        raise ValueError(f"Corrupted or invalid PDF document: {str(e)}")

    extracted = "\n".join(text_chunks).strip()
    if not extracted:
        raise ValueError("PDF contains no extractable text (file may be an image-only scan).")
    return extracted


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX including both paragraphs and embedded tables."""
    text_chunks = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                text_chunks.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_items = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_items:
                    text_chunks.append(" | ".join(row_items))
    except Exception as e:
        log.error(f"Error parsing DOCX: {e}")
        raise ValueError(f"Corrupted or invalid DOCX document: {str(e)}")

    extracted = "\n".join(text_chunks).strip()
    if not extracted:
        raise ValueError("DOCX document contains no extractable text.")
    return extracted


def parse_txt(file_bytes: bytes) -> str:
    """Extract text from plain text file with multiple encoding fallbacks."""
    for enc in ["utf-8", "utf-8-sig", "utf-16", "latin-1", "cp1252"]:
        try:
            decoded = file_bytes.decode(enc).strip()
            if decoded:
                return decoded
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("Unable to decode text file. Ensure it is encoded in UTF-8 or standard ASCII.")


def parse_resume(file_bytes: Optional[bytes] = None, filename: Optional[str] = None, raw_text: Optional[str] = None) -> str:
    """
    Parse resume from file bytes or raw text string with comprehensive validation.
    Enforces file size limit, supported extension, corruption handling, and empty detection.
    """
    if raw_text is not None:
        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("Provided resume text is empty.")
        return cleaned

    if not file_bytes or not filename:
        raise ValueError("Must provide either raw_text or both file_bytes and filename.")

    # File size validation
    max_bytes = getattr(settings, "max_upload_size_mb", 10) * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(f"File size ({len(file_bytes) / 1024 / 1024:.1f} MB) exceeds maximum allowed limit of {settings.max_upload_size_mb} MB.")

    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    # Safe filename and extension validation
    clean_name = sanitize_filename(filename)
    ext = clean_name.lower().split(".")[-1] if "." in clean_name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")

    log.info(f"Parsing resume file: {clean_name} ({len(file_bytes)} bytes)")

    if ext == "pdf":
        parsed = parse_pdf(file_bytes)
    elif ext in ["doc", "docx"]:
        parsed = parse_docx(file_bytes)
    elif ext == "txt":
        parsed = parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    cleaned = clean_text(parsed)
    if not cleaned:
        raise ValueError("Extracted resume text is empty after cleaning.")

    return cleaned
