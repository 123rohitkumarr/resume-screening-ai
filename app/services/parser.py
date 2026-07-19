import io
import fitz
from docx import Document
from app.utils.helpers import clean_text, setup_logger

log = setup_logger(__name__)

def parse_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        log.error(f"Error parsing PDF: {e}")
    return text

def parse_docx(file_bytes: bytes) -> str:
    text = ""
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        log.error(f"Error parsing DOCX: {e}")
    return text

def parse_text(content: str) -> str:
    return clean_text(content)

def parse_resume(file_bytes: bytes = None, filename: str = None, raw_text: str = None) -> str:
    log.info(f"Parsing resume: filename={filename}")
    if raw_text:
        return parse_text(raw_text)
    
    if not file_bytes or not filename:
        raise ValueError("Must provide either raw_text or both file_bytes and filename")
    
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        parsed = parse_pdf(file_bytes)
    elif ext in ['doc', 'docx']:
        parsed = parse_docx(file_bytes)
    elif ext == 'txt':
        try:
            parsed = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            parsed = file_bytes.decode('latin-1', errors='ignore')
    else:
        log.error(f"Unsupported format: {ext}")
        raise ValueError(f"Unsupported format: {ext}")
    
    return clean_text(parsed)
