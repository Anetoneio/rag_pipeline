"""
Document loader for the pipeline 
i need to def a loader func for data uploaded
Each loader returns (full_text: str, metadata: dict)
i am  keeping loader seprate because it would be easier to add other extensions later easier

"""

import os
from typing import Dict, Tuple


def load_txt_or_md(path: str) -> Tuple[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    metadata = {"source": os.path.basename(path), "type": "text"}
    return text, metadata


def load_pdf(path: str) -> Tuple[str, Dict]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    full_text = "\n".join(pages_text)
    metadata = {
        "source": os.path.basename(path),
        "type": "pdf",
        "num_pages": len(reader.pages),
    }
    return full_text, metadata


def load_document(path: str) -> Tuple[str, Dict]:
    """Dispatch on file extension. Raises ValueError for unsupported types."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md"):
        return load_txt_or_md(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")