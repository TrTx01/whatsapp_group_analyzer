import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MediaResult:
    path: str
    status: str
    text: str
    error: Optional[str] = None


def _have_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_text(path: str, whisper_model: Optional[str] = None) -> MediaResult:
    ext = Path(path).suffix.lower()

    if ext in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}:
        return _extract_image(path)
    if ext in {".pdf"}:
        return _extract_pdf(path)
    if ext in {".docx"}:
        return _extract_docx(path)
    if ext in {".mp3", ".m4a", ".wav", ".aac", ".mp4", ".mov", ".mkv"}:
        return _extract_audio(path, whisper_model)

    return MediaResult(path=path, status="unsupported", text="")


def _extract_image(path: str) -> MediaResult:
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        return MediaResult(path=path, status="missing_deps", text="", error=str(e))

    if not _have_tesseract():
        for candidate in ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            return MediaResult(path=path, status="missing_tesseract", text="")

    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
        return MediaResult(path=path, status="ok", text=text.strip())
    except Exception as e:
        return MediaResult(path=path, status="error", text="", error=str(e))


def _extract_pdf(path: str) -> MediaResult:
    try:
        from pypdf import PdfReader
    except Exception as e:
        return MediaResult(path=path, status="missing_deps", text="", error=str(e))

    try:
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return MediaResult(path=path, status="ok", text="\n".join(text).strip())
    except Exception as e:
        return MediaResult(path=path, status="error", text="", error=str(e))


def _extract_docx(path: str) -> MediaResult:
    try:
        import docx
    except Exception as e:
        return MediaResult(path=path, status="missing_deps", text="", error=str(e))

    try:
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return MediaResult(path=path, status="ok", text=text.strip())
    except Exception as e:
        return MediaResult(path=path, status="error", text="", error=str(e))


def _extract_audio(path: str, whisper_model: Optional[str]) -> MediaResult:
    if whisper_model is None:
        return MediaResult(path=path, status="missing_model", text="")
    try:
        import whisper
    except Exception as e:
        return MediaResult(path=path, status="missing_deps", text="", error=str(e))

    if not _have_ffmpeg():
        return MediaResult(path=path, status="missing_ffmpeg", text="")

    try:
        model = whisper.load_model(whisper_model)
        result = model.transcribe(path, fp16=False)
        text = result.get("text", "")
        return MediaResult(path=path, status="ok", text=text.strip())
    except Exception as e:
        return MediaResult(path=path, status="error", text="", error=str(e))
