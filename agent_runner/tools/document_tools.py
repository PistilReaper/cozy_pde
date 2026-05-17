from __future__ import annotations

from pathlib import Path

from ..safety import WorkspaceSafety
from . import failure, success


def _pdf_escape(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(character if ord(character) < 128 else "?" for character in escaped)


def _markdown_to_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw_line in content.splitlines():
        normalized = raw_line.strip()
        for prefix in ("# ", "## ", "### ", "- ", "* ", "`"):
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix)
        lines.append(normalized or " ")
    return lines or [" "]


def _build_simple_pdf(text: str) -> bytes:
    lines = _markdown_to_lines(text)
    content_lines = ["BT", "/F1 12 Tf", "72 770 Td", "14 TL"]
    for index, line in enumerate(lines):
        escaped = _pdf_escape(line)
        operator = "Tj" if index == 0 else "T*"
        if index == 0:
            content_lines.append(f"({escaped}) {operator}")
        else:
            content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("ascii", errors="ignore")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("ascii") + stream + b"\nendstream endobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def generate_methodology_pdf(*, content: str, path: str = "submission/methodology.pdf", safety: WorkspaceSafety) -> dict:
    check = safety.validate_write_path(path)
    if not check.ok:
        return failure("generate_methodology_pdf", check.error or "write check failed", path=path)
    assert check.resolved_path is not None
    if check.resolved_path.suffix.lower() != ".pdf":
        return failure("generate_methodology_pdf", "Output path must end with .pdf", path=str(check.resolved_path))
    payload = _build_simple_pdf(content)
    check.resolved_path.parent.mkdir(parents=True, exist_ok=True)
    check.resolved_path.write_bytes(payload)
    return success(
        "generate_methodology_pdf",
        f"Generated methodology PDF at {check.resolved_path.name}",
        path=str(check.resolved_path),
        size_bytes=len(payload),
    )
