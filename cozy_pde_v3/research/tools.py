from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ResearchToolFlags:
    fetch_pdf: bool = True
    fetch_url: bool = True
    parse_html: bool = True
    parse_pdf: bool = True
    search_arxiv: bool = True
    search_github: bool = True


def _client(http_client: httpx.Client | None) -> httpx.Client:
    return http_client or httpx.Client(timeout=20.0)


def search_arxiv(query: str, *, http_client: httpx.Client | None = None, max_results: int = 5) -> list[dict[str, Any]]:
    response = _client(http_client).get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link = ""
        for link_node in entry.findall("atom:link", ns):
            href = link_node.attrib.get("href", "")
            if href:
                link = href
                break
        results.append({"title": title, "url": link, "source_type": "arxiv"})
    return results


def search_github(query: str, *, http_client: httpx.Client | None = None, per_page: int = 5) -> list[dict[str, Any]]:
    response = _client(http_client).get(
        "https://api.github.com/search/repositories",
        params={"q": query, "per_page": per_page},
        headers={"Accept": "application/vnd.github+json"},
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items", []) if isinstance(payload, dict) else []
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("full_name", item.get("name", ""))).strip(),
                "url": str(item.get("html_url", "")).strip(),
                "source_type": "github_repo",
            }
        )
    return results


def fetch_url(url: str, *, http_client: httpx.Client | None = None) -> dict[str, Any]:
    response = _client(http_client).get(url)
    response.raise_for_status()
    return {
        "url": str(response.request.url),
        "content_type": response.headers.get("content-type", ""),
        "text": response.text,
    }


def fetch_pdf(url: str, *, http_client: httpx.Client | None = None) -> bytes:
    response = _client(http_client).get(url)
    response.raise_for_status()
    return response.content


def parse_html(html: str) -> dict[str, Any]:
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", html)
    body = re.sub(r"\s+", " ", body).strip()
    return {
        "title": title_match.group(1).strip() if title_match else "",
        "text": body,
    }


def parse_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    try:
        text = pdf_bytes.decode("latin-1", errors="ignore")
    except Exception:  # noqa: BLE001
        text = ""
    return {"text": text[:2000], "metadata": {"size": len(pdf_bytes)}}


__all__ = [
    "ResearchToolFlags",
    "fetch_pdf",
    "fetch_url",
    "parse_html",
    "parse_pdf",
    "search_arxiv",
    "search_github",
]
