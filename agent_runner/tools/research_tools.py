from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import httpx

from ..config import ResearchConfig
from ..research_cache import canonicalize_url
from . import failure, success

_ARXIV_LAST_REQUEST_AT = 0.0
_TEXT_CONTENT_TYPES = (
    "text/plain",
    "text/html",
    "application/json",
    "application/xml",
    "text/xml",
    "application/atom+xml",
)
_BLOCKED_URL_KEYWORDS = (
    "generate_data",
    "numerical_solver",
    "simulate",
    "solve_pde",
)


@dataclass(slots=True)
class ParsedHtml:
    title: str = ""
    text_chunks: list[str] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)


class _SimpleHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parsed = ParsedHtml()
        self._tag_stack: list[str] = []
        self._current_link: str | None = None
        self._link_text_chunks: list[str] = []
        self._capture_code = False
        self._current_code_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        attrs_map = dict(attrs)
        if tag == "a":
            self._current_link = attrs_map.get("href")
            self._link_text_chunks = []
        if tag in {"pre", "code"}:
            self._capture_code = True
            self._current_code_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        elif tag in self._tag_stack:
            self._tag_stack.remove(tag)

        if tag == "a" and self._current_link:
            self.parsed.links.append(
                {
                    "href": self._current_link,
                    "text": " ".join(chunk for chunk in self._link_text_chunks if chunk).strip(),
                }
            )
            self._current_link = None
            self._link_text_chunks = []
        if tag in {"pre", "code"} and self._capture_code:
            block = "".join(self._current_code_chunks).strip()
            if block:
                self.parsed.code_blocks.append(block)
            self._capture_code = False
            self._current_code_chunks = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag == "title" and not self.parsed.title:
            self.parsed.title = text
        if current_tag not in {"script", "style"}:
            self.parsed.text_chunks.append(text)
            if self._current_link is not None:
                self._link_text_chunks.append(text)
            if self._capture_code:
                self._current_code_chunks.append(data)


def _domain_allowed(hostname: str, research: ResearchConfig) -> bool:
    lowered = hostname.lower()
    return any(lowered == domain or lowered.endswith(f".{domain}") for domain in research.allowed_domains)


def _url_allowed(url: str, research: ResearchConfig) -> bool:
    hostname = urlsplit(url).netloc.split("@")[-1].split(":")[0]
    return bool(hostname) and _domain_allowed(hostname, research)


def _url_has_blocked_extension(url: str, research: ResearchConfig) -> str | None:
    lowered_path = urlsplit(url).path.lower()
    for extension in research.blocked_extensions:
        if lowered_path.endswith(extension):
            return extension
    return None


def _url_has_blocked_keyword(url: str) -> str | None:
    lowered = url.lower()
    for keyword in _BLOCKED_URL_KEYWORDS:
        if keyword in lowered:
            return keyword
    if "pdebench" in lowered and any(token in lowered for token in ("dataset", "download", "mirror", "checkpoint")):
        return "pdebench_data_artifact"
    return None


def _normalize_github_blob_url(url: str, research: ResearchConfig) -> str:
    parts = urlsplit(url)
    if parts.netloc.lower() != "github.com" or "/blob/" not in parts.path or not research.allow_raw_github:
        return url
    segments = parts.path.strip("/").split("/")
    if len(segments) < 5:
        return url
    owner, repo = segments[0], segments[1]
    branch = segments[3]
    path = "/".join(segments[4:])
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cache_extension(content_type: str, url: str) -> str:
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized == "text/html":
        return ".html"
    if normalized in {"application/json"}:
        return ".json"
    if normalized in {"application/xml", "text/xml", "application/atom+xml"}:
        return ".xml"
    if normalized == "application/pdf":
        return ".pdf"
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix or ".txt"


def _write_cache_file(*, research: ResearchConfig, url: str, payload: bytes, content_type: str, pdf: bool = False) -> Path:
    sha = _sha256_bytes(payload)
    extension = ".pdf" if pdf else _cache_extension(content_type, url)
    base_dir = research.papers_dir if pdf else research.raw_cache_dir
    assert base_dir is not None
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{sha}{extension}"
    path.write_bytes(payload)
    return path


def _client_context(research: ResearchConfig, http_client: httpx.Client | None) -> tuple[httpx.Client, object]:
    if http_client is not None:
        return http_client, nullcontext(http_client)
    client = httpx.Client(
        timeout=research.request_timeout_seconds,
        headers={"User-Agent": research.user_agent},
        follow_redirects=True,
    )
    return client, client


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retry_sleep_seconds: float = 1.0,
    **kwargs,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.request(method, url, **kwargs)
            if response.status_code >= 500 and attempt == 0:
                time.sleep(retry_sleep_seconds)
                continue
            return response
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(retry_sleep_seconds)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable")


def _parse_arxiv_atom(payload: str, *, query: str) -> list[dict]:
    namespaces = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(payload)
    results: list[dict] = []
    for entry in root.findall("atom:entry", namespaces):
        entry_id = (entry.findtext("atom:id", default="", namespaces=namespaces) or "").strip()
        abs_url = entry_id.replace("http://", "https://")
        pdf_url = ""
        for link in entry.findall("atom:link", namespaces):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        authors = [
            (author.findtext("atom:name", default="", namespaces=namespaces) or "").strip()
            for author in entry.findall("atom:author", namespaces)
        ]
        categories = [category.attrib.get("term", "") for category in entry.findall("atom:category", namespaces)]
        results.append(
            {
                "source_type": "arxiv",
                "title": " ".join((entry.findtext("atom:title", default="", namespaces=namespaces) or "").split()),
                "authors": [author for author in authors if author],
                "abstract": " ".join((entry.findtext("atom:summary", default="", namespaces=namespaces) or "").split()),
                "published": (entry.findtext("atom:published", default="", namespaces=namespaces) or "").strip(),
                "updated": (entry.findtext("atom:updated", default="", namespaces=namespaces) or "").strip(),
                "arxiv_id": abs_url.rsplit("/", 1)[-1],
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "categories": [category for category in categories if category],
                "relevance_hint": f"Matched arXiv query: {query}",
            }
        )
    return results


def search_arxiv(
    *,
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    research: ResearchConfig,
    http_client: httpx.Client | None = None,
) -> dict:
    if not research.enabled or not research.providers.arxiv.enabled:
        return failure("search_arxiv", "arXiv research provider is disabled")

    global _ARXIV_LAST_REQUEST_AT
    if http_client is None:
        elapsed = time.monotonic() - _ARXIV_LAST_REQUEST_AT
        wait_seconds = max(0.0, research.providers.arxiv.min_interval_seconds - elapsed)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    if http_client is not None:
        client, client_cm = _client_context(research, http_client)
        with client_cm:
            response = _request_with_retry(
                client,
                "GET",
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": sort_by,
                },
                headers={"User-Agent": research.user_agent},
            )
            response.raise_for_status()
            _ARXIV_LAST_REQUEST_AT = time.monotonic()
            results = _parse_arxiv_atom(response.text, query=query)
            return success("search_arxiv", f"Found {len(results)} arXiv results", query=query, results=results)

    params = urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
        }
    )
    request = Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": research.user_agent},
    )
    with urlopen(request, timeout=research.request_timeout_seconds) as response:
        payload = response.read().decode("utf-8", errors="replace")
    _ARXIV_LAST_REQUEST_AT = time.monotonic()
    results = _parse_arxiv_atom(payload, query=query)
    return success("search_arxiv", f"Found {len(results)} arXiv results", query=query, results=results)


def _github_request(
    *,
    research: ResearchConfig,
    http_client: httpx.Client | None,
    path: str,
    params: dict[str, str | int],
) -> httpx.Response:
    provider = research.providers.github
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": research.user_agent,
    }
    auth_headers = dict(headers)
    if provider.api_key:
        auth_headers["Authorization"] = f"Bearer {provider.api_key}"
    elif not provider.allow_unauthenticated:
        raise PermissionError("GitHub API token is required by config")

    client, client_cm = _client_context(research, http_client)
    with client_cm:
        response = _request_with_retry(
            client,
            "GET",
            f"https://api.github.com{path}",
            params=params,
            headers=auth_headers,
        )
        if response.status_code == 401 and provider.api_key and provider.allow_unauthenticated:
            response = _request_with_retry(client, "GET", f"https://api.github.com{path}", params=params, headers=headers)
        response.raise_for_status()
        return response


def search_github(
    *,
    query: str,
    kind: str = "repositories",
    max_results: int = 10,
    research: ResearchConfig,
    http_client: httpx.Client | None = None,
) -> dict:
    if not research.enabled or not research.providers.github.enabled:
        return failure("search_github", "GitHub research provider is disabled")
    if kind not in {"repositories", "code"}:
        return failure("search_github", f"Unsupported GitHub search kind: {kind}")

    path = "/search/repositories" if kind == "repositories" else "/search/code"
    try:
        response = _github_request(
            research=research,
            http_client=http_client,
            path=path,
            params={"q": query, "per_page": max_results},
        )
    except (PermissionError, httpx.HTTPStatusError) as exc:
        return failure("search_github", f"GitHub API request failed: {exc}")
    payload = response.json()
    items = payload.get("items", [])
    results: list[dict] = []
    for item in items:
        repository = item if kind == "repositories" else item.get("repository", {})
        owner = (repository.get("owner") or {}).get("login", "")
        repo = repository.get("name", "")
        license_info = repository.get("license") or {}
        record = {
            "source_type": "github_repo" if kind == "repositories" else "github_file",
            "repo": repo,
            "owner": owner,
            "path": "" if kind == "repositories" else item.get("path", ""),
            "url": item.get("html_url") or repository.get("html_url", ""),
            "raw_url": "",
            "license_hint": license_info.get("spdx_id") or license_info.get("name") or "",
            "stars": int(repository.get("stargazers_count", 0) or 0),
            "updated_at": repository.get("updated_at", ""),
            "summary": repository.get("description") or item.get("path", ""),
        }
        if kind == "code":
            default_branch = repository.get("default_branch", "main")
            path_value = item.get("path", "")
            if owner and repo and path_value:
                record["raw_url"] = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{path_value}"
        results.append(record)

    return success("search_github", f"Found {len(results)} GitHub results", query=query, kind=kind, results=results)


def _normalize_web_results(provider: str, records: list[dict], *, query: str) -> list[dict]:
    normalized: list[dict] = []
    for record in records:
        url = str(record.get("url", ""))
        normalized.append(
            {
                "source_type": "web_page",
                "title": str(record.get("title", "")),
                "url": url,
                "snippet": str(record.get("snippet", "")),
                "domain": urlsplit(url).netloc.lower(),
                "provider": provider,
                "relevance_hint": f"Matched web query: {query}",
            }
        )
    return normalized


def _provider_key_available(provider: str, research: ResearchConfig) -> bool:
    web = research.providers.web
    if provider == "tavily":
        return bool(web.tavily_api_key)
    if provider == "exa":
        return bool(web.exa_api_key)
    if provider == "brave":
        return bool(web.brave_api_key)
    if provider == "google_cse":
        return bool(web.google_api_key and web.google_cse_id)
    return False


def _search_tavily(*, query: str, max_results: int, domains: list[str] | None, research: ResearchConfig, client: httpx.Client) -> list[dict]:
    response = _request_with_retry(
        client,
        "POST",
        "https://api.tavily.com/search",
        json={
            "api_key": research.providers.web.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_domains": domains or [],
            "search_depth": "basic",
        },
        headers={"User-Agent": research.user_agent},
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    return _normalize_web_results(
        "tavily",
        [{"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")} for item in results],
        query=query,
    )


def _search_exa(*, query: str, max_results: int, domains: list[str] | None, research: ResearchConfig, client: httpx.Client) -> list[dict]:
    response = _request_with_retry(
        client,
        "POST",
        "https://api.exa.ai/search",
        json={
            "query": query,
            "numResults": max_results,
            "includeDomains": domains or [],
        },
        headers={
            "User-Agent": research.user_agent,
            "x-api-key": research.providers.web.exa_api_key or "",
        },
    )
    response.raise_for_status()
    payload = response.json()
    return _normalize_web_results(
        "exa",
        [{"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("text", "")} for item in payload.get("results", [])],
        query=query,
    )


def _search_brave(*, query: str, max_results: int, domains: list[str] | None, research: ResearchConfig, client: httpx.Client) -> list[dict]:
    params = {"q": query, "count": max_results}
    if domains:
        params["site"] = ",".join(domains)
    response = _request_with_retry(
        client,
        "GET",
        "https://api.search.brave.com/res/v1/web/search",
        params=params,
        headers={
            "User-Agent": research.user_agent,
            "X-Subscription-Token": research.providers.web.brave_api_key or "",
        },
    )
    response.raise_for_status()
    payload = response.json()
    return _normalize_web_results(
        "brave",
        [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("description", "")}
            for item in payload.get("web", {}).get("results", [])
        ],
        query=query,
    )


def _search_google_cse(*, query: str, max_results: int, domains: list[str] | None, research: ResearchConfig, client: httpx.Client) -> list[dict]:
    params = {
        "key": research.providers.web.google_api_key or "",
        "cx": research.providers.web.google_cse_id or "",
        "q": query,
        "num": max_results,
    }
    if domains:
        params["siteSearch"] = " OR ".join(domains)
    response = _request_with_retry(
        client,
        "GET",
        "https://www.googleapis.com/customsearch/v1",
        params=params,
        headers={"User-Agent": research.user_agent},
    )
    response.raise_for_status()
    payload = response.json()
    return _normalize_web_results(
        "google_cse",
        [{"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")} for item in payload.get("items", [])],
        query=query,
    )


def web_search(
    *,
    query: str,
    max_results: int = 10,
    domains: list[str] | None = None,
    research: ResearchConfig,
    http_client: httpx.Client | None = None,
) -> dict:
    if not research.enabled:
        return failure("web_search", "Research tools are disabled")

    skipped: list[str] = []
    errors: list[str] = []
    client, client_cm = _client_context(research, http_client)
    with client_cm:
        for provider in research.providers.web.provider_order:
            if not _provider_key_available(provider, research):
                skipped.append(provider)
                continue
            try:
                if provider == "tavily":
                    results = _search_tavily(query=query, max_results=max_results, domains=domains, research=research, client=client)
                elif provider == "exa":
                    results = _search_exa(query=query, max_results=max_results, domains=domains, research=research, client=client)
                elif provider == "brave":
                    results = _search_brave(query=query, max_results=max_results, domains=domains, research=research, client=client)
                elif provider == "google_cse":
                    results = _search_google_cse(
                        query=query,
                        max_results=max_results,
                        domains=domains,
                        research=research,
                        client=client,
                    )
                else:
                    skipped.append(provider)
                    continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
                continue
            return success(
                "web_search",
                f"Used {provider} and found {len(results)} results",
                provider=provider,
                skipped_providers=skipped,
                errors=errors,
                results=results,
            )

    return success(
        "web_search",
        "No configured local web search provider available",
        provider=None,
        skipped_providers=skipped,
        errors=errors,
        results=[],
    )


def fetch_url(
    *,
    url: str,
    purpose: str = "read_code_or_paper",
    research: ResearchConfig,
    http_client: httpx.Client | None = None,
) -> dict:
    del purpose
    normalized_url = _normalize_github_blob_url(canonicalize_url(url), research)
    blocked_extension = _url_has_blocked_extension(normalized_url, research)
    if blocked_extension:
        return failure("fetch_url", f"Blocked by policy: extension {blocked_extension} is not allowed", url=normalized_url, blocked=True)
    blocked_keyword = _url_has_blocked_keyword(normalized_url)
    if blocked_keyword:
        return failure("fetch_url", f"Blocked by policy: URL matches forbidden token {blocked_keyword}", url=normalized_url, blocked=True)
    if not _url_allowed(normalized_url, research):
        return failure("fetch_url", "Blocked by policy: domain is not in allowed_domains", url=normalized_url, blocked=True)

    client, client_cm = _client_context(research, http_client)
    with client_cm:
        response = _request_with_retry(client, "GET", normalized_url, headers={"User-Agent": research.user_agent})
        if response.status_code in {401, 403, 429}:
            return failure(
                "fetch_url",
                f"Blocked by remote access policy: HTTP {response.status_code}",
                url=normalized_url,
                blocked=True,
                status_code=response.status_code,
            )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type == "application/pdf":
            return failure("fetch_url", "Use fetch_pdf for PDF content", url=normalized_url, blocked=True)
        if content_type and not any(content_type.startswith(prefix) for prefix in _TEXT_CONTENT_TYPES):
            return failure("fetch_url", f"Unsupported content-type: {content_type}", url=normalized_url, blocked=True)
        payload = response.content
        if len(payload) > research.max_response_bytes:
            return failure("fetch_url", "Response exceeds max_response_bytes", url=normalized_url, blocked=True, size_bytes=len(payload))
        text = payload.decode(response.encoding or "utf-8", errors="replace")
        cache_path = _write_cache_file(research=research, url=normalized_url, payload=payload, content_type=content_type or "text/plain")
        return success(
            "fetch_url",
            f"Fetched {len(payload)} bytes from {urlsplit(normalized_url).netloc}",
            url=normalized_url,
            final_url=str(response.url),
            content_type=content_type or "text/plain",
            content=text,
            content_sha256=_sha256_bytes(payload),
            cache_path=str(cache_path),
            size_bytes=len(payload),
        )


def fetch_pdf(*, url: str, research: ResearchConfig, http_client: httpx.Client | None = None) -> dict:
    normalized_url = canonicalize_url(url)
    if not _url_allowed(normalized_url, research):
        return failure("fetch_pdf", "Blocked by policy: domain is not in allowed_domains", url=normalized_url, blocked=True)
    hostname = urlsplit(normalized_url).netloc.lower()
    if hostname not in {"arxiv.org", "export.arxiv.org"} and not normalized_url.lower().endswith(".pdf"):
        return failure("fetch_pdf", "Blocked by policy: PDF host is not explicitly allowed", url=normalized_url, blocked=True)

    client, client_cm = _client_context(research, http_client)
    with client_cm:
        response = _request_with_retry(client, "GET", normalized_url, headers={"User-Agent": research.user_agent})
        response.raise_for_status()
        payload = response.content
        if len(payload) > research.max_pdf_bytes:
            return failure("fetch_pdf", "PDF exceeds max_pdf_bytes", url=normalized_url, blocked=True, size_bytes=len(payload))
        path = _write_cache_file(research=research, url=normalized_url, payload=payload, content_type="application/pdf", pdf=True)
        return success(
            "fetch_pdf",
            f"Fetched PDF with {len(payload)} bytes",
            url=normalized_url,
            local_path=str(path),
            content_sha256=_sha256_bytes(payload),
            size_bytes=len(payload),
        )


def parse_pdf(*, path: str | Path) -> dict:
    local_path = Path(path)
    if not local_path.exists():
        return failure("parse_pdf", "PDF path does not exist", path=str(local_path))
    try:
        from pypdf import PdfReader
    except ImportError:
        return failure("parse_pdf", "pypdf is not installed", path=str(local_path))

    reader = PdfReader(str(local_path))
    texts = [(page.extract_text() or "") for page in reader.pages]
    full_text = "\n".join(texts).strip()
    warnings: list[str] = []
    if not full_text:
        warnings.append("No text extracted from PDF.")
    lowered = full_text.lower()
    reference_index = lowered.rfind("references")
    references_excerpt = full_text[reference_index : reference_index + 3000] if reference_index >= 0 else full_text[-3000:]
    return success(
        "parse_pdf",
        f"Parsed {len(reader.pages)} PDF pages",
        path=str(local_path),
        page_count=len(reader.pages),
        text_excerpt=full_text[:4000],
        references_excerpt=references_excerpt,
        warnings=warnings,
    )


def parse_html(
    *,
    path_or_url: str,
    research: ResearchConfig,
    http_client: httpx.Client | None = None,
) -> dict:
    source = path_or_url
    if path_or_url.startswith(("http://", "https://")):
        fetched = fetch_url(url=path_or_url, research=research, http_client=http_client)
        if not fetched["ok"]:
            return fetched
        html_text = fetched["data"]["content"]
        source = fetched["data"]["cache_path"]
    else:
        html_text = Path(path_or_url).read_text(encoding="utf-8")

    parser = _SimpleHtmlParser()
    parser.feed(html_text)
    main_text = " ".join(parser.parsed.text_chunks).strip()
    return success(
        "parse_html",
        f"Parsed HTML from {source}",
        source=source,
        title=parser.parsed.title,
        main_text=main_text[:8000],
        code_blocks=parser.parsed.code_blocks[:20],
        links=parser.parsed.links[:50],
    )
