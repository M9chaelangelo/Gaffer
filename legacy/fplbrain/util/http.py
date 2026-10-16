"""A polite, cached HTTP layer.

Every remote source the engine uses is read-only and rate-limited by courtesy
rather than by contract, so this module does three things: it caches responses
on disk with a per-endpoint TTL, it spaces requests to the same host, and it
retries transient failures with exponential backoff. Nothing here is specific to
any one data source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "fplbrain/0.1 (personal Fantasy Premier League research tool; "
    "single-user, low-volume)"
)

# Status codes worth retrying: transient server problems and rate limits.
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    """Raised when a request fails after exhausting retries."""

    def __init__(self, message: str, status: int | None = None, url: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url


@dataclass
class CacheEntry:
    fetched_at: float
    status: int
    body: str

    @property
    def age(self) -> float:
        return time.time() - self.fetched_at


class DiskCache:
    """A trivially simple JSON-on-disk cache keyed by URL and parameters."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(url: str, params: Mapping[str, Any] | None) -> str:
        payload = url + "?" + json.dumps(dict(params or {}), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _path(self, url: str, params: Mapping[str, Any] | None) -> Path:
        host = urlparse(url).netloc.replace(":", "_") or "local"
        directory = self.root / host
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self._key(url, params)}.json"

    def read(self, url: str, params: Mapping[str, Any] | None) -> CacheEntry | None:
        path = self._path(url, params)
        if not path.exists():
            return None
        try:
            with path.open() as fh:
                blob = json.load(fh)
            return CacheEntry(
                fetched_at=float(blob["fetched_at"]),
                status=int(blob["status"]),
                body=blob["body"],
            )
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            # A corrupt cache file is never worth crashing over.
            return None

    def write(self, url: str, params: Mapping[str, Any] | None, status: int, body: str) -> None:
        path = self._path(url, params)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w") as fh:
                json.dump(
                    {"url": url, "fetched_at": time.time(), "status": status, "body": body},
                    fh,
                )
            tmp.replace(path)
        except OSError as exc:  # pragma: no cover - disk problems
            log.warning("could not write cache for %s: %s", url, exc)

    def clear(self) -> int:
        removed = 0
        for path in self.root.rglob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


class RateLimiter:
    """Enforces a minimum gap between requests to the same host."""

    def __init__(self, min_interval: float = 0.6):
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            previous = self._last.get(host)
            if previous is not None:
                gap = self.min_interval - (now - previous)
                if gap > 0:
                    time.sleep(gap)
                    now = time.monotonic()
            self._last[host] = now


class HttpClient:
    """Cached HTTP GET with retries. JSON and text helpers on top."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval: float = 0.6,
        max_retries: int = 4,
        timeout: float = 20.0,
        offline: bool = False,
    ):
        self.cache = DiskCache(cache_dir)
        self.limiter = RateLimiter(min_interval)
        self.max_retries = max_retries
        self.timeout = timeout
        self.offline = offline
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept": "application/json, text/plain, */*"}
        )
        # Populated as requests are made, so callers can report data freshness.
        self.last_from_cache: bool = False
        self.last_fetched_at: float | None = None

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        ttl: int = 3600,
        headers: Mapping[str, str] | None = None,
        force_refresh: bool = False,
    ) -> str:
        """Return the response body, from cache when it is fresh enough."""
        entry = self.cache.read(url, params)
        if entry is not None and not force_refresh and entry.age < ttl:
            self.last_from_cache = True
            self.last_fetched_at = entry.fetched_at
            return entry.body

        if self.offline:
            if entry is not None:
                log.info("offline: serving stale cache for %s (age %.0fs)", url, entry.age)
                self.last_from_cache = True
                self.last_fetched_at = entry.fetched_at
                return entry.body
            raise HttpError(f"offline and nothing cached for {url}", url=url)

        host = urlparse(url).netloc
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait(host)
            try:
                response = self.session.get(
                    url, params=params, headers=dict(headers or {}), timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                log.debug("request error for %s (attempt %d): %s", url, attempt + 1, exc)
            else:
                if response.status_code == 200:
                    self.cache.write(url, params, response.status_code, response.text)
                    self.last_from_cache = False
                    self.last_fetched_at = time.time()
                    return response.text
                if response.status_code not in RETRY_STATUS:
                    # A stale cache beats no data at all for a hard failure.
                    if entry is not None:
                        log.warning(
                            "%s returned %s; falling back to cache aged %.0fs",
                            url,
                            response.status_code,
                            entry.age,
                        )
                        self.last_from_cache = True
                        self.last_fetched_at = entry.fetched_at
                        return entry.body
                    raise HttpError(
                        f"{url} returned HTTP {response.status_code}",
                        status=response.status_code,
                        url=url,
                    )
                last_error = HttpError(
                    f"HTTP {response.status_code}", status=response.status_code, url=url
                )

            backoff = min(2**attempt, 16) + random.uniform(0, 0.5)
            log.debug("retrying %s in %.1fs", url, backoff)
            time.sleep(backoff)

        if entry is not None:
            log.warning("all retries failed for %s; using cache aged %.0fs", url, entry.age)
            self.last_from_cache = True
            self.last_fetched_at = entry.fetched_at
            return entry.body
        raise HttpError(f"failed to fetch {url}: {last_error}", url=url)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        body = self.get(url, **kwargs)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise HttpError(f"{url} did not return JSON: {exc}", url=url) from exc
