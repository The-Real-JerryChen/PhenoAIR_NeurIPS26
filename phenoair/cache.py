from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 1


class CachePolicy(str, Enum):
    OFF = "off"
    READ_ONLY = "read-only"
    READ_WRITE = "read-write"
    REFRESH = "refresh"

    @property
    def can_read(self) -> bool:
        return self in {self.READ_ONLY, self.READ_WRITE}

    @property
    def can_write(self) -> bool:
        return self in {self.READ_WRITE, self.REFRESH}


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class JsonCacheStore:
    """Versioned, content-addressed local cache for deterministic tool results."""

    root: Path
    policy: CachePolicy = CachePolicy.READ_WRITE
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)
    _writes: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @staticmethod
    def _safe_segment(value: str, field_name: str) -> str:
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(f"{field_name} must be one non-empty path segment")
        return value

    def _path(self, namespace: str, producer_version: str, key: dict[str, Any]) -> Path:
        namespace = self._safe_segment(namespace, "namespace")
        producer_version = self._safe_segment(producer_version, "producer_version")
        digest = stable_digest(key)
        return self.root / namespace / producer_version / digest[:2] / f"{digest}.json"

    def get(
        self,
        namespace: str,
        producer_version: str,
        key: dict[str, Any],
    ) -> Any | None:
        if not self.policy.can_read:
            return None
        path = self._path(namespace, producer_version, key)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            with self._lock:
                self._misses += 1
            return None
        if (
            document.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or document.get("namespace") != namespace
            or document.get("producer_version") != producer_version
            or document.get("key") != key
        ):
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
            self._hits += 1
        return document.get("value")

    def set(
        self,
        namespace: str,
        producer_version: str,
        key: dict[str, Any],
        value: Any,
    ) -> None:
        if not self.policy.can_write:
            return
        path = self._path(namespace, producer_version, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "producer_version": producer_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "value": value,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(
            json.dumps(document, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        with self._lock:
            self._writes += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "root": str(self.root),
                "policy": self.policy.value,
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
            }
