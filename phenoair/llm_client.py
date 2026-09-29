#!/usr/bin/env python3
"""Unified LLM client backed by litellm.

Supports Azure OpenAI and Anthropic Claude through a single interface.

Usage
-----
# Azure
client = LLMClient.azure(
    deployment="gpt-5.1-chat",
    endpoint="https://YOUR-RESOURCE.openai.azure.com/",
    api_key="...",
)

# Claude
client = LLMClient.claude(
    model="claude-opus-4-5",
    api_key="...",
)

# Call
text = client.chat(messages, json_mode=True, max_tokens=400)
response = client.chat_raw(messages)          # full litellm ModelResponse
response_id = response.id
"""
from __future__ import annotations

import time
import threading
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import litellm

litellm.suppress_debug_info = True

AZURE_API_VERSION_DEFAULT = "2024-12-01-preview"


@dataclass
class LLMClient:
    """Unified LLM client backed by litellm."""

    model: str
    """litellm model string, e.g. 'azure/gpt-5.1-chat' or 'anthropic/claude-opus-4-5'."""

    api_key: str

    api_base: str | None = None
    """Required for Azure: the endpoint URL, e.g. 'https://YOUR-RESOURCE.openai.azure.com'."""

    api_version: str | None = None
    """Required for Azure: API version string."""

    default_max_tokens: int = 1024
    """Fallback token budget when max_tokens is not passed to chat()."""

    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs forwarded to every litellm.completion() call."""

    _usage_records: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _usage_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _usage_local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @staticmethod
    def _split_init_kwargs(kwargs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Separate constructor fields from provider-specific call kwargs."""
        kwargs = dict(kwargs)
        default_max_tokens = kwargs.pop("default_max_tokens", 1024)
        extra_kwargs = dict(kwargs.pop("extra_kwargs", {}))
        extra_kwargs.update(kwargs)
        return default_max_tokens, extra_kwargs

    @classmethod
    def azure(
        cls,
        deployment: str,
        endpoint: str,
        api_key: str,
        api_version: str = AZURE_API_VERSION_DEFAULT,
        **kwargs: Any,
    ) -> "LLMClient":
        """Create a client for an Azure OpenAI deployment."""
        default_max_tokens, extra_kwargs = cls._split_init_kwargs(kwargs)
        return cls(
            model=f"azure/{deployment}",
            api_key=api_key,
            api_base=endpoint.rstrip("/"),
            api_version=api_version,
            default_max_tokens=default_max_tokens,
            extra_kwargs=extra_kwargs,
        )

    @classmethod
    def claude(
        cls,
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> "LLMClient":
        """Create a client for an Anthropic Claude model.

        Parameters
        ----------
        model:
            Short model name, e.g. 'claude-opus-4-5', 'claude-sonnet-4-6'.
            The 'anthropic/' prefix is added automatically if missing.
        """
        if not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        default_max_tokens, extra_kwargs = cls._split_init_kwargs(kwargs)
        return cls(
            model=model,
            api_key=api_key,
            default_max_tokens=default_max_tokens,
            extra_kwargs=extra_kwargs,
        )

    # ------------------------------------------------------------------
    # Core call methods
    # ------------------------------------------------------------------

    def _call_kwargs(
        self,
        max_tokens: int | None,
        json_mode: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self.extra_kwargs)
        kwargs.update(extra)
        if self.model.startswith("anthropic/"):
            # Some LiteLLM builds lag Anthropic alias metadata and reject these
            # params during preflight validation even though Anthropic accepts them.
            allowed_openai_params = list(kwargs.get("allowed_openai_params", []))
            for param in ("thinking", "reasoning_effort"):
                if param in kwargs and param not in allowed_openai_params:
                    allowed_openai_params.append(param)
            if allowed_openai_params:
                kwargs["allowed_openai_params"] = allowed_openai_params
        return kwargs

    def chat_raw(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ):
        """Call the model and return the raw litellm ModelResponse.

        Use this when you need access to response.id or other metadata.
        """
        call_kwargs = self._call_kwargs(max_tokens, json_mode, kwargs)
        started = time.perf_counter()
        record: dict[str, Any] = {
            "call_id": str(uuid.uuid4()),
            "model": self.model,
            "api_base": self.api_base,
            "message_count": len(messages),
            "max_tokens": call_kwargs.get("max_tokens"),
            "json_mode": json_mode,
            "tool_count": len(call_kwargs.get("tools") or []),
            "tool_choice": call_kwargs.get("tool_choice"),
            "success": False,
        }
        try:
            response = litellm.completion(messages=messages, **call_kwargs)
            record.update(
                {
                    "success": True,
                    "response_id": getattr(response, "id", None),
                    "usage": self._jsonable(getattr(response, "usage", None)),
                }
            )
            return response
        except Exception as exc:
            record.update(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            raise
        finally:
            record["duration_seconds"] = time.perf_counter() - started
            self._record_usage(record)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """Call the model and return the response text."""
        response = self.chat_raw(messages=messages, max_tokens=max_tokens, json_mode=json_mode, **kwargs)
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Batch / parallel call methods
    # ------------------------------------------------------------------

    def batch_chat_raw(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        max_workers: int = 8,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> list:
        """Call the model on multiple message lists concurrently.

        Analogous to DataLoader num_workers — fires up to `max_workers`
        requests in parallel using a thread pool.

        Returns a list of the same length as `messages_list`, in order.
        Each element is either a litellm ModelResponse or an Exception
        (never raises; let the caller inspect per-item errors).

        Example
        -------
        responses = llm.batch_chat_raw(all_messages, max_workers=16)
        for resp in responses:
            if isinstance(resp, Exception):
                print("error:", resp)
            else:
                print(resp.choices[0].message.content)
        """
        n = len(messages_list)
        results: list[Any] = [None] * n

        def _call(idx: int, msgs: list[dict[str, str]]):
            try:
                return idx, self.chat_raw(msgs, max_tokens=max_tokens, json_mode=json_mode, **kwargs)
            except Exception as exc:
                return idx, exc

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_call, i, msgs) for i, msgs in enumerate(messages_list)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def batch_chat(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        max_workers: int = 8,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> list[str | Exception]:
        """Like batch_chat_raw but returns response texts (str) or Exceptions."""
        raw_results = self.batch_chat_raw(
            messages_list,
            max_workers=max_workers,
            max_tokens=max_tokens,
            json_mode=json_mode,
            **kwargs,
        )
        out = []
        for r in raw_results:
            if isinstance(r, Exception):
                out.append(r)
            else:
                out.append(r.choices[0].message.content or "")
        return out

    def __repr__(self) -> str:
        base = f"LLMClient(model={self.model!r}"
        if self.api_base:
            base += f", api_base={self.api_base!r}"
        return base + ")"

    # ------------------------------------------------------------------
    # Usage / timing accounting
    # ------------------------------------------------------------------

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Convert LiteLLM/Pydantic usage objects into plain JSON data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): LLMClient._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LLMClient._jsonable(v) for v in value]
        if hasattr(value, "model_dump"):
            return LLMClient._jsonable(value.model_dump())
        if hasattr(value, "dict"):
            return LLMClient._jsonable(value.dict())
        if hasattr(value, "__dict__"):
            return {
                str(k): LLMClient._jsonable(v)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        return str(value)

    @staticmethod
    def _usage_value(usage: dict[str, Any] | None, *names: str) -> int:
        usage = usage or {}
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    @staticmethod
    def _nested_usage_value(usage: dict[str, Any] | None, container: str, name: str) -> int:
        usage = usage or {}
        nested = usage.get(container) or {}
        if not isinstance(nested, dict):
            return 0
        value = nested.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    @classmethod
    def summarize_usage_records(cls, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize LiteLLM usage records emitted by this client."""
        successful = [r for r in records if r.get("success")]
        failed = [r for r in records if not r.get("success")]
        prompt_tokens = sum(cls._usage_value(r.get("usage"), "prompt_tokens", "input_tokens") for r in records)
        completion_tokens = sum(cls._usage_value(r.get("usage"), "completion_tokens", "output_tokens") for r in records)
        total_tokens = sum(cls._usage_value(r.get("usage"), "total_tokens") for r in records)
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        reasoning_tokens = sum(
            cls._nested_usage_value(r.get("usage"), "completion_tokens_details", "reasoning_tokens")
            + cls._nested_usage_value(r.get("usage"), "output_tokens_details", "reasoning_tokens")
            for r in records
        )
        durations = [float(r.get("duration_seconds", 0.0) or 0.0) for r in records]
        return {
            "request_count": len(records),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "sum_call_seconds": round(sum(durations), 3),
            "avg_call_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "max_call_seconds": round(max(durations), 3) if durations else 0.0,
            "models": sorted({str(r.get("model")) for r in records if r.get("model")}),
        }

    def _record_usage(self, record: dict[str, Any]) -> None:
        plain = self._jsonable(record)
        with self._usage_lock:
            self._usage_records.append(plain)
        thread_records = getattr(self._usage_local, "records", None)
        if thread_records is not None:
            thread_records.append(plain)

    def reset_usage(self) -> None:
        """Clear process-wide usage records for this client instance."""
        with self._usage_lock:
            self._usage_records.clear()

    def usage_records(self) -> list[dict[str, Any]]:
        """Return a snapshot of process-wide usage records for this client."""
        with self._usage_lock:
            return list(self._usage_records)

    def usage_summary(self) -> dict[str, Any]:
        """Summarize process-wide usage records for this client."""
        return self.summarize_usage_records(self.usage_records())

    def reset_thread_usage(self, metadata: dict[str, Any] | None = None) -> None:
        """Start per-thread usage collection, useful inside ThreadPool workers."""
        self._usage_local.records = []
        self._usage_local.metadata = dict(metadata or {})

    def thread_usage_records(self) -> list[dict[str, Any]]:
        """Return records collected on the current thread since reset_thread_usage()."""
        return list(getattr(self._usage_local, "records", []) or [])

    def thread_usage_summary(self, *, reset: bool = False) -> dict[str, Any]:
        """Summarize current-thread usage and optionally clear it."""
        records = self.thread_usage_records()
        summary = self.summarize_usage_records(records)
        metadata = getattr(self._usage_local, "metadata", None)
        if metadata:
            summary["metadata"] = dict(metadata)
        if reset:
            self.reset_thread_usage(metadata=None)
        return summary

    @contextmanager
    def usage_scope(self, metadata: dict[str, Any] | None = None):
        """Context manager wrapper around per-thread usage collection."""
        previous_records = getattr(self._usage_local, "records", None)
        previous_metadata = getattr(self._usage_local, "metadata", None)
        self.reset_thread_usage(metadata=metadata)
        try:
            yield self
        finally:
            self._usage_local.records = previous_records
            self._usage_local.metadata = previous_metadata
