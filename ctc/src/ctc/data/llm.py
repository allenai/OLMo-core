"""
A stdlib-only chat client for the one generator that needs a model to build its data.

Contradiction's gold pairs are ``(real PubMed sentence, minimally-perturbed sentence that
contradicts it)``, and only a language model can write the second half. Everything else in this
package is model-free.

Written against ``urllib`` rather than the ``openai`` SDK on purpose, and the pre-migration tree
gives the reason in its own comments: the training conda env has no ``openai``, so the import was
deferred inside a function in eight files, and a second client was reimplemented from stdlib to
dodge the same problem. A ~100-line client with no dependency removes the reason for both, and
keeps ``pip install ./ctc`` honest.

Two behaviours are carried over deliberately:

* **replica round-robin** over comma-separated base URLs, so a local vLLM fleet is addressed
  without a load balancer;
* **an on-disk response cache**, because a perturbation run is expensive and interrupting it half
  way through must not mean paying for the first half again.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["ChatClient"]


class ChatClient:
    """Minimal OpenAI-compatible chat client: many prompts in, one response each, in order."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: Optional[str] = None,
        max_concurrent: int = 16,
        cache_dir: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        """
        :param base_url: One or more comma-separated OpenAI-compatible endpoints, e.g.
            ``"http://127.0.0.1:8765/v1"``. Requests round-robin across them.
        :param model: Model id to request.
        :param api_key: Bearer token; falls back to ``$OPENAI_API_KEY``, and a local vLLM server
            needs neither.
        :param max_concurrent: In-flight requests.
        :param cache_dir: Response cache location. ``None`` disables caching, which is only right
            for a smoke test.
        :param timeout: Per-request timeout in seconds.
        """
        self.bases = [u.strip().rstrip("/") for u in base_url.split(",") if u.strip()]
        if not self.bases:
            raise ValueError("base_url must name at least one endpoint")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_concurrent = max_concurrent
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self._counter = 0
        self._lock = threading.Lock()

    def _next_base(self) -> str:
        with self._lock:
            base = self.bases[self._counter % len(self.bases)]
            self._counter += 1
        return base

    def _cache_path(self, prompt: str, temperature: float, max_tokens: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(
            f"{self.model}\0{temperature}\0{max_tokens}\0{prompt}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def _one(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        path = self._cache_path(prompt, temperature, max_tokens)
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get("response")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self._next_base()}/chat/completions", data=payload, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
            # A failure is a missing perturbation, not a crashed build: the caller drops that pair
            # and reports how many were lost, which is more useful than losing the whole run.
            return None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"response": text}), encoding="utf-8")
        return text

    def run(
        self, prompts: Sequence[str], *, temperature: float = 0.7, max_tokens: int = 200
    ) -> List[Optional[str]]:
        """
        :param prompts: Prompts to send.
        :param temperature: Sampling temperature.
        :param max_tokens: Output cap.

        :returns: One response per prompt **in input order**, ``None`` where the call failed.
        """
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            return list(pool.map(lambda p: self._one(p, temperature, max_tokens), prompts))

    def describe(self) -> Dict[str, object]:
        """:returns: The settings that shaped a run, for a build's provenance record."""
        return {
            "model": self.model,
            "endpoints": len(self.bases),
            "cached": self.cache_dir is not None,
        }
