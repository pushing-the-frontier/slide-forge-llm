"""SlideForge client for interacting with the environment server."""

from __future__ import annotations

import httpx

from .models import SlideForgeAction, SlideForgeObservation


class SlideForgeClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=60)

    def health(self) -> dict:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(self, **brief_kwargs) -> dict:
        resp = self._client.post("/reset", json=brief_kwargs)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: SlideForgeAction) -> dict:
        resp = self._client.post("/step", json={
            "tool": action.tool,
            "parameters": action.parameters,
        })
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
