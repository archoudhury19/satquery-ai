from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class RemoteSensingVLM:
    """
    Unified interface for SatQuery's remote-sensing VLM.

    Supported backends:
        local  -> model running in the local process
        remote -> model running on a cloud GPU/API
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 90,
    ) -> None:
        self.backend = (
            backend or os.getenv("RS_VLM_BACKEND", "remote")
        ).lower()

        self.url = url or os.getenv("RS_VLM_URL")
        self.api_key = api_key or os.getenv("RS_VLM_API_KEY")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        if self.backend == "remote":
            return bool(self.url)

        if self.backend == "local":
            return True

        return False

    def analyze(
        self,
        image_path: str | Path,
        question: str,
    ) -> Dict[str, Any]:

        if self.backend == "remote":
            return self._analyze_remote(
                image_path,
                question,
            )

        if self.backend == "local":
            return self._analyze_local(
                image_path,
                question,
            )

        raise ValueError(
            f"Unsupported RS_VLM_BACKEND: {self.backend}"
        )

    def _analyze_remote(
        self,
        image_path: str | Path,
        question: str,
    ) -> Dict[str, Any]:

        if not self.url:
            raise RuntimeError(
                "RS_VLM_URL is not configured."
            )

        image_path = Path(image_path)

        headers: Dict[str, str] = {}

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        with image_path.open("rb") as image_file:
            response = requests.post(
                self.url,
                headers=headers,
                files={
                    "image": (
                        image_path.name,
                        image_file,
                        "application/octet-stream",
                    )
                },
                data={
                    "question": question,
                },
                timeout=self.timeout,
            )

        response.raise_for_status()

        result = response.json()

        return {
            "answer": result.get("answer", ""),
            "confidence": float(
                result.get("confidence", 0.0)
            ),
            "model": result.get(
                "model",
                "remote-sensing-vlm",
            ),
            "raw": result,
        }

    def _analyze_local(
        self,
        image_path: str | Path,
        question: str,
    ) -> Dict[str, Any]:

        raise NotImplementedError(
            "Local RS-VLM has not been connected yet."
        )