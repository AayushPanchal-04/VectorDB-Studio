"""
ollama_client.py
=================
Thin HTTP client around the local Ollama REST API.
Install Ollama from https://ollama.com then run:
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""

from typing import List
import requests


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        """Returns [] if Ollama is not running or the model isn't pulled."""
        try:
            r = requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", []) or []
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        """Returns an error string if Ollama is unavailable."""
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
