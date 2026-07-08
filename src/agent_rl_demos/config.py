from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPICS_ROOT = PROJECT_ROOT / "topics"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    pinecone_api_key: str
    pinecone_embed_model: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    langsmith_api_key: str
    langsmith_tracing: str
    langsmith_project: str

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_pinecone_key(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def has_langsmith_key(self) -> bool:
        return bool(self.langsmith_api_key)


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "true")
    if not langsmith_api_key and langsmith_tracing.lower() == "true":
        langsmith_tracing = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_embed_model=os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "nvidia-agentic-rl-demo"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "agentic-rl"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        langsmith_api_key=langsmith_api_key,
        langsmith_tracing=langsmith_tracing,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "nvidia-agentic-rl-demo"),
    )


def require_live_key(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"{name} is required for --live mode. Add it to .env first.")
