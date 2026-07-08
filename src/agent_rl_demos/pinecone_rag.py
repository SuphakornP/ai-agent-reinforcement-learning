from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_rl_demos.config import Settings, require_live_key


@dataclass(frozen=True)
class RetrievalMatch:
    id: str
    score: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": round(self.score, 4), "metadata": self.metadata}


class MockIntegratedPineconeRetriever:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def search(self, query: str, top_k: int = 3) -> list[RetrievalMatch]:
        query_terms = set(_tokenize(query))
        matches: list[RetrievalMatch] = []
        for record in self.records:
            text_terms = set(_tokenize(record["text"]))
            overlap = len(query_terms & text_terms)
            score = overlap / max(len(query_terms), 1)
            if overlap == 0:
                score = _soft_topic_score(query, record)
            matches.append(
                RetrievalMatch(
                    id=record["id"],
                    score=score,
                    metadata={
                        "text": record["text"],
                        "language": record["language"],
                        "topic": record["topic"],
                        "source": record["source"],
                    },
                )
            )
        return sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]


class PineconeIntegratedRetriever:
    def __init__(self, settings: Settings) -> None:
        require_live_key("PINECONE_API_KEY", settings.pinecone_api_key)
        self.settings = settings

    def _index(self):
        from pinecone import Pinecone

        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        if not pc.has_index(self.settings.pinecone_index_name):
            pc.create_index_for_model(
                name=self.settings.pinecone_index_name,
                cloud=self.settings.pinecone_cloud,
                region=self.settings.pinecone_region,
                embed={
                    "model": self.settings.pinecone_embed_model,
                    "field_map": {"text": "text"},
                },
            )
        return pc.Index(self.settings.pinecone_index_name)

    def upsert_records(self, records: list[dict[str, Any]]) -> None:
        index = self._index()
        index.upsert_records(namespace=self.settings.pinecone_namespace, records=records)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalMatch]:
        index = self._index()
        results = index.search(
            namespace=self.settings.pinecone_namespace,
            query={"inputs": {"text": query}, "top_k": top_k},
        )
        matches = _extract_search_hits(results)
        return [_coerce_match(match) for match in matches]


def _extract_search_hits(results: Any) -> list[Any]:
    if not isinstance(results, dict) and hasattr(results, "matches"):
        return list(results.matches)
    if not isinstance(results, dict) and hasattr(results, "to_dict"):
        results = results.to_dict()
    if not isinstance(results, dict):
        raise TypeError(f"Unsupported Pinecone search response: {type(results)!r}")
    if "matches" in results:
        return list(results["matches"])
    result = results.get("result")
    if isinstance(result, dict) and "hits" in result:
        return list(result["hits"])
    return []


def _coerce_match(match: Any) -> RetrievalMatch:
    if isinstance(match, dict):
        return RetrievalMatch(
            id=str(match.get("id") or match.get("id_", "")),
            score=float(match.get("score") if "score" in match else match.get("score_", 0.0)),
            metadata=dict(match.get("metadata", {}) or match.get("fields", {})),
        )
    metadata = getattr(match, "metadata", None) or getattr(match, "fields", {}) or {}
    identifier = getattr(match, "id", None) or getattr(match, "id_", "")
    score = getattr(match, "score", None)
    if score is None:
        score = getattr(match, "score_", 0.0)
    return RetrievalMatch(id=str(identifier), score=float(score), metadata=dict(metadata))


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [part for part in cleaned.split() if part]


def _soft_topic_score(query: str, record: dict[str, Any]) -> float:
    query_lower = query.lower()
    topic = record["topic"].lower()
    if topic in query_lower:
        return 0.35
    if "กรุงเทพ" in query and record["language"] == "th":
        return 0.25
    if "reward" in query_lower and "reward" in record["text"].lower():
        return 0.25
    return 0.05
