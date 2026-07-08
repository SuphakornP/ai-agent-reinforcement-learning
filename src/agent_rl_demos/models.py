from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DemoResult:
    topic: str
    title: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "title": self.title,
            "summary": self.summary,
            "metrics": self.metrics,
            "records": self.records,
            "artifacts": self.artifacts,
        }

    def to_text(self) -> str:
        payload = self.to_dict()
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
