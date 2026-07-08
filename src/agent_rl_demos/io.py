from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_rl_demos.config import TOPICS_ROOT


def load_topic_json(topic_id: str, filename: str) -> Any:
    path = TOPICS_ROOT / topic_id / "mock_data" / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_preview(rows: list[dict[str, Any]], limit: int = 3) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows[:limit]]


def topic_path(topic_id: str) -> Path:
    return TOPICS_ROOT / topic_id
