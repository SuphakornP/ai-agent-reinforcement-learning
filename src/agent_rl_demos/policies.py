from __future__ import annotations

import json
from typing import Any


def deterministic_tool_policy(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": task["expected_tool"],
        "args": dict(task["expected_args"]),
    }


def weak_tool_policy(task: dict[str, Any]) -> str:
    tool = task["expected_tool"]
    args = dict(task["expected_args"])
    if "time" in args:
        args["time"] = "2 PM"
    if tool == "calendar.create_event":
        tool = "calendar.search"
    return json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


def rollout_candidates(task: dict[str, Any]) -> list[str]:
    correct = deterministic_tool_policy(task)
    wrong_tool = {"tool": "calendar.search", "args": dict(task["expected_args"])}
    wrong_args = {"tool": task["expected_tool"], "args": {"note": task["prompt"]}}
    unsafe = {"tool": "calendar.delete_all", "args": {"confirm": True}}
    malformed = f"Use {task['expected_tool']} with {task['expected_args']}"
    return [
        json.dumps(correct, ensure_ascii=False),
        json.dumps(wrong_tool, ensure_ascii=False),
        json.dumps(wrong_args, ensure_ascii=False),
        json.dumps(unsafe, ensure_ascii=False),
        malformed,
    ]
