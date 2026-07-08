from agent_rl_demos.rewards import bucket_failures, score_tool_call


EXPECTED = {
    "expected_tool": "calendar.create_event",
    "expected_args": {"attendee": "Alex", "day": "Tuesday", "time": "14:00"},
}


def test_score_tool_call_success() -> None:
    reward = score_tool_call(
        '{"tool":"calendar.create_event","args":{"attendee":"Alex","day":"Tuesday","time":"14:00"}}',
        EXPECTED,
    )
    assert reward.score == 1.0
    assert reward.failure_type is None


def test_score_tool_call_rejects_malformed_json() -> None:
    reward = score_tool_call("create it", EXPECTED)
    assert reward.score == -1.0
    assert reward.failure_type == "format_error"


def test_bucket_failures_counts_success() -> None:
    buckets = bucket_failures([{"failure_type": None}, {"failure_type": "wrong_tool"}])
    assert buckets == {"success": 1, "wrong_tool": 1}
