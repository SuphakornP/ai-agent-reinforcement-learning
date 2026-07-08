from agent_rl_demos.live_openai import build_tool_policy_messages


def test_tool_policy_messages_include_strict_argument_contracts() -> None:
    messages = build_tool_policy_messages(
        "Create a calendar event for Alex next Tuesday at 2 PM.",
        ["calendar.create_event", "support.create_ticket"],
    )

    system_message = messages[0][1]
    assert "exactly the required keys" in system_message
    assert "no extra keys" in system_message
    assert "calendar.create_event" in system_message
    assert "attendee, day, time" in system_message
    assert "24-hour HH:MM" in system_message
    assert "support.create_ticket" in system_message
    assert "account, priority, issue" in system_message
