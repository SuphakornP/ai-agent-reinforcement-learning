from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_rl_demos.config import Settings, require_live_key


class ToolCallSchema(BaseModel):
    tool: str = Field(description="The API or CLI tool name to call.")
    args: dict[str, Any] = Field(
        description="JSON object containing only the required arguments for the selected tool."
    )


TOOL_ARG_CONTRACTS = {
    "calendar.create_event": (
        "required args: attendee, day, time. "
        "Use only the weekday name for day, for example Tuesday. "
        "Use 24-hour HH:MM time, for example 14:00. "
        "Do not include title, subject, description, or any optional fields."
    ),
    "support.create_ticket": (
        "required args: account, priority, issue. "
        "Use the account acronym exactly as written. "
        "Use lowercase priority and a concise issue phrase, for example login failures. "
        "Do not include subject, description, body, or any optional fields."
    ),
    "project.update_task": (
        "required args: project, task_id, status. "
        "Use the project name, task id, and destination status exactly as requested. "
        "Do not include notes, title, description, or any optional fields."
    ),
}


def build_tool_policy_messages(prompt: str, tools: list[str]) -> list[tuple[str, str]]:
    allowed_tools = sorted(tools)
    contracts = "\n".join(
        f"- {tool}: {TOOL_ARG_CONTRACTS[tool]}"
        for tool in allowed_tools
        if tool in TOOL_ARG_CONTRACTS
    )
    system = (
        "Return exactly one structured tool call with shape {tool: string, args: object}. "
        "Choose only from the allowed tools. "
        "The args object must contain exactly the required keys for the selected tool and no extra keys. "
        "Normalize values to the contract instead of copying verbose natural language.\n\n"
        "Allowed tools:\n- "
        + "\n- ".join(allowed_tools)
        + "\n\nArgument contracts:\n"
        + contracts
    )
    return [("system", system), ("user", prompt)]


def call_openai_tool_policy(settings: Settings, prompt: str, tools: list[str]) -> dict[str, Any]:
    require_live_key("OPENAI_API_KEY", settings.openai_api_key)
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    structured = llm.with_structured_output(ToolCallSchema, method="function_calling")
    response = structured.invoke(build_tool_policy_messages(prompt, tools))
    return response.model_dump()
