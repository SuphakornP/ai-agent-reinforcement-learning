import pytest

from agent_rl_demos.config import load_settings
from agent_rl_demos.topic_impls import run_prompting_tool_calls, run_rag_pinecone


SETTINGS = load_settings()
LIVE_KEYS_PRESENT = (
    SETTINGS.has_openai_key and SETTINGS.has_pinecone_key and SETTINGS.has_langsmith_key
)


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_KEYS_PRESENT,
    reason="Requires OPENAI_API_KEY, PINECONE_API_KEY, and LANGSMITH_API_KEY.",
)
def test_live_openai_and_pinecone_smoke() -> None:
    tool_result = run_prompting_tool_calls(SETTINGS, "live")
    rag_result = run_rag_pinecone(SETTINGS, "live")

    assert tool_result.records
    assert rag_result.records
    assert all(row["matches"] for row in rag_result.records)
    assert rag_result.metrics["pinecone_embed_model"] == "multilingual-e5-large"
