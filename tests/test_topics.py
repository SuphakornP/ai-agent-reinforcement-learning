from agent_rl_demos.config import load_settings
from agent_rl_demos.registry import TOPICS


def test_all_topics_run_in_mock_mode() -> None:
    settings = load_settings()
    results = [topic.runner(settings, "mock") for topic in TOPICS]
    assert len(results) == 10
    assert all(result.metrics for result in results)


def test_langgraph_environment_mock_succeeds() -> None:
    settings = load_settings()
    topic = next(topic for topic in TOPICS if topic.id == "06_langgraph_agent_environment")
    result = topic.runner(settings, "mock")
    assert result.metrics["success_rate"] == 1.0
