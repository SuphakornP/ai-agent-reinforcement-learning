from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_rl_demos.config import Settings
from agent_rl_demos.models import DemoResult
from agent_rl_demos.topic_impls import (
    Mode,
    run_continuous_improvement_loop,
    run_decision_matrix,
    run_dpo_preference_pairs,
    run_grpo_rollout_simulation,
    run_langgraph_agent_environment,
    run_metrics_failure_inspection,
    run_prompting_tool_calls,
    run_rag_pinecone,
    run_rlvr_verifier_rewards,
    run_sft_data_prep,
)


TopicRunner = Callable[[Settings, Mode], DemoResult]


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    runner: TopicRunner


TOPICS: tuple[Topic, ...] = (
    Topic("00_decision_matrix", "Problem-first technique decision matrix", run_decision_matrix),
    Topic("01_prompting_tool_calls", "Prompting and JSON tool calls", run_prompting_tool_calls),
    Topic("02_rag_pinecone", "RAG with Pinecone integrated inference", run_rag_pinecone),
    Topic("03_sft_data_prep", "SFT data preparation", run_sft_data_prep),
    Topic("04_dpo_preference_pairs", "DPO preference pairs", run_dpo_preference_pairs),
    Topic("05_rlvr_verifier_rewards", "RLVR verifier rewards", run_rlvr_verifier_rewards),
    Topic("06_langgraph_agent_environment", "LangGraph agent environment", run_langgraph_agent_environment),
    Topic("07_grpo_rollout_simulation", "GRPO-style rollout simulation", run_grpo_rollout_simulation),
    Topic("08_metrics_failure_inspection", "Metrics and failure inspection", run_metrics_failure_inspection),
    Topic("09_continuous_improvement_loop", "Continuous improvement flywheel", run_continuous_improvement_loop),
)


def get_topic(topic_id: str) -> Topic:
    for topic in TOPICS:
        if topic.id == topic_id:
            return topic
    valid = ", ".join(topic.id for topic in TOPICS)
    raise KeyError(f"Unknown topic '{topic_id}'. Valid topics: {valid}")
