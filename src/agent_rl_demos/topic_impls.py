from __future__ import annotations

import statistics
from typing import Any, Literal, TypedDict

from agent_rl_demos.config import Settings
from agent_rl_demos.io import load_topic_json, write_jsonl_preview
from agent_rl_demos.models import DemoResult
from agent_rl_demos.pinecone_rag import MockIntegratedPineconeRetriever, PineconeIntegratedRetriever
from agent_rl_demos.policies import deterministic_tool_policy, rollout_candidates
from agent_rl_demos.rewards import bucket_failures, score_tool_call


Mode = Literal["mock", "live"]


def run_decision_matrix(settings: Settings, mode: Mode) -> DemoResult:
    cases = load_topic_json("00_decision_matrix", "cases.json")
    records = []
    for case in cases:
        signals = case["signals"]
        if signals.get("missing_domain_facts"):
            technique = "RAG or data injection"
        elif signals.get("format_errors"):
            technique = "Prompting, then SFT"
        elif signals.get("demonstrations"):
            technique = "SFT"
        elif signals.get("preference_pairs"):
            technique = "DPO"
        elif signals.get("verifiable_success"):
            technique = "RLVR with verifier rewards"
        else:
            technique = "Environment-based RL"
        records.append({"case": case["id"], "problem": case["problem"], "recommended": technique})
    return DemoResult(
        topic="00_decision_matrix",
        title="Problem-first technique decision matrix",
        summary=f"Mapped {len(records)} failure patterns to the lightest useful technique.",
        metrics={"cases": len(records), "mode": mode},
        records=records,
    )


def run_prompting_tool_calls(settings: Settings, mode: Mode) -> DemoResult:
    tasks = load_topic_json("01_prompting_tool_calls", "tool_tasks.json")
    tools = sorted({task["expected_tool"] for task in tasks})
    records = []
    for task in tasks:
        if mode == "live":
            from agent_rl_demos.live_openai import call_openai_tool_policy

            output = call_openai_tool_policy(settings, task["prompt"], tools)
        else:
            output = deterministic_tool_policy(task)
        reward = score_tool_call(output, task).as_dict()
        records.append({"task_id": task["id"], "output": output, **reward})
    success_rate = _success_rate(records)
    return DemoResult(
        topic="01_prompting_tool_calls",
        title="Prompting and JSON tool-call validation",
        summary="Generated tool calls and verified format, tool choice, arguments, and safety.",
        metrics={"tasks": len(tasks), "success_rate": success_rate, "mode": mode},
        records=records,
    )


def run_rag_pinecone(settings: Settings, mode: Mode) -> DemoResult:
    records = load_topic_json("02_rag_pinecone", "records.json")
    queries = load_topic_json("02_rag_pinecone", "queries.json")
    if mode == "live":
        retriever = PineconeIntegratedRetriever(settings)
        retriever.upsert_records(records)
    else:
        retriever = MockIntegratedPineconeRetriever(records)

    results = []
    for query in queries:
        matches = [match.as_dict() for match in retriever.search(query["query"], top_k=3)]
        results.append({"query_id": query["id"], "query": query["query"], "matches": matches})
    return DemoResult(
        topic="02_rag_pinecone",
        title="RAG with Pinecone integrated multilingual embedding",
        summary="Searched Thai and English mock records using Pinecone-style integrated inference.",
        metrics={
            "queries": len(queries),
            "records": len(records),
            "pinecone_embed_model": settings.pinecone_embed_model,
            "mode": mode,
        },
        records=results,
    )


def run_sft_data_prep(settings: Settings, mode: Mode) -> DemoResult:
    traces = load_topic_json("03_sft_data_prep", "teacher_traces.json")
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Return one safe JSON tool call."},
                {"role": "user", "content": trace["prompt"]},
                {"role": "assistant", "content": trace["accepted_output"]},
            ]
        }
        for trace in traces
        if trace["accepted"]
    ]
    split_at = max(1, round(len(rows) * 0.75))
    return DemoResult(
        topic="03_sft_data_prep",
        title="SFT demonstration data preparation",
        summary="Filtered accepted traces into train/eval chat-message demonstrations.",
        metrics={"train_rows": split_at, "eval_rows": len(rows) - split_at, "mode": mode},
        artifacts={"jsonl_preview": write_jsonl_preview(rows)},
    )


def run_dpo_preference_pairs(settings: Settings, mode: Mode) -> DemoResult:
    pairs = load_topic_json("04_dpo_preference_pairs", "preference_pairs.json")
    records = []
    for pair in pairs:
        chosen = score_tool_call(pair["chosen"], pair["expected"]).score
        rejected = score_tool_call(pair["rejected"], pair["expected"]).score
        records.append(
            {
                "pair_id": pair["id"],
                "chosen_reward": chosen,
                "rejected_reward": rejected,
                "margin": round(chosen - rejected, 4),
            }
        )
    return DemoResult(
        topic="04_dpo_preference_pairs",
        title="DPO preference pair inspection",
        summary="Compared chosen and rejected outputs against a deterministic verifier.",
        metrics={"pairs": len(pairs), "positive_margins": sum(1 for row in records if row["margin"] > 0)},
        records=records,
    )


def run_rlvr_verifier_rewards(settings: Settings, mode: Mode) -> DemoResult:
    cases = load_topic_json("05_rlvr_verifier_rewards", "outputs.json")
    records = []
    for case in cases:
        reward = score_tool_call(case["output"], case["expected"]).as_dict()
        records.append({"case_id": case["id"], **reward})
    return DemoResult(
        topic="05_rlvr_verifier_rewards",
        title="RLVR verifier and reward scoring",
        summary="Scored outputs with deterministic JSON/tool/argument/safety rewards.",
        metrics={"cases": len(cases), "success_rate": _success_rate(records)},
        records=records,
        artifacts={"failure_buckets": bucket_failures(records)},
    )


class AgentState(TypedDict, total=False):
    task: dict[str, Any]
    output: dict[str, Any]
    reward: dict[str, Any]
    done: bool


def run_langgraph_agent_environment(settings: Settings, mode: Mode) -> DemoResult:
    from langgraph.graph import END, START, StateGraph

    tasks = load_topic_json("06_langgraph_agent_environment", "environment_tasks.json")

    def policy_node(state: AgentState) -> AgentState:
        return {"output": deterministic_tool_policy(state["task"])}

    def verifier_node(state: AgentState) -> AgentState:
        return {"reward": score_tool_call(state["output"], state["task"]).as_dict()}

    def route_after_verify(state: AgentState) -> str:
        return END if state["reward"]["score"] >= 1.0 else "policy"

    graph = StateGraph(AgentState)
    graph.add_node("policy", policy_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "policy")
    graph.add_edge("policy", "verifier")
    graph.add_conditional_edges("verifier", route_after_verify, ["policy", END])
    app = graph.compile()

    records = []
    for task in tasks:
        result = app.invoke({"task": task}, {"recursion_limit": 4})
        records.append(
            {
                "task_id": task["id"],
                "tool": result["output"]["tool"],
                "reward": result["reward"]["score"],
                "failure_type": result["reward"]["failure_type"],
            }
        )
    return DemoResult(
        topic="06_langgraph_agent_environment",
        title="LangGraph agent environment",
        summary="Ran a stateful policy -> verifier graph with conditional routing.",
        metrics={"tasks": len(tasks), "success_rate": _success_rate(records, reward_key="reward")},
        records=records,
    )


def run_grpo_rollout_simulation(settings: Settings, mode: Mode) -> DemoResult:
    tasks = load_topic_json("07_grpo_rollout_simulation", "rollout_tasks.json")
    records = []
    for task in tasks:
        scored = []
        for output in rollout_candidates(task):
            reward = score_tool_call(output, task)
            scored.append({"output": output, "reward": reward.score, "failure_type": reward.failure_type})
        mean_reward = statistics.mean(row["reward"] for row in scored)
        for index, row in enumerate(scored):
            records.append(
                {
                    "task_id": task["id"],
                    "candidate": index,
                    "reward": row["reward"],
                    "advantage": round(row["reward"] - mean_reward, 4),
                    "failure_type": row["failure_type"],
                }
            )
    return DemoResult(
        topic="07_grpo_rollout_simulation",
        title="GRPO-style rollout simulation",
        summary="Generated grouped candidates, scored them, and computed relative advantages.",
        metrics={
            "tasks": len(tasks),
            "rollouts": len(records),
            "best_reward": max(row["reward"] for row in records),
        },
        records=records,
    )


def run_metrics_failure_inspection(settings: Settings, mode: Mode) -> DemoResult:
    cases = load_topic_json("08_metrics_failure_inspection", "eval_outputs.json")
    records = []
    for case in cases:
        reward = score_tool_call(case["output"], case["expected"]).as_dict()
        records.append({"case_id": case["id"], "latency_ms": case["latency_ms"], **reward})
    failure_buckets = bucket_failures(records)
    return DemoResult(
        topic="08_metrics_failure_inspection",
        title="Metrics and failure inspection",
        summary="Tracked validation reward, invalid outputs, unsafe actions, latency, and buckets.",
        metrics={
            "success_rate": _success_rate(records),
            "avg_latency_ms": round(statistics.mean(row["latency_ms"] for row in records), 2),
            "failure_buckets": failure_buckets,
        },
        records=records,
    )


def run_continuous_improvement_loop(settings: Settings, mode: Mode) -> DemoResult:
    failures = load_topic_json("09_continuous_improvement_loop", "production_failures.json")
    records = []
    for failure in failures:
        if failure["failure_type"] in {"format_error", "schema_error"}:
            fix = "Prompt/schema clarification, then SFT if repeated"
        elif failure["failure_type"] == "retrieval_failure":
            fix = "RAG index or chunk metadata fix"
        elif failure["failure_type"] in {"wrong_tool", "wrong_arguments"}:
            fix = "Verifier-backed RLVR eval"
        else:
            fix = "Safety policy and environment guardrail"
        records.append(
            {
                "failure_id": failure["id"],
                "new_eval_id": f"eval_{failure['id']}",
                "failure_type": failure["failure_type"],
                "lightest_fix": fix,
            }
        )
    return DemoResult(
        topic="09_continuous_improvement_loop",
        title="Continuous improvement flywheel",
        summary="Converted production failures into regression evals and next training signals.",
        metrics={"failures": len(failures), "new_evals": len(records)},
        records=records,
    )


def _success_rate(records: list[dict[str, Any]], reward_key: str = "score") -> float:
    if not records:
        return 0.0
    successes = sum(1 for row in records if row.get(reward_key, 0.0) >= 1.0)
    return round(successes / len(records), 4)
