from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent_rl_demos.config import PROJECT_ROOT, Settings
from agent_rl_demos.models import DemoResult
from agent_rl_demos.registry import TOPICS, Topic, get_topic
from agent_rl_demos.topic_impls import Mode


CheckStatus = Literal["pass", "warn", "fail"]
OverallStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class HarnessCheck:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class TopicAssessment:
    topic: str
    title: str
    mode: Mode
    status: OverallStatus
    checks: list[HarnessCheck]
    metrics: dict[str, Any]
    failure_buckets: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "title": self.title,
            "mode": self.mode,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": self.metrics,
            "failure_buckets": self.failure_buckets,
        }


@dataclass(frozen=True)
class HarnessReport:
    run_id: str
    mode: Mode
    status: OverallStatus
    generated_at: str
    topic_count: int
    summary: dict[str, Any]
    assessments: list[TopicAssessment]
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "generated_at": self.generated_at,
            "topic_count": self.topic_count,
            "summary": self.summary,
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "artifacts": self.artifacts,
        }

    def to_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def run_harness(
    settings: Settings,
    mode: Mode,
    topic_id: str = "all",
    report_dir: Path | None = None,
    write_report: bool = True,
) -> HarnessReport:
    topics = _select_topics(topic_id)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_id = f"harness-{generated_at.replace(':', '').replace('-', '')}"

    assessments = [assess_topic(topic, topic.runner(settings, mode), mode) for topic in topics]
    summary = _summarize(assessments)
    report = HarnessReport(
        run_id=run_id,
        mode=mode,
        status=summary["overall_status"],
        generated_at=generated_at,
        topic_count=len(assessments),
        summary=summary,
        assessments=assessments,
    )

    if write_report:
        target_dir = report_dir or PROJECT_ROOT / "reports"
        artifacts = write_harness_report(report, target_dir)
        report = HarnessReport(
            run_id=report.run_id,
            mode=report.mode,
            status=report.status,
            generated_at=report.generated_at,
            topic_count=report.topic_count,
            summary=report.summary,
            assessments=report.assessments,
            artifacts=artifacts,
        )
    return report


def assess_topic(topic: Topic, result: DemoResult, mode: Mode) -> TopicAssessment:
    checks = [
        _check_metrics_present(result),
        _check_behavior_surface(result),
        *_topic_specific_checks(result, mode),
    ]
    status = _status_from_checks(checks)
    return TopicAssessment(
        topic=topic.id,
        title=topic.title,
        mode=mode,
        status=status,
        checks=checks,
        metrics=result.metrics,
        failure_buckets=_collect_failure_buckets(result),
    )


def write_harness_report(report: HarnessReport, report_dir: Path) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report.run_id}.json"
    md_path = report_dir / f"{report.run_id}.md"
    json_path.write_text(report.to_text() + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _select_topics(topic_id: str) -> list[Topic]:
    if topic_id == "all":
        return list(TOPICS)
    return [get_topic(topic_id)]


def _check_metrics_present(result: DemoResult) -> HarnessCheck:
    if result.metrics:
        return HarnessCheck(
            name="metrics_present",
            status="pass",
            message="Topic produced metrics.",
            details={"metric_keys": sorted(result.metrics)},
        )
    return HarnessCheck(
        name="metrics_present",
        status="fail",
        message="Topic returned no metrics, so harness cannot compare behavior.",
    )


def _check_behavior_surface(result: DemoResult) -> HarnessCheck:
    if result.records or result.artifacts:
        return HarnessCheck(
            name="behavior_surface_present",
            status="pass",
            message="Topic produced records or artifacts for inspection.",
            details={"records": len(result.records), "artifact_keys": sorted(result.artifacts)},
        )
    return HarnessCheck(
        name="behavior_surface_present",
        status="warn",
        message="Topic has metrics but no records/artifacts to inspect.",
    )


def _topic_specific_checks(result: DemoResult, mode: Mode) -> list[HarnessCheck]:
    if result.topic == "01_prompting_tool_calls":
        return [_check_tool_outputs_safe(result), _check_success_rate(result, warn_below=0.8)]
    if result.topic == "02_rag_pinecone":
        return [_check_rag_matches(result)]
    if result.topic == "05_rlvr_verifier_rewards":
        return [_check_reward_has_positive_and_negative_examples(result)]
    if result.topic == "06_langgraph_agent_environment":
        return [_check_success_rate(result, fail_below=1.0)]
    if result.topic == "07_grpo_rollout_simulation":
        return [_check_rollout_advantages(result)]
    if result.topic == "08_metrics_failure_inspection":
        return [_check_failure_buckets_present(result)]
    if result.topic == "09_continuous_improvement_loop":
        return [_check_improvement_loop_outputs_evals(result)]
    return []


def _check_tool_outputs_safe(result: DemoResult) -> HarnessCheck:
    unsafe = [row for row in result.records if row.get("safe") is False]
    if not unsafe:
        return HarnessCheck(
            name="tool_outputs_safe",
            status="pass",
            message="No unsafe tool outputs were produced.",
        )
    return HarnessCheck(
        name="tool_outputs_safe",
        status="fail",
        message="One or more tool outputs were unsafe.",
        details={"unsafe_count": len(unsafe)},
    )


def _check_success_rate(
    result: DemoResult,
    warn_below: float | None = None,
    fail_below: float | None = None,
) -> HarnessCheck:
    success_rate = float(result.metrics.get("success_rate", 0.0))
    if fail_below is not None and success_rate < fail_below:
        return HarnessCheck(
            name="success_rate",
            status="fail",
            message=f"Success rate {success_rate:.4f} is below required {fail_below:.4f}.",
            details={"success_rate": success_rate, "threshold": fail_below},
        )
    if warn_below is not None and success_rate < warn_below:
        return HarnessCheck(
            name="success_rate",
            status="warn",
            message=f"Success rate {success_rate:.4f} is below target {warn_below:.4f}.",
            details={"success_rate": success_rate, "threshold": warn_below},
        )
    return HarnessCheck(
        name="success_rate",
        status="pass",
        message=f"Success rate {success_rate:.4f} meets the harness threshold.",
        details={"success_rate": success_rate},
    )


def _check_rag_matches(result: DemoResult) -> HarnessCheck:
    empty_queries = [row["query_id"] for row in result.records if not row.get("matches")]
    if not empty_queries:
        return HarnessCheck(
            name="rag_matches_present",
            status="pass",
            message="Every RAG query returned at least one match.",
            details={"queries": len(result.records)},
        )
    return HarnessCheck(
        name="rag_matches_present",
        status="fail",
        message="Some RAG queries returned no matches.",
        details={"empty_queries": empty_queries},
    )


def _check_reward_has_positive_and_negative_examples(result: DemoResult) -> HarnessCheck:
    buckets = _collect_failure_buckets(result)
    has_success = buckets.get("success", 0) > 0
    has_failure = sum(count for name, count in buckets.items() if name != "success") > 0
    if has_success and has_failure:
        return HarnessCheck(
            name="reward_coverage",
            status="pass",
            message="Reward demo includes both passing and failing examples.",
            details={"failure_buckets": buckets},
        )
    return HarnessCheck(
        name="reward_coverage",
        status="fail",
        message="Reward demo needs both positive and negative examples.",
        details={"failure_buckets": buckets},
    )


def _check_rollout_advantages(result: DemoResult) -> HarnessCheck:
    missing = [row for row in result.records if "advantage" not in row]
    if result.records and not missing:
        return HarnessCheck(
            name="rollout_advantages",
            status="pass",
            message="Every rollout row includes a relative advantage.",
            details={"rollouts": len(result.records)},
        )
    return HarnessCheck(
        name="rollout_advantages",
        status="fail",
        message="Rollout simulation did not produce complete advantage rows.",
        details={"missing_count": len(missing)},
    )


def _check_failure_buckets_present(result: DemoResult) -> HarnessCheck:
    buckets = _collect_failure_buckets(result)
    if buckets:
        return HarnessCheck(
            name="failure_buckets_present",
            status="pass",
            message="Failure buckets are available for inspection.",
            details={"failure_buckets": buckets},
        )
    return HarnessCheck(
        name="failure_buckets_present",
        status="fail",
        message="Failure inspection topic did not expose failure buckets.",
    )


def _check_improvement_loop_outputs_evals(result: DemoResult) -> HarnessCheck:
    missing = [row for row in result.records if not row.get("new_eval_id")]
    if result.records and not missing:
        return HarnessCheck(
            name="failures_promoted_to_evals",
            status="pass",
            message="Every production failure was mapped to a new eval id.",
            details={"new_evals": len(result.records)},
        )
    return HarnessCheck(
        name="failures_promoted_to_evals",
        status="fail",
        message="Some production failures were not promoted to eval ids.",
        details={"missing_count": len(missing)},
    )


def _collect_failure_buckets(result: DemoResult) -> dict[str, int]:
    artifact_buckets = result.artifacts.get("failure_buckets")
    if isinstance(artifact_buckets, dict):
        return {str(key): int(value) for key, value in artifact_buckets.items()}
    metric_buckets = result.metrics.get("failure_buckets")
    if isinstance(metric_buckets, dict):
        return {str(key): int(value) for key, value in metric_buckets.items()}

    buckets: dict[str, int] = {}
    for row in result.records:
        if "failure_type" in row:
            failure_type = row.get("failure_type") or "success"
            buckets[failure_type] = buckets.get(failure_type, 0) + 1
    return buckets


def _status_from_checks(checks: list[HarnessCheck]) -> OverallStatus:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _summarize(assessments: list[TopicAssessment]) -> dict[str, Any]:
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    check_counts = {"pass": 0, "warn": 0, "fail": 0}
    for assessment in assessments:
        status_counts[assessment.status] += 1
        for check in assessment.checks:
            check_counts[check.status] += 1

    overall: OverallStatus = "pass"
    if status_counts["fail"]:
        overall = "fail"
    elif status_counts["warn"]:
        overall = "warn"

    return {
        "overall_status": overall,
        "topic_status_counts": status_counts,
        "check_status_counts": check_counts,
    }


def _render_markdown(report: HarnessReport) -> str:
    lines = [
        f"# Harness Report: {report.run_id}",
        "",
        f"- Mode: `{report.mode}`",
        f"- Status: `{report.status}`",
        f"- Generated: `{report.generated_at}`",
        f"- Topics: `{report.topic_count}`",
        "",
        "## Summary",
        "",
        "| Type | Pass | Warn | Fail |",
        "| --- | ---: | ---: | ---: |",
        (
            "| Topics | "
            f"{report.summary['topic_status_counts']['pass']} | "
            f"{report.summary['topic_status_counts']['warn']} | "
            f"{report.summary['topic_status_counts']['fail']} |"
        ),
        (
            "| Checks | "
            f"{report.summary['check_status_counts']['pass']} | "
            f"{report.summary['check_status_counts']['warn']} | "
            f"{report.summary['check_status_counts']['fail']} |"
        ),
        "",
        "## Topic Assessments",
        "",
    ]
    for assessment in report.assessments:
        lines.extend(
            [
                f"### `{assessment.topic}`",
                "",
                f"- Title: {assessment.title}",
                f"- Status: `{assessment.status}`",
                f"- Metrics: `{json.dumps(assessment.metrics, ensure_ascii=False, sort_keys=True)}`",
                "",
                "| Check | Status | Message |",
                "| --- | --- | --- |",
            ]
        )
        for check in assessment.checks:
            lines.append(f"| `{check.name}` | `{check.status}` | {check.message} |")
        lines.append("")
    return "\n".join(lines)
