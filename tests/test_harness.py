from pathlib import Path

from agent_rl_demos.config import load_settings
from agent_rl_demos.harness import run_harness


def test_harness_run_all_mock_without_writing_report() -> None:
    settings = load_settings()
    report = run_harness(settings, "mock", write_report=False)

    assert report.topic_count == 10
    assert report.status in {"pass", "warn"}
    assert report.summary["topic_status_counts"]["fail"] == 0


def test_harness_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    settings = load_settings()
    report = run_harness(
        settings,
        "mock",
        topic_id="05_rlvr_verifier_rewards",
        report_dir=tmp_path,
        write_report=True,
    )

    assert Path(report.artifacts["json"]).exists()
    assert Path(report.artifacts["markdown"]).exists()
    assert "reward_coverage" in Path(report.artifacts["markdown"]).read_text(encoding="utf-8")
