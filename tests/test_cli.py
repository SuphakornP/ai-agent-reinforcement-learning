from agent_rl_demos.cli import main


def test_cli_list(capsys) -> None:
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "02_rag_pinecone" in output


def test_cli_run_topic(capsys) -> None:
    assert main(["run", "05_rlvr_verifier_rewards", "--mock"]) == 0
    output = capsys.readouterr().out
    assert "RLVR verifier" in output


def test_cli_harness_run_without_report(capsys) -> None:
    assert main(["harness", "run", "05_rlvr_verifier_rewards", "--mock", "--no-report"]) == 0
    output = capsys.readouterr().out
    assert '"run_id": "harness-' in output
    assert "reward_coverage" in output
