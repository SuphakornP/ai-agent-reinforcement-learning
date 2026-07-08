from __future__ import annotations

import argparse
import sys

from langchain_core.tracers.langchain import wait_for_all_tracers

from agent_rl_demos.config import PROJECT_ROOT, load_settings
from agent_rl_demos.harness import run_harness
from agent_rl_demos.registry import TOPICS, get_topic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-rl-demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List demo topics.")

    run_parser = subparsers.add_parser("run", help="Run one topic or all topics.")
    run_parser.add_argument("topic", help="Topic id or 'all'.")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Run deterministic local mode.")
    mode.add_argument("--live", action="store_true", help="Run live external integrations.")

    harness_parser = subparsers.add_parser("harness", help="Run harness engineering checks.")
    harness_subparsers = harness_parser.add_subparsers(dest="harness_command", required=True)
    harness_run = harness_subparsers.add_parser("run", help="Run harness checks for one topic or all.")
    harness_run.add_argument("topic", nargs="?", default="all", help="Topic id or 'all'.")
    harness_mode = harness_run.add_mutually_exclusive_group()
    harness_mode.add_argument("--mock", action="store_true", help="Run deterministic local mode.")
    harness_mode.add_argument("--live", action="store_true", help="Run live external integrations.")
    harness_run.add_argument("--report-dir", default="reports", help="Directory for report files.")
    harness_run.add_argument("--no-report", action="store_true", help="Do not write report files.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "list":
        for topic in TOPICS:
            print(f"{topic.id}\t{topic.title}")
        return 0

    try:
        if args.command == "harness":
            selected_mode = "live" if args.live else "mock"
            report = run_harness(
                settings,
                selected_mode,
                topic_id=args.topic,
                report_dir=PROJECT_ROOT / args.report_dir,
                write_report=not args.no_report,
            )
            print(report.to_text())
            return 1 if report.status == "fail" else 0

        selected_mode = "live" if args.live else "mock"
        if args.topic == "all":
            for topic in TOPICS:
                result = topic.runner(settings, selected_mode)
                print(result.to_text())
            return 0
        topic = get_topic(args.topic)
        result = topic.runner(settings, selected_mode)
        print(result.to_text())
        return 0
    finally:
        wait_for_all_tracers()


if __name__ == "__main__":
    sys.exit(main())
