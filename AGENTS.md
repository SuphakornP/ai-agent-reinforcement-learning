# AGENTS.md

This repo contains CLI demos for NVIDIA-style agentic RL and harness engineering.

## Project Rules

- Use `uv` for dependency management and command execution.
- Keep demos runnable in `--mock` mode without external services.
- Keep live integrations behind explicit `--live` mode.
- Do not print, copy, summarize, or commit secrets from `.env`.
- Do not use OpenAI embeddings in this project. RAG embeddings must use Pinecone integrated inference with `multilingual-e5-large`.
- Do not create new Pinecone indexes unless the user explicitly asks and quota is available. The current real `.env` uses `PINECONE_INDEX_NAME=ap-thailand-kb-qa`.
- Keep topic folders self-contained with `README.md`, `demo.py`, and `mock_data/`.
- Prefer deterministic verifiers before model-based judges.
- Add tests whenever changing parsers, rewards, live adapters, or CLI behavior.

## Commands

Install or sync dependencies:

```bash
uv sync
```

List topics:

```bash
uv run agent-rl-demo list
```

Run all mock demos:

```bash
uv run agent-rl-demo run all --mock
```

Run all live demos:

```bash
uv run agent-rl-demo run all --live
```

Run harness checks:

```bash
uv run agent-rl-demo harness run all --mock
uv run agent-rl-demo harness run all --live
```

Run tests:

```bash
uv run pytest
```

Run live tests:

```bash
uv run pytest -m live
```

## Architecture Notes

- `src/agent_rl_demos/registry.py` lists all topics.
- `src/agent_rl_demos/topic_impls.py` implements all topic runners.
- `src/agent_rl_demos/rewards.py` owns verifier and reward logic.
- `src/agent_rl_demos/pinecone_rag.py` owns mock and live Pinecone integrated retrieval.
- `src/agent_rl_demos/live_openai.py` owns live OpenAI tool-call generation.
- `topics/*/mock_data/*.json` are the task/eval datasets.
- `docs/harness-engineering.md` contains the current harness engineering research note.

## Validation Expectations

Before finishing code changes, run:

```bash
uv run pytest
uv run agent-rl-demo run all --mock
uv run agent-rl-demo harness run all --mock
```

If the change touches OpenAI, Pinecone, LangSmith, or live config, also run:

```bash
uv run pytest -m live
uv run agent-rl-demo run all --live
uv run agent-rl-demo harness run all --live
```

## Secret Handling

- Safe checks: verifying whether required env vars are present.
- Unsafe checks: `cat .env`, `rg KEY .env`, printing env var values, or logging full config objects.
- `.env.example` may contain variable names and empty placeholders only.

## Harness Engineering Defaults

When adding a new demo:

- Start with mock data.
- Define the expected action schema.
- Add at least one negative example.
- Add a deterministic verifier if possible.
- Add metrics and failure buckets.
- Add tests for behavior and parser shapes.
- Add a topic README.
- Keep the demo narrow enough to explain in a workshop.

## Known Environment Notes

- This workspace was not initialized as a git repository at the time of setup.
- Pinecone project had 20/20 serverless indexes, so creating a new demo index failed. Reuse the configured integrated index unless the user changes the Pinecone project or frees quota.
