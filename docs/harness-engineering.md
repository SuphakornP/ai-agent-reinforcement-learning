# Harness Engineering Research Note

เอกสารนี้สรุป research เรื่อง harness engineering สำหรับ repo นี้ โดยเน้นสิ่งที่นำไปใช้กับ agentic RL demos ได้จริง.

## Definition

Harness engineering คือการออกแบบระบบรอบตัวโมเดลให้ agent ทำงานได้อย่างวัดผลได้และควบคุมได้ ไม่ใช่แค่เขียน prompt ให้ดีขึ้น.

ในบริบท AI agent, harness รวมถึง:

- prompt และ instruction contract
- tool catalog และ tool schema
- runtime state และ memory
- routing / middleware / hooks
- environment ที่รับ action แล้วให้ feedback
- verifier และ reward function
- trace collection
- evaluation datasets
- regression tests
- safety guardrails
- CI/CD checks

LangChain อธิบาย harness engineering ว่าเป็นการปรับ tooling รอบ model เพื่อ optimize task performance, token efficiency, latency, tool choice, execution flow, middleware, prompts, memory, and delegation. ประเด็นสำคัญคือสามารถ improve agent ได้โดยไม่ต้องเปลี่ยน model weights เสมอไป.

## Why This Matters For Agentic RL

บทความ NVIDIA วาง RL loop สำหรับ LLM/agents ไว้เป็น 7 ส่วน:

1. policy model
2. task
3. action
4. environment
5. verifier
6. rollouts
7. policy update

ในงาน agentic RL จริง, harness เป็นตัวกำหนดข้อ 2-6 เกือบทั้งหมด:

- task file คือ dataset
- action schema คือ output/tool-call contract
- environment คือ code ที่ execute action
- verifier คือ reward signal
- rollouts คือการ sample หลาย attempt
- metrics คือการตัดสินว่าควร update หรือ deploy หรือไม่

ถ้า harness ผิด, RL จะ optimize ผิดสัญญาณ. NVIDIA จึงแนะนำให้เริ่มจาก baseline eval, inspect failures, และ profile verifier/reward ก่อน train.

## Harness vs Prompt vs Context

Prompt engineering:

- ปรับคำสั่งให้ model ตอบดีขึ้น
- เหมาะกับ format, tone, reasoning hints, output contract

Context engineering:

- เลือกข้อมูลที่ใส่ให้ model เห็นในแต่ละ turn
- เหมาะกับ RAG, memory, retrieval, examples, tool docs

Harness engineering:

- ออกแบบ execution system รอบ model
- เหมาะกับ tools, state, guardrails, verifiers, evals, traces, CI, and feedback loops

ตัวอย่างใน repo นี้:

- Prompt: system instruction ใน `live_openai.py`
- Context: mock RAG records ใน `topics/02_rag_pinecone/mock_data`
- Harness: CLI + topic registry + LangGraph environment + verifier + tests + Pinecone adapter

## Practical Harness Patterns

### 1. Baseline Before Training

ก่อน SFT, DPO, หรือ RLVR ให้รัน baseline แล้ววัด:

- valid JSON rate
- correct tool rate
- argument match rate
- execution success rate
- unsafe action rate
- latency
- cost / token usage

Repo mapping:

- `00_decision_matrix`
- `08_metrics_failure_inspection`
- `tests/test_topics.py`

### 2. Deterministic Verifier First

เริ่มจาก verifier ที่ deterministic, cheap, reproducible:

- JSON schema check
- tool name check
- required argument check
- safety denylist
- execution result check

เพิ่ม LLM judge เฉพาะจุดที่ deterministic check ตัดสินไม่ได้.

Repo mapping:

- `src/agent_rl_demos/rewards.py`
- `05_rlvr_verifier_rewards`
- `tests/test_rewards.py`

### 3. Trajectory Evaluation

สำหรับ agent ที่ใช้ tools หลายขั้นตอน, final answer อย่างเดียวไม่พอ ต้องดู trajectory:

- tool calls ถูกตัวหรือไม่
- tool order ถูกหรือไม่
- argument drift เกิดจุดไหน
- retry loop ใช้เหตุผลหรือวนซ้ำ
- safety gate ทำงานหรือไม่

LangSmith และ LangChain AgentEvals รองรับทั้ง trajectory match และ LLM-as-judge trajectory evaluation.

Repo mapping:

- `06_langgraph_agent_environment`
- `tests/test_live_smoke.py`

### 4. Failure Buckets Become Work Items

อย่าเก็บแค่ pass/fail. ให้จัด bucket:

- `format_error`
- `schema_error`
- `wrong_tool`
- `wrong_arguments`
- `retrieval_failure`
- `unsafe_action`
- `long_horizon_failure`

จากนั้น map ไปหา fix ที่เบาที่สุด:

- prompt/schema fix
- RAG data/index fix
- SFT data
- preference pair
- verifier/reward update
- environment guardrail

Repo mapping:

- `09_continuous_improvement_loop`
- `bucket_failures(...)`

### 5. Harness Regression Tests

ทุกครั้งที่แก้ prompt, retriever, tool schema, verifier, หรือ routing ต้อง rerun evals.

Minimum local checks:

```bash
uv run pytest
uv run agent-rl-demo run all --mock
```

Live checks:

```bash
uv run pytest -m live
uv run agent-rl-demo run all --live
```

### 6. Controlled Live Integrations

Live integration ต้องมี fallback/mock mode เพื่อให้ demo และ CI ไม่พึ่ง external service ทุกครั้ง.

Repo mapping:

- `MockIntegratedPineconeRetriever`
- `PineconeIntegratedRetriever`
- `.env.example`
- live test skip behavior

### 7. Trace First, Dataset Later

OpenAI agent eval guidance แนะนำให้เริ่มจาก traces เมื่อยัง debug behavior แล้วค่อยย้ายไป datasets/eval runs เมื่อรู้แล้วว่า "good" คืออะไร.

Practical loop:

```text
trace -> inspect -> bucket -> add eval row -> update harness -> rerun -> compare
```

## How The Current Repo Implements The Harness

| Harness layer | Current implementation |
| --- | --- |
| Task set | `topics/*/mock_data/*.json` |
| Policy | `policies.py`, `live_openai.py` |
| Tool/action schema | JSON tool call with `tool` and `args` |
| Environment | `topic_impls.py`, LangGraph `StateGraph` |
| Verifier | `rewards.py` |
| Rollouts | `rollout_candidates(...)` |
| Metrics | `DemoResult.metrics` |
| Observability | LangSmith env config and tracer flush |
| RAG context | Pinecone integrated inference / mock retriever |
| Regression | `tests/*`, mock and live CLI runs |

## Harness Runtime Added To This Repo

Repo นี้มี harness runtime จริงที่รันได้ผ่าน CLI:

```bash
uv run agent-rl-demo harness run all --mock
uv run agent-rl-demo harness run all --live
```

Implementation:

- `src/agent_rl_demos/harness.py`
- `agent-rl-demo harness run`
- `reports/harness-*.json`
- `reports/harness-*.md`

Harness runtime ทำงานเป็นชั้นควบคุมเหนือ demo topics:

1. เลือก topic เดียวหรือทุก topic
2. รัน topic runner ใน mock/live mode
3. ประเมิน output ด้วย harness checks
4. สรุป status เป็น `pass`, `warn`, หรือ `fail`
5. เขียน report สำหรับ review และ regression tracking

Current checks:

- `metrics_present`
- `behavior_surface_present`
- `tool_outputs_safe`
- `success_rate`
- `rag_matches_present`
- `reward_coverage`
- `rollout_advantages`
- `failure_buckets_present`
- `failures_promoted_to_evals`

แนวคิดสำคัญคือ demo output ไม่ได้จบที่ "รันได้" แต่ต้องถูก harness ตรวจต่อว่า behavior น่าเชื่อถือหรือไม่.

## Design Checklist For New Agent Demos

Use this checklist before adding a new topic:

- Define the task input shape.
- Define the action/output schema.
- Add at least 3 mock examples.
- Add at least 1 negative/failure example.
- Add a deterministic verifier where possible.
- Define metrics before writing live code.
- Decide whether the demo needs external API calls.
- Add mock mode first.
- Add live mode only after mock mode is stable.
- Add tests for parser/adapter response shapes.
- Add README notes for the topic.
- Add failure buckets and continuous-improvement mapping if relevant.

## Recommended Next Improvements

1. Add LangSmith dataset upload/export for selected mock tasks.
2. Add `agentevals` trajectory evaluator for `06_langgraph_agent_environment`.
3. Add a `reports/` folder that stores timestamped eval summaries.
4. Add a command to promote production failures into `topics/09_continuous_improvement_loop/mock_data`.
5. Add a lightweight cost/latency summary for live OpenAI and Pinecone runs.
6. Add a model/prompt comparison command for `01_prompting_tool_calls`.

## Sources

- NVIDIA: [Mastering Agentic Techniques: AI Agent Reinforcement Learning](https://developer.nvidia.com/blog/mastering-agentic-techniques-ai-agent-reinforcement-learning/)
- LangChain: [Improving Deep Agents with harness engineering](https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering)
- LangSmith: [How to evaluate your agent with trajectory evaluations](https://docs.langchain.com/langsmith/trajectory-evals)
- LangSmith: [Evaluate a complex agent](https://docs.langchain.com/langsmith/evaluate-complex-agent)
- OpenAI: [Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)
- Anthropic: [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Pinecone: [multilingual-e5-large model docs](https://docs.pinecone.io/models/multilingual-e5-large)
- Survey: [Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems](https://arxiv.org/html/2605.18747v1)
