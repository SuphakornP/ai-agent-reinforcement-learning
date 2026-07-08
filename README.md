# NVIDIA Agentic RL Demo + Harness Engineering

โปรเจกต์นี้เป็นชุด demo แบบ Python CLI สำหรับอธิบายแนวคิดจากบทความ NVIDIA
["Mastering Agentic Techniques: AI Agent Reinforcement Learning"](https://developer.nvidia.com/blog/mastering-agentic-techniques-ai-agent-reinforcement-learning/)
โดยแปลงแนวคิดเรื่อง agentic reinforcement learning ให้เป็น workshop ที่รันได้จริง.

เป้าหมายของ repo นี้ไม่ใช่การ fine-tune หรือ train น้ำหนักโมเดลจริง แต่เป็นการทำให้คนอ่านเข้าใจว่า ก่อนจะไปถึง RL จริง เราต้องมีระบบรอบโมเดลที่วัดผลได้ก่อน:

- มี task/eval data ที่ชัด
- มี action schema หรือ tool-call contract
- มี verifier ที่บอกได้ว่า output ถูกหรือผิด
- มี reward score ที่เอาไปใช้เป็น training signal ได้
- มี failure buckets เพื่อรู้ว่าควรแก้ด้วย prompt, RAG, SFT, DPO, RLVR หรือ environment guardrail
- มี harness runtime สำหรับรันซ้ำ, ตรวจผล, และสร้าง report

Repo นี้จึงเป็นทั้ง demo ของ agentic RL และตัวอย่าง harness engineering สำหรับ agent workflow.

## Quick Start

ติดตั้ง dependency:

```bash
uv sync
```

ดูรายการ topic:

```bash
uv run agent-rl-demo list
```

รันทุก demo แบบ mock โดยไม่ต้องใช้ external key:

```bash
uv run agent-rl-demo run all --mock
```

รัน harness checks ทุก topic:

```bash
uv run agent-rl-demo harness run all --mock
```

รัน tests:

```bash
uv run pytest
```

ถ้า `.env` มี OpenAI, Pinecone, และ LangSmith keys แล้ว สามารถรัน live path ได้:

```bash
uv run agent-rl-demo run all --live
uv run agent-rl-demo harness run all --live
```

## What This Repo Demonstrates

บทความ NVIDIA อธิบายว่า agentic RL ไม่ได้เริ่มจากการ train model ทันที. ขั้นตอนที่ควรทำก่อนคือวิเคราะห์ว่า agent ล้มเหลวแบบไหน และเรามี signal อะไรในการวัดผล.

Repo นี้แตกแนวคิดนั้นเป็น 10 topics:

| Topic | คำถามหลัก | สิ่งที่ demo แสดง |
| --- | --- | --- |
| `00_decision_matrix` | ปัญหาแบบนี้ควรใช้เทคนิคอะไร | เลือกวิธีแก้ที่เบาที่สุดตาม failure signal |
| `01_prompting_tool_calls` | prompt/tool-call ถูกต้องแค่ไหน | วัด JSON validity, tool choice, args, safety |
| `02_rag_pinecone` | ถ้าขาดข้อมูล domain ทำอย่างไร | ใช้ Pinecone integrated inference + multilingual RAG |
| `03_sft_data_prep` | SFT data หน้าตาอย่างไร | แปลง accepted traces เป็น chat-message rows |
| `04_dpo_preference_pairs` | preference data ใช้ตอนไหน | เปรียบเทียบ chosen/rejected output ด้วย reward margin |
| `05_rlvr_verifier_rewards` | reward function สร้างจาก eval ได้อย่างไร | แปลง verifier เป็น reward score และ failure bucket |
| `06_langgraph_agent_environment` | environment สำหรับ agent คืออะไร | ใช้ LangGraph ทำ policy -> verifier loop |
| `07_grpo_rollout_simulation` | rollout/advantage คืออะไร | สร้างหลาย candidate แล้วคำนวณ relative advantage |
| `08_metrics_failure_inspection` | จะรู้ได้อย่างไรว่า agent ดีขึ้น | รวม success rate, latency, failure buckets |
| `09_continuous_improvement_loop` | production failures กลับมาเป็น eval อย่างไร | แปลง failures เป็น regression evals และ next fixes |

## Mental Model

คิดแบบนี้จะอ่าน repo ง่าย:

```text
task -> policy/model -> action/tool call -> environment/verifier -> reward -> metrics -> next improvement
```

ใน repo นี้:

- `topics/*/mock_data/*.json` คือ task/eval data
- `src/agent_rl_demos/policies.py` และ `src/agent_rl_demos/live_openai.py` คือ policy
- output รูปแบบ `{"tool": "...", "args": {...}}` คือ action
- `src/agent_rl_demos/rewards.py` คือ verifier + reward function
- `src/agent_rl_demos/topic_impls.py` คือ runtime environment ของแต่ละ demo
- `src/agent_rl_demos/harness.py` คือ harness ที่รันทุกอย่างแล้วสรุป pass/warn/fail

ถ้า verifier ผิด, RL จะ optimize signal ผิด. เพราะฉะนั้น repo นี้ตั้งใจให้เห็นว่า "ตรวจให้ถูกก่อน train" สำคัญแค่ไหน.

## Glossary / คำศัพท์

**Agent** คือระบบที่ให้โมเดลตัดสินใจมากกว่าการตอบข้อความอย่างเดียว เช่น เลือก tool, ส่ง arguments, อ่านผลลัพธ์, และทำขั้นตอนต่อไป.

**Policy** คือส่วนที่เลือก action. ใน demo นี้ policy อาจเป็น deterministic mock policy หรือ live OpenAI model `gpt-5.4-mini`.

**Action** คือสิ่งที่ agent ทำ. ใน repo นี้ action ส่วนใหญ่คือ JSON tool call เช่น `calendar.create_event` พร้อม `args`.

**Tool call** คือการให้โมเดลเรียกเครื่องมือภายนอกด้วยชื่อ tool และ arguments ที่เป็น structured data.

**Tool schema / action contract** คือสัญญาว่า tool call ต้องมี field อะไรบ้าง เช่น `support.create_ticket` ต้องมี `account`, `priority`, `issue`.

**Environment** คือ runtime ที่รับ action, execute หรือ simulate action, แล้วส่งผลกลับ. ใน `06_langgraph_agent_environment` environment คือ LangGraph graph ที่มี policy node และ verifier node.

**Verifier** คือ logic ที่ตรวจว่า output ถูกหรือผิด เช่น valid JSON, tool ถูก, args ตรง, action ปลอดภัย.

**Reward** คือคะแนนที่ verifier คืนกลับมา. ตัวอย่างใน repo นี้ให้คะแนนจาก tool ถูก, argument match, และ safety.

**RLVR** ย่อจาก Reinforcement Learning with Verifiable Rewards. ใช้ reward ที่ตรวจได้ด้วย rule/test/execution แทนการพึ่ง human preference อย่างเดียว.

**Rollout** คือการลองให้ policy สร้างคำตอบหรือ action หลายครั้งต่อ task เดียวกัน เพื่อดู candidate หลายแบบ.

**Trajectory** คือเส้นทางการทำงานของ agent ตั้งแต่ input, tool calls, state updates, verifier result, จนถึง final output.

**GRPO-style simulation** ใน repo นี้คือการจำลอง group-relative scoring: สร้างหลาย candidate, หาคะแนนเฉลี่ยของกลุ่ม, แล้วคำนวณ advantage ของแต่ละ candidate. ไม่ได้ train model จริง.

**Advantage** คือคะแนนแบบ relative ว่า candidate หนึ่งดีกว่าค่าเฉลี่ยของกลุ่มเท่าไร. ค่าเป็นบวกคือดีกว่ากลุ่ม, ค่าเป็นลบคือแย่กว่ากลุ่ม.

**SFT** ย่อจาก Supervised Fine-Tuning. ใช้ตัวอย่างคำตอบที่ถูกต้องเพื่อสอน format, instruction following, และ task pattern.

**DPO** ย่อจาก Direct Preference Optimization. ใช้คู่ chosen/rejected เพื่อสอน preference เช่น style, helpfulness, หรือคำตอบที่ reviewer ชอบกว่า.

**RAG** ย่อจาก Retrieval-Augmented Generation. คือการดึงข้อมูลจาก knowledge base มาให้ model ใช้ ก่อนตอบหรือก่อนตัดสินใจ.

**Embedding** คือ vector representation ของข้อความ ใช้สำหรับค้นหาข้อความที่มีความหมายใกล้กัน.

**Vector database** คือฐานข้อมูลที่เก็บ vector และค้นหา nearest neighbors. Repo นี้ใช้ Pinecone เมื่อรัน live mode.

**Pinecone integrated inference** คือการให้ Pinecone ทำ embedding และ vector search ใน service เดียว. Repo นี้ใช้ `multilingual-e5-large` บน Pinecone แทน OpenAI embeddings.

**LangChain** คือ library สำหรับเชื่อม LLM, tools, prompts, structured output, และ integrations.

**LangGraph** คือ framework สำหรับทำ workflow/agent graph แบบมี state, node, edge, และ routing.

**LangSmith** คือระบบ tracing/evaluation/observability สำหรับดูว่า agent ทำอะไรในแต่ละ run.

**Harness engineering** คือการออกแบบระบบรอบโมเดล: prompts, tools, schemas, state, routing, verifiers, eval datasets, tracing, metrics, และ CI checks.

**Failure bucket** คือหมวดของความล้มเหลว เช่น `format_error`, `wrong_tool`, `wrong_arguments`, `retrieval_failure`, `unsafe_action`.

**Regression eval** คือ eval case ที่เพิ่มเข้ามาเพื่อกันไม่ให้ bug หรือ failure เดิมกลับมาอีก.

**Mock mode** คือโหมด local deterministic ที่ไม่เรียก external APIs เหมาะสำหรับ workshop, tests, และ CI.

**Live mode** คือโหมดที่เรียก OpenAI/Pinecone/LangSmith จริงตาม `.env`.

## Project Structure

```text
.
├── src/agent_rl_demos/
│   ├── cli.py                 # CLI entrypoint
│   ├── config.py              # .env loading and key checks
│   ├── harness.py             # harness runtime and report writer
│   ├── live_openai.py         # live OpenAI structured tool-call policy
│   ├── pinecone_rag.py        # Pinecone integrated RAG and mock retriever
│   ├── policies.py            # deterministic policies and rollout candidates
│   ├── registry.py            # topic registry
│   ├── rewards.py             # verifier, reward scoring, failure buckets
│   └── topic_impls.py         # implementation for all 10 topics
├── topics/
│   └── NN_topic_name/
│       ├── README.md          # explanation for that topic
│       ├── demo.py            # convenience runner
│       └── mock_data/         # task/eval/failure data
├── tests/                     # unit, CLI, mock, and live smoke tests
├── docs/
│   └── harness-engineering.md # research note
├── reports/                   # generated harness reports, ignored by git
├── AGENTS.md                  # instructions for future coding agents
├── .env.example               # env names only
├── pyproject.toml
└── uv.lock
```

## Configuration

Mock mode does not require keys.

Live mode uses `.env`:

```bash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini

PINECONE_API_KEY=
PINECONE_EMBED_MODEL=multilingual-e5-large
PINECONE_INDEX_NAME=ap-thailand-kb-qa
PINECONE_NAMESPACE=agentic-rl
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=nvidia-agentic-rl-demo
```

Important notes:

- อย่า print หรือ commit ค่า secret จาก `.env`.
- Repo นี้ตั้งใจไม่ใช้ OpenAI embedding model.
- RAG live path ใช้ Pinecone integrated inference `multilingual-e5-large`.
- `PINECONE_INDEX_NAME=ap-thailand-kb-qa` ถูกใช้เพราะมี integrated index อยู่แล้ว และ Pinecone project นี้เคยเต็ม quota 20/20 serverless indexes.
- ถ้าไม่มี `LANGSMITH_API_KEY`, mock/test path จะยังรันได้.

## Commands

List topics:

```bash
uv run agent-rl-demo list
```

Run one topic:

```bash
uv run agent-rl-demo run 05_rlvr_verifier_rewards --mock
```

Run all topics:

```bash
uv run agent-rl-demo run all --mock
uv run agent-rl-demo run all --live
```

Run Pinecone RAG live demo:

```bash
uv run agent-rl-demo run 02_rag_pinecone --live
```

Run harness without writing report:

```bash
uv run agent-rl-demo harness run all --mock --no-report
```

Run harness and write report:

```bash
uv run agent-rl-demo harness run all --mock
```

Run tests:

```bash
uv run pytest
uv run pytest -m live
```

## How To Read Demo Output

ทุก `agent-rl-demo run ...` command คืน JSON ที่มี shape เดียวกัน:

```json
{
  "topic": "05_rlvr_verifier_rewards",
  "title": "RLVR verifier and reward scoring",
  "summary": "...",
  "metrics": {},
  "records": [],
  "artifacts": {}
}
```

ความหมายของ field:

- `topic`: id ของ demo
- `title`: ชื่ออ่านง่าย
- `summary`: สิ่งที่ demo ทำ
- `metrics`: ตัวเลขสรุป เช่น `success_rate`, `cases`, `avg_latency_ms`
- `records`: รายละเอียดต่อ task/case/candidate
- `artifacts`: output เสริม เช่น failure bucket หรือ JSONL preview

ถ้าเห็น `success_rate: 1.0` แปลว่า case ทั้งหมดผ่าน verifier. ถ้าเห็น `failure_type` แปลว่า verifier จัดหมวด failure ได้แล้ว และสามารถเอาไปสร้าง regression eval ต่อได้.

## Topic Outputs At A Glance

| Topic | Command | ผลลัพธ์หลัก |
| --- | --- | --- |
| `00_decision_matrix` | `uv run agent-rl-demo run 00_decision_matrix --mock` | ได้ 6 records ที่ map failure pattern ไปหา technique เช่น RAG, SFT, DPO, RLVR |
| `01_prompting_tool_calls` | `uv run agent-rl-demo run 01_prompting_tool_calls --live` | ได้ tool call 3 รายการ พร้อม reward score และ failure type |
| `02_rag_pinecone` | `uv run agent-rl-demo run 02_rag_pinecone --live` | ได้ query results 3 รายการ โดยแต่ละ query มี Pinecone-style matches |
| `03_sft_data_prep` | `uv run agent-rl-demo run 03_sft_data_prep --mock` | ได้ train/eval counts และ `jsonl_preview` ของ accepted traces |
| `04_dpo_preference_pairs` | `uv run agent-rl-demo run 04_dpo_preference_pairs --mock` | ได้ chosen/rejected reward และ margin ต่อ preference pair |
| `05_rlvr_verifier_rewards` | `uv run agent-rl-demo run 05_rlvr_verifier_rewards --mock` | ได้ score, valid_json, correct_tool, argument_score, safe, failure bucket |
| `06_langgraph_agent_environment` | `uv run agent-rl-demo run 06_langgraph_agent_environment --mock` | ได้ผล policy -> verifier graph ต่อ task พร้อม reward |
| `07_grpo_rollout_simulation` | `uv run agent-rl-demo run 07_grpo_rollout_simulation --mock` | ได้ 10 rollout rows พร้อม reward และ advantage |
| `08_metrics_failure_inspection` | `uv run agent-rl-demo run 08_metrics_failure_inspection --mock` | ได้ success rate, average latency, และ failure bucket |
| `09_continuous_improvement_loop` | `uv run agent-rl-demo run 09_continuous_improvement_loop --mock` | ได้ mapping จาก production failures ไปเป็น eval ids และ lightest fixes |

รายละเอียดเชิงลึกอยู่ใน `topics/*/README.md`.

## Harness Runtime

คำสั่ง `run` แสดงผล demo. คำสั่ง `harness run` ตรวจว่า demo output มีคุณภาพพอหรือไม่.

```bash
uv run agent-rl-demo harness run all --mock
```

Harness ทำงานแบบนี้:

1. เลือก topic เดียวหรือทุก topic
2. รัน topic runner
3. ตรวจ generic checks เช่น metrics และ records/artifacts
4. ตรวจ topic-specific checks เช่น RAG matches, reward coverage, rollout advantage
5. สรุป status เป็น `pass`, `warn`, หรือ `fail`
6. เขียน report เป็น JSON และ Markdown ใน `reports/`

Current checks:

- `metrics_present`: topic ต้องมี metrics
- `behavior_surface_present`: topic ต้องมี records หรือ artifacts ให้ inspect
- `tool_outputs_safe`: tool call ต้องไม่เป็น unsafe action
- `success_rate`: success rate ต้องถึง threshold ที่กำหนด
- `rag_matches_present`: RAG query ทุกตัวต้องมี matches
- `reward_coverage`: reward demo ต้องมีทั้ง success และ failure cases
- `rollout_advantages`: rollout rows ต้องมี relative advantage
- `failure_buckets_present`: metrics ต้องมี failure bucket
- `failures_promoted_to_evals`: production failure ต้องถูก map เป็น eval id

ตัวอย่าง output summary:

```json
{
  "status": "pass",
  "summary": {
    "topic_status_counts": {"pass": 10, "warn": 0, "fail": 0},
    "check_status_counts": {"pass": 28, "warn": 0, "fail": 0}
  }
}
```

## Pinecone RAG Design

Topic `02_rag_pinecone` ใช้ Pinecone integrated inference ตาม requirement:

- embedding model: `multilingual-e5-large`
- live write path: `upsert_records(...)`
- live query path: `index.search(...)`
- mock path: in-memory retriever ที่คืน result shape คล้าย Pinecone

เหตุผลที่ใช้ integrated inference คือ Pinecone embed + store + search ใน service เดียว ลด network hop และลดการต้องจัดการ embedding client แยก.

Mock data มีทั้งไทยและอังกฤษ:

- Thai refund policy
- Thai safety escalation
- English reward design
- English held-out eval guidance

## LangSmith Tracing

ถ้า `.env` มี `LANGSMITH_API_KEY` และ `LANGSMITH_TRACING=true`, LangChain/LangGraph calls จะถูกส่งเข้า LangSmith project ที่ตั้งไว้.

Tracing มีประโยชน์ตอนตรวจ:

- prompt ที่ส่งเข้า model
- structured output ที่ model คืน
- tool/action arguments
- latency
- run-by-run comparison

## Testing Philosophy

Tests แบ่งเป็น deterministic และ live:

Deterministic tests:

- config defaults
- reward scoring
- failure buckets
- mock Pinecone retriever
- Pinecone response parsing
- LangGraph routing
- CLI dispatch
- harness report writer
- prompt contract builder

Live tests:

- OpenAI structured tool-call generation
- Pinecone integrated inference upsert/search
- RAG matches

Validation ล่าสุด:

```bash
uv run pytest
# 16 passed

uv run agent-rl-demo harness run all --mock --no-report
# pass: 10 topics, 28 checks

uv run agent-rl-demo harness run all --live --no-report
# pass: 10 topics, 28 checks
```

## How To Extend This Repo

เมื่อเพิ่ม topic ใหม่:

1. สร้าง `topics/NN_name/README.md`, `demo.py`, และ `mock_data/*.json`
2. เพิ่ม runner ใน `src/agent_rl_demos/topic_impls.py`
3. register topic ใน `src/agent_rl_demos/registry.py`
4. ถ้ามี verifier ใหม่ ให้เพิ่มหรือ reuse logic ใน `src/agent_rl_demos/rewards.py`
5. เพิ่ม harness check ถ้า topic มี expected behavior เฉพาะ
6. เพิ่ม tests
7. รัน `uv run pytest` และ `uv run agent-rl-demo harness run all --mock`

แนวคิดหลักคือทุก demo ควรตอบได้ 4 คำถาม:

- input/task คืออะไร
- expected action/output คืออะไร
- verifier ตรวจอะไร
- output ของ demo แปลว่าอะไร

## Additional Reading

- [docs/harness-engineering.md](docs/harness-engineering.md) อธิบาย harness engineering, research summary, และ checklist
- [AGENTS.md](AGENTS.md) กติกาสำหรับ coding agents ที่จะทำงานต่อใน repo นี้
