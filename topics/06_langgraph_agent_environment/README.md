# 06 LangGraph Agent Environment

Topic นี้สาธิตการใช้ LangGraph เป็น agent environment แบบเล็ก ๆ. Graph มี policy node และ verifier node เพื่อจำลอง loop ของ agentic RL โดยไม่ต้อง train model.

## แนวคิดหลัก

Environment สำหรับ agent ไม่ใช่แค่ prompt. มันคือ runtime ที่เก็บ state, เรียก policy, ตรวจ action, และตัดสินว่าจะจบหรือ retry.

Graph ใน demo นี้:

```text
START -> policy -> verifier -> END
                    |
                    +-> policy ถ้า reward ยังไม่ผ่าน
```

ใน mock demo policy เป็น deterministic จึงผ่านทันที.

## Mock Data

ไฟล์ `mock_data/environment_tasks.json` มี 2 tasks:

- `env_001`: calendar event สำหรับ Alex
- `env_002`: update project task `A-42` ไป `review`

แต่ละ task มี expected tool และ expected args.

## วิธีรัน

```bash
uv run agent-rl-demo run 06_langgraph_agent_environment --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "tasks": 2,
  "success_rate": 1.0
}
```

แต่ละ record มี:

- `task_id`
- `tool`
- `reward`
- `failure_type`

ใน mock mode `reward` ควรเป็น `1.0` และ `failure_type` ควรเป็น `null`.

## วิธีตีความ

Topic นี้แสดงความต่างระหว่าง "เรียกฟังก์ชันครั้งเดียว" กับ "agent environment". เมื่อมี graph เราสามารถเพิ่ม retry, routing, guardrails, memory, และ trace ได้.

สำหรับงานจริง LangGraph เหมาะเมื่อ agent มีหลายขั้นตอน หรือมีเงื่อนไขว่าต้องกลับไปแก้ action ก่อนจบ.

## Related Files

- `topics/06_langgraph_agent_environment/mock_data/environment_tasks.json`
- `src/agent_rl_demos/topic_impls.py`
- `tests/test_topics.py`
