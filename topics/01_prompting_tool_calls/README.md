# 01 Prompting Tool Calls

Topic นี้สาธิตการวัดคุณภาพของ prompt/tool-call policy ก่อนทำ training. จุดสำคัญคือ output ของ agent ต้องไม่ใช่แค่ "ดูเหมือนถูก" แต่ต้องตรวจด้วย schema และ expected arguments ได้.

## แนวคิดหลัก

Agent ที่เรียก tool ต้องทำ 4 อย่างให้ถูก:

- สร้าง JSON หรือ structured output ที่ parse ได้
- เลือก tool ถูกตัว
- ส่ง arguments ถูก key และถูก value
- ไม่เรียก unsafe tool

ใน live mode repo นี้ใช้ OpenAI `gpt-5.4-mini` ผ่าน LangChain structured output. Prompt contract ใน `src/agent_rl_demos/live_openai.py` ระบุ required keys ของแต่ละ tool เพื่อให้ model ไม่เติม field เกิน เช่น `title` หรือ `description`.

## Mock Data

ไฟล์ `mock_data/tool_tasks.json` มี 3 tasks:

- `calendar_001`: สร้าง calendar event ให้ Alex วัน Tuesday เวลา `14:00`
- `ticket_001`: เปิด support ticket ของ account `ACME` priority `high` issue `login failures`
- `project_001`: ย้าย task `A-42` ใน project `Phoenix` ไป status `review`

แต่ละ task มี `expected_tool` และ `expected_args` สำหรับ verifier.

## วิธีรัน

Mock mode:

```bash
uv run agent-rl-demo run 01_prompting_tool_calls --mock
```

Live mode:

```bash
uv run agent-rl-demo run 01_prompting_tool_calls --live
```

## ผลลัพธ์ที่ควรเห็น

Output มี `metrics` เช่น:

```json
{
  "tasks": 3,
  "success_rate": 1.0,
  "mode": "mock"
}
```

ใน `records` แต่ละแถวจะมี:

- `task_id`: id ของ task
- `output`: tool call ที่ policy สร้าง
- `score`: reward score
- `failure_type`: `null` ถ้าผ่าน หรือเป็น bucket เช่น `wrong_arguments`
- `valid_json`: output parse ได้หรือไม่
- `correct_tool`: เลือก tool ถูกหรือไม่
- `argument_score`: args match กี่ส่วน
- `safe`: tool อยู่ใน allow/safe behavior หรือไม่

## ผลลัพธ์จากการรันจริงล่าสุด

รันด้วยคำสั่ง:

```bash
uv run agent-rl-demo run 01_prompting_tool_calls --live
```

ผลลัพธ์ที่ได้:

```json
{
  "metrics": {
    "mode": "live",
    "success_rate": 1.0,
    "tasks": 3
  },
  "records_count": 3
}
```

Tool calls ที่ model สร้างและ verifier ให้ผ่าน:

| Task | Tool | Args summary | Score |
| --- | --- | --- | --- |
| `calendar_001` | `calendar.create_event` | `attendee=Alex`, `day=Tuesday`, `time=14:00` | `1.0` |
| `ticket_001` | `support.create_ticket` | `account=ACME`, `priority=high`, `issue=login failures` | `1.0` |
| `project_001` | `project.update_task` | `project=Phoenix`, `task_id=A-42`, `status=review` | `1.0` |

ทุก record มี `valid_json=true`, `correct_tool=true`, `argument_score=1.0`, `safe=true`, และ `failure_type=null`. แปลว่า live OpenAI structured output ตรงกับ tool contract ทั้ง 3 tasks.

## วิธีตีความ

ถ้า `success_rate = 1.0` แปลว่า policy สร้าง tool calls ตรง contract ทั้งหมด. ถ้า live mode ได้ `wrong_arguments`, มักแปลว่า prompt contract เปิดช่องให้ model normalize ค่าไม่ตรงที่ verifier คาด เช่น `2 PM` แทน `14:00` หรือเติม field เกิน.

Topic นี้แสดงบทเรียนสำคัญของ harness engineering: prompt ที่ดีต้องผูกกับ verifier ที่วัดได้ ไม่ใช่แค่คำสั่งภาษาอังกฤษกว้าง ๆ.

## Related Files

- `topics/01_prompting_tool_calls/mock_data/tool_tasks.json`
- `src/agent_rl_demos/live_openai.py`
- `src/agent_rl_demos/rewards.py`
- `tests/test_live_openai.py`
