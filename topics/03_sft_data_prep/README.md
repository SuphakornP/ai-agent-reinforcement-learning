# 03 SFT Data Prep

Topic นี้สาธิตการเตรียมข้อมูลสำหรับ Supervised Fine-Tuning (SFT). ใน agent workflow, SFT เหมาะกับการสอน format, pattern, และ task behavior ที่มีตัวอย่างคำตอบที่ approved แล้ว.

## แนวคิดหลัก

SFT ไม่ใช่วิธีแก้ทุกอย่าง. ใช้เมื่อเรามี demonstration ที่ถูกต้องและอยากให้ model เลียนแบบ behavior นั้นให้สม่ำเสมอขึ้น.

ใน repo นี้ SFT data ถูกเตรียมเป็น chat-message rows:

- system message: instruction contract
- user message: prompt
- assistant message: accepted output

Trace ที่ไม่ปลอดภัยหรือไม่ accepted จะถูกกรองออก.

## Mock Data

ไฟล์ `mock_data/teacher_traces.json` มี 3 traces:

- `trace_001`: accepted calendar tool call
- `trace_002`: accepted support ticket tool call
- `trace_003`: rejected unsafe delete-all calendar action

ตัว demo จะใช้เฉพาะ `accepted: true`.

## วิธีรัน

```bash
uv run agent-rl-demo run 03_sft_data_prep --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "train_rows": 2,
  "eval_rows": 0,
  "mode": "mock"
}
```

และมี artifact:

```json
{
  "jsonl_preview": ["..."]
}
```

`jsonl_preview` คือ preview ของ rows ที่พร้อมเอาไปเขียนเป็น JSONL สำหรับ fine-tuning workflow.

## ผลลัพธ์จากการรันจริงล่าสุด

รันด้วยคำสั่ง:

```bash
uv run agent-rl-demo run 03_sft_data_prep --live
```

ผลลัพธ์ที่ได้:

```json
{
  "metrics": {
    "eval_rows": 0,
    "mode": "live",
    "train_rows": 2
  },
  "records_count": 0,
  "artifacts": ["jsonl_preview"]
}
```

Preview rows ที่ถูกสร้างมี 2 rows:

| Source trace | User prompt | Assistant target |
| --- | --- | --- |
| `trace_001` | `Create a calendar event for Alex next Tuesday at 2 PM.` | `calendar.create_event` with `attendee=Alex`, `day=Tuesday`, `time=14:00` |
| `trace_002` | `Open a high priority ticket for ACME.` | `support.create_ticket` with `account=ACME`, `priority=high`, `issue=login failures` |

Trace `trace_003` ไม่ถูกนำเข้า training rows เพราะเป็น rejected unsafe action (`calendar.delete_all`). ผลลัพธ์นี้จึงแสดง filtering behavior ของ SFT data prep ชัดเจน.

## วิธีตีความ

ผลลัพธ์นี้แสดงว่า SFT data pipeline ควรมี filtering step เสมอ. ถ้าเอา rejected trace เช่น `calendar.delete_all` เข้า training data, model อาจเรียนรู้ unsafe behavior.

ใน workflow จริง หลังจากได้ rows แล้วควรแยก train/eval set ให้ชัด และเก็บ held-out eval ไว้ตรวจว่า model ดีขึ้นจริง ไม่ใช่จำ training examples.

## Related Files

- `topics/03_sft_data_prep/mock_data/teacher_traces.json`
- `src/agent_rl_demos/topic_impls.py`
- `src/agent_rl_demos/io.py`
