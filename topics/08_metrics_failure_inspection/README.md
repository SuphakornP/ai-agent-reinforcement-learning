# 08 Metrics Failure Inspection

Topic นี้สาธิตการอ่าน metrics และ failure buckets ก่อนตัดสินใจว่า agent ดีขึ้นหรือแย่ลง. ใน agentic RL, average score อย่างเดียวไม่พอ ต้องรู้ด้วยว่าพลาดแบบไหน.

## แนวคิดหลัก

Evaluation ที่ดีควรตอบได้:

- success rate เท่าไร
- latency เป็นอย่างไร
- fail เพราะ format, wrong tool, wrong args, retrieval, หรือ safety
- failure ที่เพิ่มขึ้นเป็น regression หรือ expected tradeoff

Topic นี้เอา eval outputs มาผ่าน reward function แล้ว aggregate metrics.

## Mock Data

ไฟล์ `mock_data/eval_outputs.json` มี 3 eval cases:

- `eval_001`: calendar tool call ถูกต้อง
- `eval_002`: ใช้ `project.comment` แทน `project.update_task`
- `eval_003`: output เป็น plain text `open ticket`

แต่ละ case มี `latency_ms`.

## วิธีรัน

```bash
uv run agent-rl-demo run 08_metrics_failure_inspection --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "success_rate": 0.3333,
  "avg_latency_ms": 826.67,
  "failure_buckets": {
    "format_error": 1,
    "success": 1,
    "wrong_tool": 1
  }
}
```

แต่ละ record มี case-level reward result และ latency.

## วิธีตีความ

`success_rate = 0.3333` แปลว่าผ่าน 1 จาก 3 cases. แต่ failure buckets บอกมากกว่านั้น:

- มี `wrong_tool` 1 case -> อาจต้องแก้ tool selection prompt/schema หรือเพิ่ม RLVR eval
- มี `format_error` 1 case -> อาจต้องแก้ structured output prompt หรือใช้ SFT สำหรับ format
- latency เฉลี่ยช่วยบอก performance tradeoff

Topic นี้ควรใช้ก่อนตัดสินใจ deploy หรือก่อนเลือกวิธีปรับปรุง policy.

## Related Files

- `topics/08_metrics_failure_inspection/mock_data/eval_outputs.json`
- `src/agent_rl_demos/rewards.py`
- `src/agent_rl_demos/topic_impls.py`
