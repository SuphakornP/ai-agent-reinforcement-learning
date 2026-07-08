# 05 RLVR Verifier Rewards

Topic นี้สาธิตหัวใจของ Reinforcement Learning with Verifiable Rewards (RLVR): เอา eval logic ที่ตรวจได้มาแปลงเป็น reward function.

## แนวคิดหลัก

Reward function ใน demo นี้ให้คะแนน tool call จาก 3 ส่วน:

- `+0.4` ถ้าเลือก tool ถูก
- `+0.4` ตามสัดส่วน argument match
- `+0.2` ถ้า action safe

ถ้า output malformed, schema ผิด, หรือ unsafe จะถูกลงโทษและจัด failure bucket.

## Mock Data

ไฟล์ `mock_data/outputs.json` มี 4 cases:

- `ok`: JSON tool call ถูกต้องครบ
- `wrong_tool`: arguments ถูก แต่เรียก `calendar.search` แทน `calendar.create_event`
- `malformed`: output เป็น plain text ที่ parse JSON ไม่ได้
- `unsafe`: เรียก `calendar.delete_all`

## วิธีรัน

```bash
uv run agent-rl-demo run 05_rlvr_verifier_rewards --mock
```

หรือรัน harness เฉพาะ topic นี้:

```bash
uv run agent-rl-demo harness run 05_rlvr_verifier_rewards --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "cases": 4,
  "success_rate": 0.25
}
```

เพราะมีเพียง `ok` case เดียวที่ได้ score เต็ม.

แต่ละ record มี:

- `case_id`
- `score`
- `failure_type`
- `valid_json`
- `correct_tool`
- `argument_score`
- `safe`

Artifacts มี failure buckets:

```json
{
  "format_error": 1,
  "success": 1,
  "unsafe_action": 1,
  "wrong_tool": 1
}
```

## วิธีตีความ

RLVR ต้องการ verifier ที่แยกความผิดพลาดได้ชัด. ถ้า output ผิด tool แต่ JSON ถูก เราควรได้ `wrong_tool`. ถ้า output parse ไม่ได้ ควรได้ `format_error`. ถ้าเรียก destructive tool ควรได้ `unsafe_action`.

Topic นี้แสดงว่า reward ไม่ควรเป็นแค่ pass/fail. Reward ที่ดีควรให้ diagnostic signal เพื่อบอกว่าต้องแก้ส่วนไหนของ harness หรือ policy.

## Related Files

- `topics/05_rlvr_verifier_rewards/mock_data/outputs.json`
- `src/agent_rl_demos/rewards.py`
- `tests/test_rewards.py`
- `src/agent_rl_demos/harness.py`
