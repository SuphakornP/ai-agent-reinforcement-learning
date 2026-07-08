# 04 DPO Preference Pairs

Topic นี้สาธิตข้อมูลแบบ chosen/rejected สำหรับ Direct Preference Optimization (DPO). DPO เหมาะเมื่อ output หลายแบบอาจ valid แต่ reviewer มี preference ว่าแบบไหนดีกว่า.

## แนวคิดหลัก

Preference pair มีสองฝั่ง:

- `chosen`: output ที่ reviewer หรือ rule เลือกว่าเหมาะกว่า
- `rejected`: output ที่ด้อยกว่า

ใน repo นี้เราใช้ deterministic verifier เดิมมาช่วยดู margin ระหว่าง chosen และ rejected. ใน production จริง บางกรณี preference อาจมาจาก human review หรือ LLM judge.

## Mock Data

ไฟล์ `mock_data/preference_pairs.json` มี 2 pairs:

- `pair_001`: chosen ใช้ `project.update_task`, rejected ใช้ `project.comment`
- `pair_002`: chosen เป็น JSON tool call ที่ถูกต้อง, rejected เป็น plain text ที่ไม่ใช่ tool call

แต่ละ pair มี `expected` เพื่อให้ verifier คำนวณ reward ได้.

## วิธีรัน

```bash
uv run agent-rl-demo run 04_dpo_preference_pairs --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "pairs": 2,
  "positive_margins": 2
}
```

แต่ละ record มี:

- `pair_id`
- `chosen_reward`
- `rejected_reward`
- `margin`

`margin = chosen_reward - rejected_reward`.

## ผลลัพธ์จากการรันจริงล่าสุด

รันด้วยคำสั่ง:

```bash
uv run agent-rl-demo run 04_dpo_preference_pairs --live
```

ผลลัพธ์ที่ได้:

```json
{
  "metrics": {
    "pairs": 2,
    "positive_margins": 2
  },
  "records_count": 2
}
```

Reward margin ของแต่ละ pair:

| Pair | Chosen reward | Rejected reward | Margin | Interpretation |
| --- | --- | --- | --- | --- |
| `pair_001` | `1.0` | `0.3333` | `0.6667` | chosen ใช้ `project.update_task` ถูกต้องกว่า rejected ที่ใช้ `project.comment` |
| `pair_002` | `1.0` | `-1.0` | `2.0` | chosen เป็น valid tool call ส่วน rejected เป็น plain text ที่ verifier ให้ `format_error` |

ทั้ง 2 pairs มี positive margin จึงผ่าน sanity check ว่า chosen ดีกว่า rejected ตาม reward function.

## วิธีตีความ

ถ้า `margin > 0` แปลว่า chosen ดีกว่า rejected ตาม verifier. ถ้า pair มี margin ติดลบ แปลว่า data หรือ verifier อาจมีปัญหา เพราะ rejected ได้คะแนนสูงกว่า chosen.

Topic นี้ทำให้เห็นว่า preference data ควรถูก sanity check ก่อนเอาไปใช้ train. DPO จะมีประโยชน์เมื่อ preference signal สะอาดพอ.

## Related Files

- `topics/04_dpo_preference_pairs/mock_data/preference_pairs.json`
- `src/agent_rl_demos/rewards.py`
- `src/agent_rl_demos/topic_impls.py`
