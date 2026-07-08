# 07 GRPO Rollout Simulation

Topic นี้จำลองแนวคิด GRPO-style rollout แบบเบา ๆ: สร้างหลาย candidate ต่อ task, ให้ verifier ให้คะแนน, แล้วคำนวณ relative advantage ภายในกลุ่ม.

## แนวคิดหลัก

Rollout คือการลองหลายคำตอบหรือหลาย action สำหรับ task เดียว. แทนที่จะดู output เดียว เราดู distribution ของ candidate:

- candidate ที่ถูกต้อง
- candidate ที่เลือก tool ผิด
- candidate ที่ args ผิด
- candidate ที่ unsafe
- candidate ที่ malformed

จากนั้นคำนวณ group mean reward และ advantage:

```text
advantage = candidate_reward - group_mean_reward
```

## Mock Data

ไฟล์ `mock_data/rollout_tasks.json` มี 2 tasks:

- `rollout_001`: calendar event
- `rollout_002`: support ticket

ฟังก์ชัน `rollout_candidates(...)` สร้าง 5 candidates ต่อ task รวม 10 rollout rows.

## วิธีรัน

```bash
uv run agent-rl-demo run 07_grpo_rollout_simulation --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "tasks": 2,
  "rollouts": 10,
  "best_reward": 1.0
}
```

แต่ละ record มี:

- `task_id`
- `candidate`
- `reward`
- `advantage`
- `failure_type`

Harness จะตรวจว่า rollout rows ทุกแถวมี `advantage`.

## วิธีตีความ

Candidate ที่ reward สูงกว่าค่าเฉลี่ยจะมี advantage เป็นบวก. Candidate ที่ malformed หรือ unsafe จะมี advantage เป็นลบ. ใน RL จริง signal นี้ช่วยบอกว่าการตอบแบบไหนควรถูก reinforce มากกว่า.

Repo นี้ไม่ได้ update model weights แต่แสดง mechanics ที่ต้องมีใน harness ก่อนเข้าสู่ training loop จริง.

## Related Files

- `topics/07_grpo_rollout_simulation/mock_data/rollout_tasks.json`
- `src/agent_rl_demos/policies.py`
- `src/agent_rl_demos/rewards.py`
- `src/agent_rl_demos/harness.py`
