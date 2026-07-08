# 09 Continuous Improvement Loop

Topic นี้สาธิต flywheel ของ agent improvement: production failure ไม่ควรจบแค่ bug report แต่ควรถูกแปลงเป็น regression eval และเชื่อมกลับไปหา fix ที่เบาที่สุด.

## แนวคิดหลัก

Loop ที่ repo นี้ต้องการสื่อคือ:

```text
production trace -> failure bucket -> eval case -> harness check -> fix -> rerun
```

ถ้าไม่มี loop นี้ agent จะพัฒนาแบบเดาสุ่ม. ถ้ามี loop นี้ ทุก failure ที่พบจริงจะกลายเป็น test ที่ป้องกัน regression ในอนาคต.

## Mock Data

ไฟล์ `mock_data/production_failures.json` มี 3 production failures:

- `prod_001`: wrong tool, เรียก `project.comment` แทน `project.update_task`
- `prod_002`: retrieval failure, ดึง English eval notes แทน Thai refund policy
- `prod_003`: unsafe action, พยายามเรียก destructive tool

## วิธีรัน

```bash
uv run agent-rl-demo run 09_continuous_improvement_loop --mock
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "failures": 3,
  "new_evals": 3
}
```

แต่ละ record มี:

- `failure_id`
- `new_eval_id`
- `failure_type`
- `lightest_fix`

ตัวอย่าง mapping:

- `wrong_tool` -> `Verifier-backed RLVR eval`
- `retrieval_failure` -> `RAG index or chunk metadata fix`
- `unsafe_action` -> `Safety policy and environment guardrail`

## วิธีตีความ

ถ้า production failure ถูกแปลงเป็น `new_eval_id` ครบ แปลว่า harness มีทางเก็บ regression case แล้ว. ขั้นตอนถัดไปในระบบจริงคือเพิ่ม eval row เข้า dataset, rerun harness, แล้วแก้ prompt/RAG/verifier/policy ตาม `lightest_fix`.

Topic นี้คือภาพรวมของ continuous improvement: ไม่ใช่แค่แก้ครั้งเดียว แต่ทำให้ failure ทุกครั้งเพิ่มคุณภาพ eval suite.

## Related Files

- `topics/09_continuous_improvement_loop/mock_data/production_failures.json`
- `src/agent_rl_demos/topic_impls.py`
- `src/agent_rl_demos/harness.py`
