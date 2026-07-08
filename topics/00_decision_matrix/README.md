# 00 Decision Matrix

Topic นี้สาธิตแนวคิด problem-first ของ NVIDIA: อย่าเริ่มจากการเลือกเทคนิคที่ซับซ้อนที่สุด แต่ให้ดูว่า agent ล้มเหลวแบบไหน และมี signal อะไรให้วัดผล.

## แนวคิดหลัก

Decision matrix คือเครื่องมือช่วยตัดสินว่า failure แบบหนึ่งควรแก้ด้วยอะไร:

- ขาดข้อมูล domain -> ใช้ RAG หรือ data injection
- format หรือ JSON tool call ไม่นิ่ง -> เริ่มจาก prompting แล้วค่อย SFT
- มีตัวอย่างคำตอบที่ approved แล้ว -> ใช้ SFT
- คำตอบถูกหลายแบบแต่ reviewer ชอบแบบหนึ่งมากกว่า -> ใช้ DPO
- ความถูกต้องตรวจด้วย rule/test/execution ได้ -> ใช้ RLVR
- agent ล้มเหลวจากหลาย tool calls และ state updates -> ใช้ environment-based RL หรือ agent harness ที่เข้มขึ้น

## Mock Data

ไฟล์ `mock_data/cases.json` มี 6 failure patterns:

- `facts`: assistant ไม่รู้ internal policy facts
- `format`: assistant สร้าง JSON tool calls ไม่สม่ำเสมอ
- `examples`: model ต้องเลียนแบบ approved analyst notes
- `preferences`: คำตอบถูกหลายแบบ แต่ reviewer ชอบ style หนึ่ง
- `verifiable`: correctness ตรวจได้ด้วย schema และ execution tests
- `long_horizon`: agent พลาดหลังจากหลาย tool calls และ state updates

## วิธีรัน

```bash
uv run agent-rl-demo run 00_decision_matrix --mock
```

หรือรันผ่าน convenience script:

```bash
uv run python topics/00_decision_matrix/demo.py
```

## ผลลัพธ์ที่ควรเห็น

ผลลัพธ์เป็น JSON ที่มี `metrics.cases = 6` และ `records` 6 แถว. แต่ละแถวมี:

- `case`: id ของ failure pattern
- `problem`: คำอธิบายปัญหา
- `recommended`: technique ที่เบาที่สุดสำหรับแก้ปัญหา

ตัวอย่าง mapping:

- `facts` -> `RAG or data injection`
- `format` -> `Prompting, then SFT`
- `examples` -> `SFT`
- `preferences` -> `DPO`
- `verifiable` -> `RLVR with verifier rewards`
- `long_horizon` -> `Environment-based RL`

## ผลลัพธ์จากการรันจริงล่าสุด

รันด้วยคำสั่ง:

```bash
uv run agent-rl-demo run 00_decision_matrix --live
```

ผลลัพธ์ที่ได้:

```json
{
  "metrics": {
    "cases": 6,
    "mode": "live"
  },
  "records_count": 6
}
```

รายการ recommendation ที่ได้จริง:

| Case | Recommended technique |
| --- | --- |
| `facts` | `RAG or data injection` |
| `format` | `Prompting, then SFT` |
| `examples` | `SFT` |
| `preferences` | `DPO` |
| `verifiable` | `RLVR with verifier rewards` |
| `long_horizon` | `Environment-based RL` |

ผลลัพธ์นี้ยืนยันว่า decision matrix map failure signal ทั้ง 6 แบบไปยังเทคนิคที่เหมาะสมครบทุก case.

## วิธีตีความ

ถ้า output แนะนำ RAG แปลว่าปัญหาไม่ได้อยู่ที่ model reasoning แต่อยู่ที่ model ไม่มีข้อมูล. ถ้า output แนะนำ RLVR แปลว่าเรามี verifier ที่ตรวจ correctness ได้แล้ว จึงสามารถเปลี่ยน eval เป็น reward signal ได้.

Topic นี้ควรใช้เป็นจุดเริ่มต้นก่อนดู topic อื่น เพราะมันบอกว่าแต่ละเทคนิคเหมาะกับปัญหาแบบไหน.

## Related Files

- `topics/00_decision_matrix/mock_data/cases.json`
- `src/agent_rl_demos/topic_impls.py`
- `src/agent_rl_demos/registry.py`
