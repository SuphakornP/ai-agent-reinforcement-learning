# 02 RAG Pinecone

Topic นี้สาธิต RAG สำหรับกรณีที่ agent ตอบผิดเพราะไม่มีข้อมูล domain. Live mode ใช้ Pinecone integrated inference ด้วย model `multilingual-e5-large` เพื่อให้ Pinecone embed, store, และ search ใน service เดียว.

## แนวคิดหลัก

RAG เหมาะเมื่อปัญหาคือ knowledge gap ไม่ใช่ reasoning gap. ถ้า model ไม่รู้ policy ภายในองค์กร การ train อาจไม่ใช่ทางเลือกแรก เพราะข้อมูลเปลี่ยนได้. การดึงข้อมูลล่าสุดจาก knowledge base มักง่ายกว่าและตรวจสอบได้มากกว่า.

Repo นี้ตั้งใจไม่ใช้ OpenAI embeddings. Embedding path คือ Pinecone integrated inference:

- `create_index_for_model(...)` สำหรับสร้าง integrated index เมื่อ quota พร้อม
- `upsert_records(...)` สำหรับส่ง raw text ให้ Pinecone embed และเก็บ
- `index.search(...)` สำหรับค้นหาโดยให้ Pinecone embed query และ search

## Mock Data

ไฟล์ `mock_data/records.json` มี 4 documents:

- Thai refund policy สำหรับลูกค้าในกรุงเทพ
- English reward design note
- Thai safety escalation note
- English held-out eval guidance

ไฟล์ `mock_data/queries.json` มี 3 queries:

- `q-th-refund`: ถามภาษาไทยเรื่องการขอคืนเงินในกรุงเทพ
- `q-en-reward`: ถามอังกฤษเรื่อง reward function ที่ trustworthy
- `q-th-safety`: ถามไทยเรื่องคำขอลบข้อมูลทั้งหมด

## วิธีรัน

Mock mode:

```bash
uv run agent-rl-demo run 02_rag_pinecone --mock
```

Live mode:

```bash
uv run agent-rl-demo run 02_rag_pinecone --live
```

## ผลลัพธ์ที่ควรเห็น

Output มี metrics:

```json
{
  "queries": 3,
  "records": 4,
  "pinecone_embed_model": "multilingual-e5-large",
  "mode": "mock"
}
```

ใน `records` ของผลลัพธ์ แต่ละ query จะมี:

- `query_id`
- `query`
- `matches`: list ของ documents ที่ retrieve ได้

แต่ละ match มี shape คล้าย Pinecone:

- `id`
- `score`
- `text`
- `metadata`

## ผลลัพธ์จากการรันจริงล่าสุด

รันด้วยคำสั่ง:

```bash
uv run agent-rl-demo run 02_rag_pinecone --live
```

ผลลัพธ์ที่ได้:

```json
{
  "metrics": {
    "mode": "live",
    "pinecone_embed_model": "multilingual-e5-large",
    "queries": 3,
    "records": 4
  },
  "records_count": 3
}
```

Top match ของแต่ละ query จาก Pinecone integrated inference:

| Query id | Query language | Top match | Topic | Score |
| --- | --- | --- | --- | --- |
| `q-th-refund` | Thai | `doc-th-001` | `refund policy` | `0.8855` |
| `q-en-reward` | English | `doc-en-001` | `reward design` | `0.8393` |
| `q-th-safety` | Thai | `doc-th-002` | `safety escalation` | `0.8825` |

ตัวอย่างผลที่สำคัญคือ query ภาษาไทย `ลูกค้าในกรุงเทพต้องใช้อะไรในการขอคืนเงิน` retrieve เอกสารภาษาไทย `doc-th-001` ที่พูดถึงใบเสร็จ, เลขคำสั่งซื้อ, และกรอบเวลา 30 วันได้เป็นอันดับหนึ่ง.

คะแนน similarity อาจเปลี่ยนเล็กน้อยในแต่ละ run ตาม Pinecone service/runtime แต่ expected behavior คือทุก query ต้องมี match และ top match ควรตรง topic.

## วิธีตีความ

ถ้า query ภาษาไทยเรื่อง refund ได้ document Thai refund policy เป็น match แปลว่า multilingual retrieval path ทำงานตามที่ต้องการ. ถ้า live mode ไม่มี matches, ให้ดู Pinecone index, namespace, field map, และข้อมูลที่ upsert.

Harness check `rag_matches_present` จะ fail ถ้า query ใดไม่มี matches. นี่ทำให้ RAG demo ไม่ใช่แค่เรียก API ได้ แต่ต้อง retrieve ข้อมูลกลับมาได้จริง.

## Related Files

- `topics/02_rag_pinecone/mock_data/records.json`
- `topics/02_rag_pinecone/mock_data/queries.json`
- `src/agent_rl_demos/pinecone_rag.py`
- `tests/test_rag.py`
