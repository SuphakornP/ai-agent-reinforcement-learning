from agent_rl_demos.pinecone_rag import MockIntegratedPineconeRetriever, _coerce_match, _extract_search_hits


def test_mock_integrated_retriever_returns_pinecone_like_matches() -> None:
    retriever = MockIntegratedPineconeRetriever(
        [
            {
                "id": "th",
                "text": "นโยบายการขอคืนเงินในกรุงเทพต้องแนบใบเสร็จ",
                "language": "th",
                "topic": "refund policy",
                "source": "mock",
            },
            {
                "id": "en",
                "text": "Reward functions should measure the real task.",
                "language": "en",
                "topic": "reward design",
                "source": "mock",
            },
        ]
    )
    matches = retriever.search("กรุงเทพ ขอคืนเงิน", top_k=1)
    assert matches[0].id == "th"
    assert set(matches[0].as_dict()) == {"id", "score", "metadata"}


def test_integrated_search_response_shape_is_supported() -> None:
    response = {
        "result": {
            "hits": [
                {
                    "id_": "doc-en-001",
                    "score_": 0.83,
                    "fields": {"text": "Reward functions should measure the real task."},
                }
            ]
        }
    }
    hits = _extract_search_hits(response)
    match = _coerce_match(hits[0])
    assert match.id == "doc-en-001"
    assert match.score == 0.83
    assert match.metadata["text"].startswith("Reward functions")
