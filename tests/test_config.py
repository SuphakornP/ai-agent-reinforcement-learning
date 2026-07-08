from agent_rl_demos.config import load_settings


def test_config_uses_pinecone_embedding_default_not_openai_embedding() -> None:
    settings = load_settings()
    assert settings.openai_model == "gpt-5.4-mini"
    assert settings.pinecone_embed_model == "multilingual-e5-large"
    assert not hasattr(settings, "openai_embedding_model")


def test_langsmith_tracing_is_disabled_without_key() -> None:
    settings = load_settings()
    if not settings.langsmith_api_key:
        assert settings.langsmith_tracing == "false"
