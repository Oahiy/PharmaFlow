"""Create the OpenAI-compatible model used by the agent and compatibility checks."""

from langchain_openai import ChatOpenAI

from .config import AgentSettings, LLMSettings

TEMPERATURE = 0


def create_llm(settings: LLMSettings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=str(settings.base_url),
        api_key=settings.api_key,
        model=settings.model,
        temperature=TEMPERATURE,
        timeout=60,
        max_retries=2,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}}
        if settings.thinking_disabled
        else None,
    )


def create_summary_llm(settings: AgentSettings) -> ChatOpenAI | None:
    summary = settings.summary_llm
    return None if summary is None else create_llm(summary)
