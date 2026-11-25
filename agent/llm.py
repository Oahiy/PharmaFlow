"""Create the OpenAI-compatible model used by the agent and compatibility checks."""

from langchain_openai import ChatOpenAI

from .config import LLMSettings


def create_llm(settings: LLMSettings) -> ChatOpenAI:
    deepseek = (
        "deepseek" in settings.model.casefold()
        or settings.base_url.host == "api.deepseek.com"
    )
    disable_thinking = deepseek or settings.disable_thinking
    return ChatOpenAI(
        base_url=str(settings.base_url),
        api_key=settings.api_key,
        model=settings.model,
        temperature=0,
        timeout=60,
        max_retries=2,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}} if disable_thinking else None,
    )
