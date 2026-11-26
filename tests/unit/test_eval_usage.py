import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from eval.usage import AgentUsage


def reply(input_tokens):
    return AIMessage(
        content="结果",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 1,
            "total_tokens": input_tokens + 1,
        },
    )


@pytest.mark.parametrize(
    ("second", "expected"), [(reply(7), 17), (AIMessage(content="缺 usage"), None)]
)
async def test_usage_is_all_or_nothing(second, expected, fake_chat_model):
    model = fake_chat_model(responses=[reply(10), second])
    usage = AgentUsage()
    for _ in range(2):
        await model.ainvoke("问题", {"callbacks": [usage]})
    assert usage.input_tokens == expected


@pytest.mark.parametrize("first_call_fails", [True, False])
async def test_model_failure_leaves_the_case_usage_null(first_call_fails):
    class FailingModel(FakeMessagesListChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("model unavailable")

    usage = AgentUsage()
    if not first_call_fails:
        await FakeMessagesListChatModel(responses=[reply(10)]).ainvoke(
            "问题", {"callbacks": [usage]}
        )
        assert usage.input_tokens == 10
    with pytest.raises(RuntimeError, match="model unavailable"):
        await FailingModel(responses=[]).ainvoke("问题", {"callbacks": [usage]})
    assert usage.input_tokens is None
