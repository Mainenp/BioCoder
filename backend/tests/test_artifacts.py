import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import (
    SAFE_SUMMARY_FAILURE,
    BioAgent,
    extract_turn_artifacts,
    final_answer,
    has_tool_protocol_error,
    parse_dsml_tool_calls,
)


def test_extracts_and_deduplicates_sources_for_latest_turn() -> None:
    old = ToolMessage(
        name="search_pubmed",
        tool_call_id="old",
        content=json.dumps({"results": [{"title": "Old", "url": "https://example.com/old"}]}),
    )
    row = {
        "title": "Evidence",
        "url": "https://example.com/new",
        "source_type": "pubmed",
        "snippet": "A result",
        "metadata": {"pmid": "1"},
    }
    messages = [
        HumanMessage(content="old question"),
        old,
        HumanMessage(content="new question"),
        ToolMessage(name="search_pubmed", tool_call_id="1", content=json.dumps({"results": [row]})),
        ToolMessage(name="search_pubmed", tool_call_id="2", content=json.dumps({"results": [row]})),
        AIMessage(content="Answer"),
    ]
    sources, tools = extract_turn_artifacts(messages)
    assert [source.title for source in sources] == ["Evidence"]
    assert tools == ["search_pubmed"]
    assert final_answer(messages) == "Answer"


def test_recovers_deepseek_dsml_as_standard_tool_calls() -> None:
    content = """继续检索。
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="search_pubmed">
<｜｜DSML｜｜parameter name="max_results" string="false">8</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="query" string="true">Exscientia pipeline</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>"""

    visible, calls = parse_dsml_tool_calls(content, {"search_pubmed"})

    assert visible == "继续检索。"
    assert calls[0]["name"] == "search_pubmed"
    assert calls[0]["args"] == {"max_results": 8, "query": "Exscientia pipeline"}


def test_final_answer_never_exposes_dsml_or_reuses_an_old_turn() -> None:
    leaked = "<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"search_pubmed\"></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old answer"),
        HumanMessage(content="new"),
        AIMessage(content=leaked),
    ]

    assert final_answer(messages) == SAFE_SUMMARY_FAILURE
    assert has_tool_protocol_error(messages) is True


def test_forced_summary_retries_without_replaying_tool_protocol() -> None:
    leaked = "<｜｜DSML｜｜tool_calls></｜｜DSML｜｜tool_calls>"

    class FakeSummaryModel:
        def __init__(self) -> None:
            self.responses = [AIMessage(content=leaked), AIMessage(content="Clean final answer")]
            self.prompts = []

        async def ainvoke(self, messages):
            self.prompts.append(messages)
            return self.responses.pop(0)

    model = FakeSummaryModel()
    agent = BioAgent.__new__(BioAgent)
    agent.llm = model
    agent.settings = SimpleNamespace(
        prompt_cost_per_million=0.0,
        completion_cost_per_million=0.0,
    )
    state = {
        "messages": [
            HumanMessage(content="What is EGFR?"),
            ToolMessage(
                name="search_pubmed",
                tool_call_id="1",
                content=json.dumps(
                    {
                        "results": [
                            {
                                "title": "EGFR paper",
                                "url": "https://example.com/egfr",
                                "snippet": "EGFR is a receptor tyrosine kinase.",
                            }
                        ]
                    }
                ),
            ),
        ],
        "plan": ["retrieve", "summarize"],
    }

    result = asyncio.run(agent._forced_summary(state))

    assert result["messages"][0].content == "Clean final answer"
    assert len(model.prompts) == 2
    assert all(leaked not in str(prompt) for prompt in model.prompts)
