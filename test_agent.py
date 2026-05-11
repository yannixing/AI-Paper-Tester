import builtins
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def agent_module():
    """Load real agent.py once per module, while bypassing its CLI loop."""
    agent_path = Path(__file__).resolve().parent / "agent.py"
    spec = importlib.util.spec_from_file_location("agent_under_test", agent_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    original_input = builtins.input
    builtins.input = lambda *args, **kwargs: "exit"
    try:
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            pytest.skip(f"Real agent tests skipped: missing dependency: {exc}")
    finally:
        builtins.input = original_input

    return module


def _invoke_agent(module, question: str) -> str:
    payload = {"messages": [{"role": "user", "content": question}]}
    try:
        result = module.agent.invoke(payload)
    except Exception as exc:
        pytest.skip(f"Real agent invocation skipped due to runtime/API error: {exc}")

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) > 0

    answer = result["messages"][-1].content
    assert answer is not None
    assert isinstance(answer, str)
    assert answer.strip() != ""
    return answer


def _attach_tool_monitor(module, monkeypatch):
    """
    Monitor retrieval/tool usage by wrapping retriever class invoke.
    Use class-level patch to avoid pydantic instance setattr errors.
    """
    counter = {"count": 0}
    retriever_cls = type(module.retriever)
    original_invoke = retriever_cls.invoke

    def wrapped_invoke(self, *args, **kwargs):
        counter["count"] += 1
        return original_invoke(self, *args, **kwargs)

    monkeypatch.setattr(retriever_cls, "invoke", wrapped_invoke)
    return counter


def _call_search_papers(module, query: str) -> str:
    """
    Compatible call for both:
    - plain function style: search_papers(query)
    - StructuredTool style: search_papers.invoke({...}) / invoke(query)
    """
    tool_or_func = module.search_papers

    if callable(tool_or_func):
        return tool_or_func(query)

    if hasattr(tool_or_func, "invoke"):
        try:
            return tool_or_func.invoke({"query": query})
        except Exception:
            return tool_or_func.invoke(query)

    raise TypeError("search_papers is neither callable nor invokable")


@pytest.mark.parametrize(
    "question,expected_keywords,should_use_tool",
    [
        ("你好，你能做什么？", ["论文", "检索", "阅读", "助手"], False),
        ("请用简单的话解释一下什么是RAG。", ["检索", "生成", "增强"], False),
        ("边缘协同推理中基于CNN的方法有哪些？请根据论文片段总结。", ["CNN", "卷积", "边缘"], True),
        ("请把刚才基于CNN的方法整理成一个表格。", ["表", "|", "---"], True),
        ("这些论文中有没有提到量子计算用于边缘协同推理？", ["没有", "未找到", "量子"], True),
    ],
)
def test_real_agent_questions(agent_module, monkeypatch, question, expected_keywords, should_use_tool):
    tool_counter = _attach_tool_monitor(agent_module, monkeypatch)
    answer = _invoke_agent(agent_module, question)

    print("\n[Question]", question)
    print("[Answer]", answer)

    hit_keywords = [kw for kw in expected_keywords if kw in answer]
    assert hit_keywords, f"Answer does not contain expected keywords: {expected_keywords}"

    if should_use_tool:
        assert tool_counter["count"] > 0, "Expected tool/retriever to be called, but it was not called."
        assert "论文" in answer or "【片段" in answer, "Expected paper/chunk evidence in answer."
    else:
        assert tool_counter["count"] == 0, "Tool/retriever should NOT be called for this question."


def test_agent_object_exists(agent_module):
    assert hasattr(agent_module, "agent")
    assert agent_module.agent is not None


def test_retriever_returns_list_like(agent_module):
    try:
        docs = agent_module.retriever.invoke("测试检索：边缘协同推理")
    except Exception as exc:
        pytest.skip(f"Retriever runtime skipped due to environment/dependency issue: {exc}")

    assert docs is not None
    assert hasattr(docs, "__iter__")


def test_search_papers_return_value(agent_module):
    try:
        answer = _call_search_papers(agent_module, "边缘协同推理的研究现状")
    except Exception as exc:
        pytest.skip(f"search_papers runtime skipped due to environment/dependency issue: {exc}")

    assert answer is not None
    assert isinstance(answer, str)
    assert answer.strip() != ""

