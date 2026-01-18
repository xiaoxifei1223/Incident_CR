"""Agent 2：知识匹配修复 Agent"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from app.config import get_bedrock_llm
from app.tools.knowledge_base import (
    generate_ssm_automation_suggestion,
    search_incident_knowledge_base,
)


PROMPT_FILE_NAME = "kb_agent_prompt.txt"


def load_kb_system_prompt() -> str:
    """从同目录的 prompt 文件中读取系统提示词。"""

    prompt_path = Path(__file__).with_name(PROMPT_FILE_NAME)
    return prompt_path.read_text(encoding="utf-8")


class KBAgentState(TypedDict):
    """Agent 2 内部使用的状态定义，仅包含消息列表。"""

    messages: Annotated[List[BaseMessage], operator.add]


def build_kb_agent():
    """构建 Agent 2：知识匹配修复 Agent（ReAct 风格工具调用循环，带 SystemMessage）。"""

    llm = get_bedrock_llm()
    tools = [
        search_incident_knowledge_base,
        generate_ssm_automation_suggestion,
    ]
    tool_node = ToolNode(tools)

    def call_model(state: KBAgentState) -> KBAgentState:
        messages = state["messages"]
        # 如果还没有 SystemMessage，则从文件加载并插入到最前面
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_prompt = load_kb_system_prompt()
            messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(messages)
        return {"messages": messages + [response]}

    graph = StateGraph(KBAgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    app = graph.compile()
    return app
