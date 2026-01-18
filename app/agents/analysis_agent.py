"""Agent 1：数据分析预测 Agent"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from app.config import get_bedrock_llm
from app.tools.aws_logs import query_cloudwatch_logs
from app.tools.aws_metrics import get_cloudwatch_metric_statistics
from app.tools.timeseries import (
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    forecast_moving_average,
)


PROMPT_FILE_NAME = "analysis_agent_prompt.txt"


def load_analysis_system_prompt() -> str:
    """从同目录的 prompt 文件中读取系统提示词。"""

    prompt_path = Path(__file__).with_name(PROMPT_FILE_NAME)
    return prompt_path.read_text(encoding="utf-8")


class AnalysisAgentState(TypedDict):
    """Agent 1 内部使用的状态定义，仅包含消息列表。"""

    messages: Annotated[List[BaseMessage], operator.add]


def build_analysis_agent():
    """构建 Agent 1：数据分析预测 Agent（ReAct 风格工具调用循环，带 SystemMessage）。"""

    llm = get_bedrock_llm()
    tools = [
        get_cloudwatch_metric_statistics,
        query_cloudwatch_logs,
        detect_anomalies_zscore,
        detect_anomalies_iqr,
        forecast_moving_average,
    ]
    tool_node = ToolNode(tools)

    def call_model(state: AnalysisAgentState) -> AnalysisAgentState:
        messages = state["messages"]
        # 如果还没有 SystemMessage，则从文件加载并插入到最前面
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_prompt = load_analysis_system_prompt()
            messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(messages)
        return {"messages": messages + [response]}

    graph = StateGraph(AnalysisAgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    app = graph.compile()
    return app
