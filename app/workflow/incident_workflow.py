"""双 Agent Incident 工作流编排"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from typing_extensions import Annotated, TypedDict
import operator

from app.agents.analysis_agent import build_analysis_agent
from app.agents.kb_agent import build_kb_agent


class IncidentState(TypedDict):
    """用于多 Agent 工作流的共享状态。"""

    messages: Annotated[List[BaseMessage], operator.add]
    incident_context: Dict[str, Any]


def build_incident_workflow():
    """构建一个简单的“分析 -> 知识匹配修复”的双 Agent 工作流。"""

    analysis_agent_app = build_analysis_agent()
    kb_agent_app = build_kb_agent()

    def run_analysis_agent(state: IncidentState) -> IncidentState:
        result = analysis_agent_app.invoke(state)
        return {
            "messages": result["messages"],
            "incident_context": state.get("incident_context", {}),
        }

    def run_kb_agent(state: IncidentState) -> IncidentState:
        result = kb_agent_app.invoke(state)
        return {
            "messages": result["messages"],
            "incident_context": state.get("incident_context", {}),
        }

    workflow = StateGraph(IncidentState)
    workflow.add_node("analysis_agent", run_analysis_agent)
    workflow.add_node("kb_agent", run_kb_agent)

    workflow.set_entry_point("analysis_agent")
    workflow.add_edge("analysis_agent", "kb_agent")
    workflow.add_edge("kb_agent", END)

    app = workflow.compile()
    return app
