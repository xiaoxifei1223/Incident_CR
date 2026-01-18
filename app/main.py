"""应用入口：运行双 Agent Incident 工作流示例"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.workflow.incident_workflow import IncidentState, build_incident_workflow


def run_demo() -> None:
    """在当前虚拟环境中运行一个简单的示例对话。

    前置条件：
    - 已在虚拟环境中安装 langgraph / langchain-aws / boto3 等依赖；
    - 已配置 AWS_REGION、Bedrock 模型 ID 以及访问 Bedrock/CloudWatch/Logs 的 IAM 权限。
    """

    app = build_incident_workflow()

    user_description = (
        "我们在生产环境的某个微服务上出现了大量 5xx 错误，"
        "CloudWatch 告警显示该服务的 CPU 利用率在过去 10 分钟持续高于 85%，"
        "同时 API 延迟明显升高。请先帮我分析可能的原因和影响范围，"
        "然后基于历史经验给出一个修复方案和回滚策略。"
    )

    initial_state: IncidentState = {
        "messages": [HumanMessage(content=user_description)],
        "incident_context": {
            "service_name": "example-service",
            "environment": "prod",
        },
    }

    final_state = app.invoke(initial_state)

    last_message = final_state["messages"][-1]
    print("===== 最终回复 =====")
    print(last_message.content)


if __name__ == "__main__":
    run_demo()
