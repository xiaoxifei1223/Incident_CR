"""Incident 知识库与自动化相关工具（OpenSearch 矢量检索 + 自动化建议）"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_aws import BedrockEmbeddings
from langchain_core.tools import tool
from opensearchpy import OpenSearch


@tool
def search_incident_knowledge_base(
    query: str,
    index_name: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """使用 OpenSearch 矢量检索 Incident 知识库。

    参数：
    - query: 自然语言查询语句（事件描述、告警信息等）。
    - index_name: OpenSearch 索引名，默认从环境变量 OPENSEARCH_KB_INDEX 读取。
    - top_k: 返回相似文档条数，默认 5。

    依赖配置（通过环境变量）：
    - OPENSEARCH_ENDPOINT: OpenSearch 访问地址，例如 "https://search-xxx.region.es.amazonaws.com"。
    - OPENSEARCH_KB_INDEX: 默认的知识库索引名，例如 "incident-kb-index"。
    - OPENSEARCH_VECTOR_FIELD: 存储向量的字段名，默认 "vector"。
    - OPENSEARCH_TEXT_FIELD: 存储原始文本的字段名，默认 "content"。
    - OPENSEARCH_USER / OPENSEARCH_PASSWORD: 如果使用 Basic Auth，可配置用户名密码。
    - BEDROCK_EMBEDDING_MODEL_ID: Bedrock 向量模型 ID，例如 "amazon.titan-embed-text-v1"。

    返回：包含命中结果的 JSON 字符串，结构大致为：
    {
      "query": "...",
      "hits": [
        {"id": "...", "score": 1.23, "source": {...}}
      ]
    }
    """

    endpoint = os.getenv("OPENSEARCH_ENDPOINT")
    if not endpoint:
        raise RuntimeError("环境变量 OPENSEARCH_ENDPOINT 未配置，无法连接 OpenSearch。")

    index = index_name or os.getenv("OPENSEARCH_KB_INDEX", "incident-kb-index")
    vector_field = os.getenv("OPENSEARCH_VECTOR_FIELD", "vector")
    text_field = os.getenv("OPENSEARCH_TEXT_FIELD", "content")

    user = os.getenv("OPENSEARCH_USER")
    password = os.getenv("OPENSEARCH_PASSWORD")
    http_auth = (user, password) if user and password else None

    client = OpenSearch(
        hosts=[endpoint],
        http_auth=http_auth,
        use_ssl=endpoint.startswith("https"),
        verify_certs=True,
    )

    embedding_model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
    region = os.getenv("AWS_REGION", "us-east-1")
    embeddings = BedrockEmbeddings(model_id=embedding_model_id, region_name=region)

    # 计算查询向量
    query_vector: List[float] = embeddings.embed_query(query)

    body = {
        "size": top_k,
        "query": {
            "knn": {
                vector_field: {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
        "_source": [text_field],
    }

    resp = client.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    result_hits = []
    for h in hits:
        result_hits.append(
            {
                "id": h.get("_id"),
                "score": h.get("_score"),
                "source": h.get("_source", {}),
            }
        )

    result: Dict[str, Any] = {
        "query": query,
        "index": index,
        "hits": result_hits,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def generate_ssm_automation_suggestion(
    incident_summary: str,
    risk_level: str = "medium",
) -> str:
    """生成基于 AWS Systems Manager Automation 的自动化修复建议（占位）。

    参数：
    - incident_summary: 当前 Incident/告警的概要描述
    - risk_level: 风险等级（low/medium/high），可影响建议的自动化程度与是否需要审批。

    返回：一个结构化 JSON（字符串形式），描述建议的 Automation 文档与关键步骤。
    """

    suggestion: Dict[str, Any] = {
        "incident_summary": incident_summary,
        "risk_level": risk_level,
        "automation_document_suggestion": {
            "name": "AWS-Incident-ExampleAutomation",
            "description": "示例 Automation 文档占位，实际请使用你自定义的 SSM 文档。",
            "parameters": {
                "RollbackAllowed": True,
                "NeedApproval": risk_level.lower() == "high",
            },
            "steps_summary": [
                "收集 CloudWatch 指标与近期日志样本",
                "执行预检查（健康检查、依赖检查）",
                "根据策略执行扩容、重启、流量切换等操作",
                "执行回归验证（关键接口、业务指标）",
            ],
        },
    }
    return json.dumps(suggestion, ensure_ascii=False)
