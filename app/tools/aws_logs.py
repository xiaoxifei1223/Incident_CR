"""与 AWS CloudWatch Logs Insights 相关的工具函数"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
from langchain_core.tools import tool

from .aws_metrics import _get_boto3_client


@tool
def query_cloudwatch_logs(
    log_group_name: str,
    query_string: str,
    minutes: int = 15,
    limit: int = 50,
) -> str:
    """使用 CloudWatch Logs Insights 查询日志样本。

    参数：
    - log_group_name: 日志组名称，例如 "/aws/lambda/my-function"。
    - query_string: Logs Insights 查询语句，例如：
        "fields @timestamp, @message | sort @timestamp desc | limit 20"
    - minutes: 查询最近多少分钟的日志，默认 15 分钟。
    - limit: 截断返回的结果条数上限，避免结果过大。

    返回：查询结果的 JSON 字符串（已序列化，包含前若干条记录）。
    """

    logs = _get_boto3_client("logs")

    end_time = int(datetime.utcnow().timestamp())
    start_time = int((datetime.utcnow() - timedelta(minutes=minutes)).timestamp())

    start_resp: Dict[str, Any] = logs.start_query(
        logGroupName=log_group_name,
        startTime=start_time,
        endTime=end_time,
        queryString=query_string,
    )

    query_id = start_resp["queryId"]

    for _ in range(30):
        resp = logs.get_query_results(queryId=query_id)
        status = resp["status"]
        if status in ("Complete", "Failed", "Cancelled"):
            results: List[Any] = resp.get("results", [])[:limit]
            simplified_rows: List[Dict[str, Any]] = []
            for row in results:
                simplified_rows.append({field["field"]: field["value"] for field in row})

            simplified = {
                "status": status,
                "rows": simplified_rows,
            }
            return json.dumps(simplified, default=str)
        time.sleep(1)

    return json.dumps(
        {
            "status": "Timeout",
            "message": "CloudWatch Logs Insights 查询在超时时间内未完成，请缩小时间范围或优化查询。",
        },
        ensure_ascii=False,
    )
