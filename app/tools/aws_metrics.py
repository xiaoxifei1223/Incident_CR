"""与 AWS CloudWatch 指标相关的工具函数"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
from langchain_core.tools import tool


def _get_boto3_client(service_name: str, region_name: Optional[str] = None):
    """简单封装 boto3.client，方便在不同工具中复用。"""

    import os

    region = region_name or os.getenv("AWS_REGION", "us-east-1")
    return boto3.client(service_name, region_name=region)


@tool
def get_cloudwatch_metric_statistics(
    namespace: str,
    metric_name: str,
    dimensions_json: str,
    period_minutes: int = 15,
    statistic: str = "Average",
) -> str:
    """查询指定 CloudWatch 指标的统计数据。

    参数：
    - namespace: 指标命名空间，例如 "AWS/EC2"、"AWS/ECS" 等。
    - metric_name: 指标名称，例如 "CPUUtilization"。
    - dimensions_json: 维度列表的 JSON 字符串，例如：
        '[{"Name": "InstanceId", "Value": "i-1234567890"}]'
    - period_minutes: 查询时间窗口（分钟），默认最近 15 分钟。
    - statistic: 统计方法，例如 "Average"、"Sum"、"Maximum"、"Minimum"。

    返回：CloudWatch get_metric_statistics 原始响应的简化 JSON 字符串。
    """

    cw = _get_boto3_client("cloudwatch")

    try:
        dimensions: List[Dict[str, Any]] = json.loads(dimensions_json) if dimensions_json else []
    except json.JSONDecodeError as e:
        raise ValueError("dimensions_json 必须是合法的 JSON 字符串") from e

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=period_minutes)

    resp = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=dimensions,
        StartTime=start_time,
        EndTime=end_time,
        Period=max(60, period_minutes * 60 // 5),
        Statistics=[statistic],
    )

    simplified = {
        "label": resp.get("Label"),
        "datapoints": resp.get("Datapoints", []),
        "metric_name": metric_name,
        "namespace": namespace,
        "statistic": statistic,
        "time_window_minutes": period_minutes,
    }
    return json.dumps(simplified, default=str)
