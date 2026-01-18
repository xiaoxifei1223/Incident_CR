"""时序分析与异常检测相关工具函数

这些工具不依赖外部重型库（如 Prophet、statsmodels），
实现了一些基础的统计类时序分析能力，方便 ReAct Agent 调用。
后续如果你希望接入 ARIMA/Prophet/LSTM，可以在此文件基础上扩展。
"""

from __future__ import annotations

import json
import math
from statistics import mean, pstdev
from typing import Any, Dict, List

from langchain_core.tools import tool


def _parse_series(series_json: str) -> List[Dict[str, Any]]:
    """解析时序 JSON，期望格式：
    [
      {"timestamp": "2025-01-01T00:00:00Z", "value": 0.123},
      ...
    ]
    """

    try:
        data = json.loads(series_json)
    except json.JSONDecodeError as e:
        raise ValueError("series_json 必须是合法的 JSON 字符串") from e

    if not isinstance(data, list):
        raise ValueError("series_json 顶层必须是列表")
    for item in data:
        if not isinstance(item, dict) or "value" not in item:
            raise ValueError("列表元素必须是包含 'value' 字段的对象")
    return data


@tool
def detect_anomalies_zscore(series_json: str, threshold: float = 3.0) -> str:
    """基于 Z-Score 的简单异常检测。

    参数：
    - series_json: 时序数据 JSON 字符串，格式为 [{"timestamp": ..., "value": ...}, ...]
    - threshold: 判定阈值，默认 3.0（即 |z| > 3 视为异常）

    返回：JSON 字符串，包含整体统计信息与异常点列表。
    """

    series = _parse_series(series_json)
    values = [float(p["value"]) for p in series]

    if len(values) < 2:
        result = {
            "summary": "样本太少，无法进行 Z-Score 异常检测。",
            "count": len(values),
        }
        return json.dumps(result, ensure_ascii=False)

    mu = mean(values)
    sigma = pstdev(values) or 1e-9  # 避免除零

    anomalies: List[Dict[str, Any]] = []
    for idx, point in enumerate(series):
        v = float(point["value"])
        z = (v - mu) / sigma
        if abs(z) > threshold:
            anomalies.append(
                {
                    "index": idx,
                    "timestamp": point.get("timestamp"),
                    "value": v,
                    "z_score": z,
                }
            )

    result: Dict[str, Any] = {
        "mean": mu,
        "std": sigma,
        "threshold": threshold,
        "total_points": len(values),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def detect_anomalies_iqr(series_json: str, factor: float = 1.5) -> str:
    """基于 IQR（四分位距）的异常检测（偏稳健，自适应长尾分布）。

    参数：
    - series_json: 时序数据 JSON 字符串，格式为 [{"timestamp": ..., "value": ...}, ...]
    - factor: IQR 放大因子，默认 1.5；值越大异常越少。

    返回：JSON 字符串，包含上下界与异常点列表。
    """

    series = _parse_series(series_json)
    values = sorted(float(p["value"]) for p in series)

    if len(values) < 4:
        result = {
            "summary": "样本太少，无法进行 IQR 异常检测。",
            "count": len(values),
        }
        return json.dumps(result, ensure_ascii=False)

    def percentile(sorted_vals: List[float], p: float) -> float:
        k = (len(sorted_vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    anomalies: List[Dict[str, Any]] = []
    for idx, point in enumerate(series):
        v = float(point["value"])
        if v < lower or v > upper:
            anomalies.append(
                {
                    "index": idx,
                    "timestamp": point.get("timestamp"),
                    "value": v,
                }
            )

    result: Dict[str, Any] = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "factor": factor,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def forecast_moving_average(
    series_json: str,
    window: int = 5,
    horizon: int = 3,
) -> str:
    """基于简单滑动平均的短期预测（基线能力）。

    参数：
    - series_json: 时序数据 JSON 字符串，格式为 [{"timestamp": ..., "value": ...}, ...]
    - window: 计算滑动平均的窗口长度，默认 5。
    - horizon: 预测未来点数量，默认 3。

    假设时间间隔近似均匀，仅对数值趋势进行粗略预测。
    返回：JSON 字符串，包含历史统计和未来 horizon 个点的预测值。
    """

    series = _parse_series(series_json)
    values = [float(p["value"]) for p in series]

    if len(values) < window:
        result = {
            "summary": "样本长度不足以计算滑动平均，请增大样本或减小 window。",
            "count": len(values),
        }
        return json.dumps(result, ensure_ascii=False)

    window_values = values[-window:]
    base = mean(window_values)

    # 这里采用非常简单的假设：未来 horizon 个点都接近最近 window 的均值
    forecasts = [base for _ in range(horizon)]

    result: Dict[str, Any] = {
        "window": window,
        "horizon": horizon,
        "history_count": len(values),
        "recent_window_values": window_values,
        "forecast_values": forecasts,
    }
    return json.dumps(result, ensure_ascii=False)
