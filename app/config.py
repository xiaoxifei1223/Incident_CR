"""全局配置与 Bedrock 封装"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_aws import ChatBedrock


@dataclass
class BedrockConfig:
    """Bedrock LLM 配置

    可通过环境变量覆盖默认值：
    - AWS_REGION
    - BEDROCK_MODEL_ID
    - BEDROCK_TEMPERATURE
    - BEDROCK_MAX_TOKENS
    """

    region_name: str = os.getenv("AWS_REGION", "us-east-1")
    model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-sonnet-20240229-v1:0",
    )
    temperature: float = float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))


def get_bedrock_llm(config: Optional[BedrockConfig] = None) -> ChatBedrock:
    """构造一个 Bedrock Chat LLM 实例。

    注意：
    - 需要在环境中配置好 AWS 凭证与权限（对 Bedrock 的调用权限）。
    - 如果你的账户中可用模型不同，请修改 model_id。
    """

    cfg = config or BedrockConfig()
    return ChatBedrock(
        model_id=cfg.model_id,
        region_name=cfg.region_name,
        model_kwargs={
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        },
    )
