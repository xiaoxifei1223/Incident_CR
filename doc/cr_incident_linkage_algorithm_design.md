# CR-Incident 因果关联算法设计方案

## 问题诊断

### 当前方法的根本缺陷
传统的"文本相似度 + 时间窗口 + 服务匹配"方法存在以下问题：
1. **无法区分相关性与因果性**：两个事件相似≠一个导致另一个
2. **缺乏传播路径建模**：CR 如何通过依赖链引发 Incident 是黑盒
3. **忽略时序因果关系**：仅用时间窗口过滤，未建模事件演化过程
4. **缺少反事实验证**：无法回答"如果不做这个 CR，incident 会不会发生？"

---

## 解决方案：多算法融合 + LLM 增强

### 核心思路
采用**因果推断框架** + **图传播建模** + **LLM 语义理解**的组合方案

---

## 方案一：因果推断算法（Causal Inference）

### 1.1 Granger 因果检验（时序因果）
**适用场景**：判断 CR 时间序列是否"格兰杰因果"于 Incident 时间序列

**原理**：
- 如果 CR 事件的历史信息能显著提升预测 Incident 的能力，则认为存在因果关系

**实现步骤**：
```python
# 伪代码
from statsmodels.tsa.stattools import grangercausalitytests

# 构建时间序列
cr_series = daily_cr_count_by_service[service_id]
incident_series = daily_incident_count_by_service[service_id]

# Granger 检验（最大滞后 7 天）
result = grangercausalitytests(
    np.column_stack([incident_series, cr_series]), 
    maxlag=7
)
# p-value < 0.05 则认为 CR → Incident 存在因果
```

**优点**：
- 统计学严谨，可给出置信度
- 适合检测"某服务的 CR 频繁引发 incident"

**缺点**：
- 仅能检测"服务级别"因果，无法精确到单个 CR
- 需要足够长的时间序列数据

---

### 1.2 倾向得分匹配（Propensity Score Matching, PSM）

**适用场景**：消除混淆因素，对比"有 CR 的服务"和"无 CR 的服务"的 incident 发生率

**原理**：
- 构建"倾向得分"模型，预测一个服务是否会实施 CR
- 匹配倾向得分相似但 CR 实施情况不同的服务对
- 对比这两组服务的 incident 率差异

**实现步骤**：
```python
# 1. 训练倾向得分模型（预测服务是否会实施 CR）
from sklearn.linear_model import LogisticRegression

X = service_features[['历史CR频率', '服务复杂度', '团队规模', ...]]
y = service_has_cr  # 是否实施了 CR

ps_model = LogisticRegression().fit(X, y)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# 2. 匹配（使用 causalml 库）
from causalml.match import NearestNeighborMatch

matcher = NearestNeighborMatch(caliper=0.05)
matched_pairs = matcher.match(
    data=service_data,
    treatment_col='has_cr',
    score_col='propensity_score'
)

# 3. 计算因果效应（ATE: Average Treatment Effect）
ate = matched_pairs[matched_pairs.has_cr==1].incident_count.mean() - \
      matched_pairs[matched_pairs.has_cr==0].incident_count.mean()
```

**优点**：
- 可量化"CR 的因果效应"（做 CR 比不做 CR 增加多少 incident 风险）
- 控制混淆变量（如服务本身就不稳定）

**缺点**：
- 仍然是服务级别，无法精确到单个 CR

---

### 1.3 结构因果模型（Structural Causal Model, SCM）

**适用场景**：建模 CR → 服务变更 → 依赖服务故障 → Incident 的因果链

**原理**：
- 用有向无环图（DAG）表示变量间的因果关系
- 估计每条因果路径的效应

**示例因果图**：
```
CR 类型 → 服务 A 变更
    ↓
服务 A QPS 下降
    ↓
依赖服务 B 超时
    ↓
Incident 发生
```

**实现工具**：
- `dowhy`（微软开源）
- `causalnex`（Quantumblack 开源）

```python
import dowhy

# 定义因果图
causal_graph = """
digraph {
    CR -> ServiceA_Change;
    ServiceA_Change -> ServiceA_QPS;
    ServiceA_QPS -> ServiceB_Timeout;
    ServiceB_Timeout -> Incident;
    ServiceComplexity -> ServiceA_Change;  # 混淆因素
    ServiceComplexity -> Incident;
}
"""

model = dowhy.CausalModel(
    data=cr_incident_data,
    treatment='CR',
    outcome='Incident',
    graph=causal_graph
)

# 估计因果效应
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand)
print(f"CR → Incident 的因果效应: {estimate.value}")
```

**优点**：
- 可建模复杂因果链
- 可量化每条路径的贡献

**缺点**：
- 需要领域知识构建因果图
- 数据质量要求高

---

## 方案二：图神经网络 + 时空传播建模

### 2.1 时空图神经网络（Spatio-Temporal GNN）

**核心思想**：
- 构建"服务依赖图 + 时间维度"
- 用 GNN 建模"CR 在服务图上的传播"
- 预测 CR 引发的 incident 在时空中的扩散

**架构设计**：
```
输入：
- 服务依赖图 G = (V, E)
- CR 事件时间序列：X_t = [cr1, cr2, ...]
- 服务特征：node_features = [QPS, 错误率, CPU, ...]

输出：
- 每个时刻每个服务的 incident 概率：P(incident | CR, t, service)
```

**模型选择**：
- **STGCN**（Spatio-Temporal Graph Convolutional Network）
- **DCRNN**（Diffusion Convolutional Recurrent Neural Network）
- **Graph WaveNet**

**实现伪代码**（基于 PyTorch Geometric）：
```python
import torch
from torch_geometric.nn import GCNConv, GRU

class CRIncidentPropagationGNN(torch.nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim):
        super().__init__()
        self.gcn1 = GCNConv(node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = GRU(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)  # 预测 incident 概率
        
    def forward(self, x, edge_index, edge_weight, seq_len):
        # x: [batch, seq_len, num_nodes, features]
        # edge_index: 服务依赖图
        
        h_list = []
        for t in range(seq_len):
            # 空间卷积（在服务依赖图上传播）
            h = self.gcn1(x[:, t], edge_index, edge_weight)
            h = torch.relu(h)
            h = self.gcn2(h, edge_index, edge_weight)
            h_list.append(h)
        
        # 时间建模（用 GRU 捕捉延迟故障）
        h_seq = torch.stack(h_list, dim=1)  # [batch, seq_len, num_nodes, hidden]
        h_temporal, _ = self.gru(h_seq)
        
        # 预测
        incident_prob = torch.sigmoid(self.fc(h_temporal[:, -1]))  # 最后时刻
        return incident_prob
```

**数据准备**：
- 从 CMDB 构建服务依赖图
- 将 CR 事件编码为节点属性（one-hot 或 embedding）
- 滑动窗口构建训练样本（如 7 天窗口预测第 8 天 incident）

**优点**：
- 显式建模传播路径
- 捕捉跨服务依赖的级联故障
- 可解释性强（可视化传播热力图）

**缺点**：
- 需要准确的服务依赖图
- 计算复杂度高

---

## 方案三：LLM 赋能因果推理（关键创新）

### 3.1 LLM 的三种应用模式

#### 模式 1：文本因果提取（Information Extraction）
**目标**：从 CR/Incident 的非结构化文本中提取因果关系

**Prompt 设计**：
```python
prompt = f"""
你是一个 IT 运维专家，请分析以下 Change Request 和 Incident 是否存在因果关系。

**Change Request (CR-12345)**
- 时间：2024-11-01 10:00
- 服务：payment-service
- 描述：{cr_description}
- 影响：重启了 3 个 Tomcat 实例

**Incident (INC-67890)**
- 时间：2024-11-01 10:15
- 服务：order-service
- 现象：{incident_description}
- 根因分析：{incident_root_cause}

**任务**：
1. 判断 CR 是否导致了 Incident（是/否/不确定）
2. 如果是，请说明因果路径（例如：CR 重启 → payment-service 短暂不可用 → order-service 调用超时 → 用户下单失败）
3. 给出置信度（0-1）

**输出格式（JSON）**：
{{
    "is_causal": true/false,
    "causal_path": "...",
    "confidence": 0.85,
    "reasoning": "..."
}}
"""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)
```

**应用场景**：
- 自动标注 CR→Incident 训练数据
- 挖掘历史工单中隐藏的因果链路
- 辅助人工审核（提升效率）

---

#### 模式 2：根因推理（Root Cause Reasoning）
**目标**：给定 Incident，让 LLM 推理"可能是哪个 CR 引发的"

**Prompt 设计**：
```python
prompt = f"""
**Incident 信息**：
- 时间：2024-11-01 15:30
- 服务：user-service
- 现象：登录接口 500 错误，错误率 30%
- 日志：NullPointerException at UserAuthService.validate()
- 监控：数据库连接池耗尽

**候选 CR 列表**（过去 24 小时内）：
1. CR-001：user-service 升级 JDK 11 → 17（10:00）
2. CR-002：修改数据库连接池配置，maxPoolSize 从 50 → 20（14:00）
3. CR-003：新增用户画像接口（与登录无关）（09:00）

**任务**：
- 排序哪个 CR 最可能是根因
- 给出推理链路
- 输出 JSON：[{"cr_id": "CR-002", "probability": 0.9, "reasoning": "..."}, ...]
"""
```

**工程化方案**：
- 用 LLM 对候选 CR 进行重排序（Re-ranking）
- 结合传统召回（相似度、时间窗口）+ LLM 精排

---

#### 模式 3：知识增强（Knowledge-Augmented）
**目标**：用 LLM 提取领域知识，辅助特征工程

**示例应用**：
```python
# 用 LLM 提取"高风险操作关键词"
prompt = """
请列出 IT 运维中常见的高风险操作关键词（中文和英文），例如：
- 数据库：drop table, truncate, delete without where, alter schema
- 服务：restart, kill, force stop
- 配置：disable, remove, set to null

请输出 JSON 格式：{"database": [...], "service": [...], "config": [...]}
"""

risk_keywords = llm.generate(prompt)

# 用于特征工程
def extract_risk_score(cr_description):
    score = 0
    for category, keywords in risk_keywords.items():
        for kw in keywords:
            if kw.lower() in cr_description.lower():
                score += 1
    return score
```

---

### 3.2 LLM + 传统算法的混合架构

**推荐架构**：
```
阶段 1：粗召回（传统算法）
├── 时间窗口过滤（CR 实施后 1h~7天）
├── 服务/组件匹配
└── 文本相似度 Top-100

↓

阶段 2：因果验证（LLM）
├── 批量调用 LLM API
├── 每个 (CR, Incident) 对生成因果推理
└── 过滤 confidence < 0.7 的候选

↓

阶段 3：图传播验证（GNN）
├── 检查服务依赖路径是否存在
├── 模拟故障传播概率
└── 输出最终置信度

↓

输出：高置信 CR→Incident 链路
```

---

## 方案四：多模态学习（Text + Time + Graph）

### 核心思想
融合三种信号：
1. **文本信号**：CR/Incident 描述的语义相似度（BERT embeddings）
2. **时序信号**：事件时间的因果先验（Granger causality score）
3. **图信号**：服务依赖路径的可达性（GNN path score）

### 模型架构
```python
class MultiModalCausalityModel(nn.Module):
    def __init__(self):
        self.text_encoder = BertModel.from_pretrained('bert-base')
        self.time_encoder = TemporalConvNet(...)
        self.graph_encoder = GNN(...)
        self.fusion = nn.Linear(768 + 64 + 128, 1)
        
    def forward(self, cr_text, inc_text, time_features, graph_features):
        # 文本编码
        cr_emb = self.text_encoder(cr_text).pooler_output
        inc_emb = self.text_encoder(inc_text).pooler_output
        text_sim = cosine_similarity(cr_emb, inc_emb)
        
        # 时序编码
        time_score = self.time_encoder(time_features)
        
        # 图编码
        graph_score = self.graph_encoder(graph_features)
        
        # 融合
        combined = torch.cat([text_sim, time_score, graph_score], dim=-1)
        causality_prob = torch.sigmoid(self.fusion(combined))
        return causality_prob
```

---

## 实施建议：分阶段落地

### 阶段 1（1-2 周）：快速验证
- [ ] 使用 **LLM 文本因果提取**，标注 100 个样本
- [ ] 训练简单的 **二分类模型**（XGBoost），验证 AUC

### 阶段 2（3-4 周）：引入因果推断
- [ ] 实现 **Granger 因果检验**（服务级别）
- [ ] 实现 **PSM 倾向得分匹配**，量化 CR 的因果效应

### 阶段 3（1-2 月）：图传播建模
- [ ] 构建服务依赖图
- [ ] 实现 **时空 GNN**，预测故障传播路径

### 阶段 4（2-3 月）：LLM 深度集成
- [ ] 部署 LLM 推理服务（使用 Vertex AI 或自建）
- [ ] 实现 **LLM + GNN 混合排序**

---

## 评估指标设计

### 传统指标
- **Precision@K**：Top K 个预测的 CR 中，真实导致 incident 的比例
- **Recall@K**：真实导致 incident 的 CR 中，被召回到 Top K 的比例
- **AUC**：二分类模型的 ROC 曲线下面积

### 因果指标（更重要）
- **因果准确率**：模型预测的因果关系中，经人工验证为真的比例
- **反事实一致性**：如果移除预测的 CR，incident 是否消失（需要仿真或历史数据验证）

---

## 技术栈建议

| 模块 | 开源工具 | GCP 产品 |
|------|----------|----------|
| 因果推断 | dowhy, causalml | - |
| 图神经网络 | PyTorch Geometric, DGL | Vertex AI Training (GPU) |
| LLM 推理 | OpenAI API, Llama 2 | Vertex AI PaLM API |
| 时序分析 | statsmodels | - |
| 特征存储 | - | Vertex AI Feature Store |

---

## 总结：推荐方案组合

**短期（1 个月内）**：
- LLM 文本因果提取 + XGBoost 二分类

**中期（3 个月内）**：
- Granger 因果 + 时空 GNN + LLM 重排序

**长期（6 个月+）**：
- 结构因果模型（SCM）+ 多模态学习 + 反事实推理

---

## 参考文献
1. Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models"
2. Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
3. Microsoft DoWhy: https://github.com/py-why/dowhy
4. Spatio-Temporal GNN Survey: https://arxiv.org/abs/2104.13408
5. LLM for Causal Inference: https://arxiv.org/abs/2305.00050
