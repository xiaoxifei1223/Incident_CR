# CR-Incident 因果推理系统实施计划

## 项目概述

构建一个基于 GCP 的变更风险评估与因果推理系统，用于：
1. 预测哪些 CR 可能诱发 incident
2. 提出解决方案的 CR
3. 预测新 CR 对系统的影响

---

## Phase 0: 准备阶段（1-2周）

### 目标
- 搭建 GCP 基础设施
- 建立开发环境
- 完成数据源对接

### 关键任务

#### 0.1 GCP 环境初始化
- [ ] 创建 GCP 项目
- [ ] 配置 IAM 权限策略
- [ ] 设置 VPC 网络与防火墙规则
- [ ] 配置 Cloud Billing 预算告警

#### 0.2 核心服务部署
- [ ] 启用 BigQuery 并创建数据集
  - Dataset: `cr_incident_raw`（原始数据）
  - Dataset: `cr_incident_processed`（处理后数据）
  - Dataset: `cr_incident_features`（特征数据）
- [ ] 配置 Cloud Storage buckets
  - `cr-incident-raw-data`（原始导出）
  - `cr-incident-models`（模型存储）
  - `cr-incident-artifacts`（中间产物）
- [ ] 创建 Pub/Sub Topics
  - `cr-events`（CR 事件流）
  - `incident-events`（Incident 事件流）

#### 0.3 开发环境搭建
- [ ] 创建 Vertex AI Workbench 实例
- [ ] 配置 Cloud Composer（Airflow）环境
- [ ] 设置 Cloud Build CI/CD pipeline
- [ ] 建立代码仓库（GitHub + Cloud Source Repositories）

#### 0.4 数据源对接
- [ ] 对接 CR 系统 API/数据导出
- [ ] 对接 Incident 系统 API/数据导出
- [ ] 对接 CMDB 服务拓扑数据
- [ ] 对接 Trace/Log 系统（可选）

### 交付物
- [x] GCP 项目架构文档
- [x] 数据源对接文档
- [x] 开发环境访问指南

---

## Phase 1: 数据基础建设（2-3周）

### 目标
- 建立数据采集与清洗流程
- 构建 CR→Incident 链路数据集
- 完成初步数据探索

### 关键任务

#### 1.1 数据采集层实现
- [ ] 实现 Pub/Sub → Dataflow → BigQuery 实时流
  - CR 事件实时写入
  - Incident 事件实时写入
- [ ] 实现批量数据导入脚本
  - 历史 CR 数据（建议至少 6 个月）
  - 历史 Incident 数据（同步时间范围）
  - CMDB 服务关系数据

#### 1.2 数据清洗与标准化
- [ ] 数据质量检查脚本
  - 缺失值处理
  - 重复数据去重
  - 时间戳格式统一
- [ ] 字段映射与标准化
  - CR 字段标准化（ID、时间、服务、类型、描述等）
  - Incident 字段标准化（ID、时间、服务、严重级别、根因等）

#### 1.3 链路挖掘（核心任务）
- [ ] **规则挖掘**：BigQuery SQL 实现
  - 正则提取 Incident resolution 中的 CR ID
  - 正则提取 CR description 中的 Incident ID
  - 时间窗口匹配（CR 实施后 1h~7天内的 incident）
- [ ] **相似度匹配**：候选对生成
  - 基于服务/组件匹配
  - 基于文本相似度（TF-IDF）
  - 基于 assign group 匹配
- [ ] **人工标注平台**（可选）
  - 使用 Vertex AI Data Labeling 或自建工具
  - 标注至少 100-500 个 CR→Incident 对

#### 1.4 数据探索与分析
- [ ] 统计分析报告
  - CR 数量、类型分布
  - Incident 数量、严重级别分布
  - CR→Incident 链路覆盖率
- [ ] 创建 Looker Studio Dashboard（初版）
  - CR/Incident 时间趋势
  - 服务故障 Top 10
  - 高风险 CR 类型分析

### 交付物
- [x] 链路数据集（至少 1000 条 CR→Incident 对）
- [x] 数据质量报告
- [x] 数据探索 Dashboard

---

## Phase 2: 特征工程与模型训练（3-4周）

### 目标
- 构建完整的特征工程流程
- 训练风险预测模型
- 验证模型效果

### 关键任务

#### 2.1 特征工程实现
- [ ] **结构化特征**
  - CR 属性特征：类型、优先级、影响服务数、是否紧急
  - 时间特征：是否周末、是否高峰期、是否发布窗口
  - 历史特征：同服务过去 30 天 CR 次数、incident 次数、失败率
- [ ] **文本特征**
  - CR description 的 TF-IDF 特征
  - 风险关键词匹配（drop、delete、restart、truncate 等）
  - 使用 Vertex AI Text Embeddings（BERT）生成语义向量
- [ ] **图特征**（可选）
  - 构建服务依赖图（NetworkX）
  - 计算节点中心度、PageRank
  - 使用 DGL 训练 GNN 节点嵌入
- [ ] **Feature Store 集成**
  - 将特征注册到 Vertex AI Feature Store
  - 配置特征在线/离线服务

#### 2.2 模型训练
- [ ] **基线模型：XGBoost**
  - 二分类任务：CR 是否诱发 incident
  - 训练集/验证集/测试集划分（70/15/15）
  - 超参数调优（使用 Vertex AI Hyperparameter Tuning）
  - 目标：AUC > 0.75
- [ ] **进阶模型：BERT 文本分类**
  - 微调 DistilBERT/ALBERT 模型
  - 仅使用 CR description 文本
  - 目标：AUC > 0.78
- [ ] **图神经网络（可选）**
  - 使用 DGL + PyTorch
  - 建模服务依赖图 + CR 影响路径
  - 目标：提升召回率

#### 2.3 模型评估与解释
- [ ] 评估指标计算
  - AUC、Precision、Recall、F1-Score
  - Top-K 准确率（Top 10% 高风险 CR 的命中率）
- [ ] 模型可解释性
  - SHAP 值分析（XGBoost）
  - Vertex AI Explainable AI 集成
  - 生成特征重要性报告
- [ ] 错误案例分析
  - 分析 False Positive/False Negative
  - 迭代优化特征

### 交付物
- [x] 特征工程 Pipeline（Airflow DAG）
- [x] 训练好的模型（注册到 Vertex AI Model Registry）
- [x] 模型评估报告

---

## Phase 3: 影响范围预测与案例推荐（2-3周）

### 目标
- 实现新 CR 影响范围预测
- 构建历史案例推荐引擎
- 完成模型部署

### 关键任务

#### 3.1 服务依赖图构建
- [ ] 从 CMDB 提取服务调用关系
- [ ] 构建有向图（NetworkX / Neo4j）
- [ ] 添加边权重（调用频率、历史故障传播概率）

#### 3.2 影响范围模拟
- [ ] **概率传播算法**
  - 给定 CR 影响的初始服务节点
  - 计算故障沿依赖链传播的概率
  - 输出"可能受影响的服务列表 + 概率"
- [ ] **GNN 预测**（可选）
  - 使用图神经网络预测影响范围
  - 训练数据：历史 CR 影响的实际服务集合

#### 3.3 案例推荐引擎
- [ ] **向量化历史 CR**
  - 使用 BERT embeddings 或 Vertex AI Text Embeddings
  - 存储到 Vertex AI Matching Engine
- [ ] **相似度召回**
  - 给定新 CR，召回 Top-10 相似历史 CR
  - 附带解决方案、影响范围、是否成功
- [ ] **案例库管理**
  - BigQuery 存储 CR 案例详情
  - 支持按服务、类型、时间筛选

#### 3.4 模型部署
- [ ] 部署风险预测模型到 Vertex AI Endpoint
  - 配置 Auto-scaling
  - 配置 Monitoring & Alerting
- [ ] 部署影响范围预测 API
- [ ] 部署案例推荐 API

### 交付物
- [x] 服务依赖图（可视化）
- [x] 影响范围预测 API
- [x] 案例推荐 API
- [x] API 文档

---

## Phase 4: 系统集成与前端开发（2-3周）

### 目标
- 构建用户界面
- 集成所有后端服务
- 完成端到端测试

### 关键任务

#### 4.1 API 网关与服务编排
- [ ] 使用 Cloud Endpoints 或 API Gateway
- [ ] 统一 API 鉴权（OAuth 2.0 / Service Account）
- [ ] API 限流与配额管理

#### 4.2 前端开发
- [ ] **Looker Studio Dashboard**
  - CR 风险评分实时监控
  - Incident 趋势分析
  - 高风险 CR 告警列表
- [ ] **内部 Portal（React / Vue）**
  - CR 提交时实时风险评分
  - 影响范围可视化（服务依赖图）
  - 相似案例推荐展示
  - 风险解释（SHAP 可视化）

#### 4.3 告警与通知
- [ ] 集成 Pub/Sub + Cloud Functions
  - 高风险 CR 自动发送邮件/Slack 通知
  - Incident 发生时自动关联可疑 CR
- [ ] 配置 Cloud Monitoring Alerting
  - API 延迟、错误率告警
  - 模型性能下降告警

#### 4.4 端到端测试
- [ ] 功能测试
  - 测试 CR 提交 → 风险评分 → 推荐案例 全流程
- [ ] 性能测试
  - 压测 API（目标：P95 延迟 < 200ms）
- [ ] 用户验收测试（UAT）
  - 邀请 5-10 个团队试用
  - 收集反馈并迭代

### 交付物
- [x] 前端界面（Portal + Dashboard）
- [x] API 文档
- [x] 用户使用手册

---

## Phase 5: 上线与运营（持续）

### 目标
- 灰度发布
- 监控与优化
- 持续迭代

### 关键任务

#### 5.1 灰度发布
- [ ] 选择 2-3 个团队作为 pilot
- [ ] Vertex AI Endpoint 配置流量分配（10% → 50% → 100%）
- [ ] 监控关键指标
  - 用户采纳率（CR 提交时查看风险评分的比例）
  - 模型准确率（实际发生 incident 的 CR 是否被预测为高风险）
  - API 可用性（SLA > 99.5%）

#### 5.2 模型再训练与优化
- [ ] 每月自动再训练模型（Airflow 调度）
- [ ] 新增标注数据持续入库
- [ ] A/B 测试新特征/新模型

#### 5.3 运营优化
- [ ] 建立反馈机制
  - 用户可标记"误报/漏报"
  - 将反馈数据加入训练集
- [ ] 构建数据飞轮
  - CR 提交 → 风险预测 → 实际结果 → 模型优化 → 更好预测

#### 5.4 文档与培训
- [ ] 编写运维手册
- [ ] 组织用户培训
- [ ] 定期发布系统使用报告

### 交付物
- [x] 灰度上线报告
- [x] 月度运营报告
- [x] 模型性能监控 Dashboard

---

## 资源估算

### 人力资源
- **数据工程师**：2 人（Phase 0-1）
- **机器学习工程师**：2 人（Phase 2-3）
- **前端开发**：1 人（Phase 4）
- **DevOps/SRE**：1 人（全周期）
- **产品经理**：1 人（全周期）

### GCP 成本估算（月）
- **BigQuery**：$500-1000（存储 + 查询）
- **Vertex AI Training**：$300-500（模型训练）
- **Vertex AI Endpoint**：$200-400（在线预测）
- **Cloud Composer**：$300（Airflow 托管）
- **Cloud Storage**：$50-100
- **其他（Pub/Sub、Dataflow 等）**：$200
- **总计**：约 $1500-2500/月

### 时间估算
- **总工期**：3-4 个月（不含 Phase 5 持续运营）
- **MVP 上线**：2 个月（Phase 0-2）

---

## 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| CR/Incident 数据质量差 | 模型效果差 | Phase 1 强化数据清洗，提前与数据源 owner 对齐 |
| 链路数据标注量不足 | 模型训练困难 | 使用半监督学习，主动学习减少标注成本 |
| 跨团队数据权限问题 | 无法获取完整数据 | 提前申请数据访问权限，设计联邦学习方案 |
| 模型上线后效果不达预期 | 用户信任度低 | 先从"辅助决策"而非"自动拦截"切入 |
| GCP 成本超预算 | 项目中止 | 使用 Preemptible VMs，优化 BigQuery 查询 |

---

## 成功指标（KPI）

### 技术指标
- 模型 AUC > 0.75（Phase 2）
- API P95 延迟 < 200ms（Phase 4）
- 系统可用性 SLA > 99.5%（Phase 5）

### 业务指标
- 高风险 CR 识别率 > 60%（Top 10% 评分的 CR 中，实际诱发 incident 的比例）
- 用户采纳率 > 50%（提交 CR 时查看风险评分的用户比例）
- Incident 数量同比下降 > 10%（6 个月后）

---

## 附录

### 技术栈总结
| 层级 | GCP 产品 | 开源工具 |
|------|----------|----------|
| 数据采集 | Pub/Sub, Dataflow, Cloud Storage | - |
| 数据仓库 | BigQuery | - |
| 调度编排 | Cloud Composer | Airflow |
| 特征管理 | Vertex AI Feature Store | - |
| 模型训练 | Vertex AI Training | XGBoost, TensorFlow, PyTorch, DGL |
| 模型部署 | Vertex AI Endpoint | - |
| 向量召回 | Vertex AI Matching Engine | - |
| 图计算 | - | NetworkX, DGL |
| 可视化 | Looker Studio | - |
| 前端 | - | React / Vue |

### 参考资料
- [Vertex AI 文档](https://cloud.google.com/vertex-ai/docs)
- [BigQuery 最佳实践](https://cloud.google.com/bigquery/docs/best-practices)
- [Cloud Composer 指南](https://cloud.google.com/composer/docs)
- [DGL 图神经网络](https://www.dgl.ai/)
