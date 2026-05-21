# Cozy PDE v3 设计规格

## 文档状态

- 状态：已确认设计，待实现计划
- 日期：2026-05-21
- 存放位置说明：按用户要求，第三版设计规格存放在 `dev_docs/`，不写入 `docs/`

## 1. 目标与边界

Cozy PDE v3 的目标是把当前系统重构为一个由 Python 程序主导的科研状态机，而不是由 LLM 驱动的长对话执行器。第三版必须严格按照 `dev_docs/pde_agent_implementation_plan.md` 的方向完成整体重构，并吸收当前会话中已经确认的补充约束。

第三版的核心定义为：

`Responses-only transport + native API tool calling + deterministic Python router + structured state + local memory store + cache-first context packer + deterministic experiment engine + provenance-safe packaging`

第三版的上层语义必须统一为 Responses 协议，不允许 provider 差异、proxy 细节、fallback 细节、research 细节反向污染 `agent_loop`。

本次重构的边界如下：

- 允许按文档直接重组模块、入口和命名，不要求沿用第二版形式兼容。
- 第三版只支持单一 task 正式运行，不支持多任务共享 session，也不支持串行多个任务。
- 保留 `DeepSeek` 作为 Responses 备用通道，但它只是 provider failover，不是第二套上层协议。
- 保留 `scripts/proxy.py`，并要求其尽早通过第三版验收。
- 保留第二版中 `E/F` 两类能力：
  - `E`：日志审计、可追溯、导出、监控、校验
  - `F`：本地 research cache / search / fetch / parse 能力
- 删除第二版中 `B/D` 两类能力：
  - `B`：`JsonActionClient`、`chat.completions` 兼容层、LLM router、json_action 协议及其测试壳
  - `D`：`autonomous_dry_run`、`autonomous_rehearsal`、rehearsal profile 与其相关工作流
- 不迁移第二版的旧 mode 集合；第三版使用新的 CLI 语义。

本次重构还包括实验调度与任务策略的重构，不仅限于 transport、router、memory、logging 基础设施。

## 2. 非目标

以下内容不属于第三版设计目标：

- 不保留 `chat.completions` 作为任何正式或过渡主路径。
- 不保留 `json_action` 作为任何上层动作协议。
- 不保留 LLM 作为 router、shell 编排器、最终 gate 决策器。
- 不保留多任务 formal session、共享日志导出、多任务串行正式运行。
- 不允许在 `methodology` 生成阶段让 LLM 自由发挥、补写事实或美化到与运行记录不一致。

## 3. 总体架构

第三版采用新核心包，不继续以当前 `agent_runner/main.py` 为系统中心。推荐模块结构如下：

```text
cozy_pde/
├── cozy_pde_v3/
│   ├── __init__.py
│   ├── cli.py
│   ├── agent_loop.py
│   ├── responses_client.py
│   ├── responses_ledger.py
│   ├── tool_registry.py
│   ├── deterministic_router.py
│   ├── state.py
│   ├── memory_store.py
│   ├── context_packer.py
│   ├── experiment_engine.py
│   ├── profiles.py
│   ├── prompts.py
│   ├── logging.py
│   ├── proxy_logs.py
│   ├── provider_capabilities.py
│   ├── research/
│   │   ├── cache.py
│   │   ├── tools.py
│   │   └── providers.py
│   ├── validation/
│   │   ├── submission.py
│   │   ├── logs.py
│   │   └── provenance.py
│   └── tools/
│       ├── fs_tools.py
│       ├── hdf5_tools.py
│       ├── python_tools.py
│       ├── shell_tools.py
│       ├── package_tools.py
│       └── document_tools.py
├── scripts/
│   └── proxy.py
└── agent_runner/
```

架构原则如下：

- `cozy_pde_v3` 是第三版产品本体。
- `responses_client.py` 是唯一 LLM 传输入口。
- `agent_loop.py` 只消费第三版标准化后的 Responses items。
- provider 主备、proxy、raw response、research、provenance 是下层能力，不得反向泄漏到上层协议。
- 第二版 `agent_runner` 在迁移完成后只允许保留极薄兼容壳，若无必要应删除。

## 4. Responses 协议与 Provider 适配

### 4.1 统一协议要求

第三版只允许使用 `client.responses.create`。

明确禁止：

- `client.chat.completions.create`
- 任何 chat-style `role="tool"` 协议模拟
- `JsonActionClient`
- 任何 `json_action` 输出约束
- “先转消息，再模拟 Responses”的兼容层

上层模块只认第三版统一 turn 结构，建议为：

```python
@dataclass
class ResponsesTurn:
    provider: str
    model: str
    raw_response: dict[str, Any]
    provider_output_items: list[dict[str, Any]]
    standard_output_items: list[dict[str, Any]]
    usage: dict[str, Any]
    failover_from: str | None
    failover_reason: str | None
```

约束如下：

- `provider_output_items` 保存供应商原始 item，作为证据层数据。
- `standard_output_items` 保存 Cozy PDE 归一化后的标准 item，作为执行层数据。
- `raw_response` 保留完整原始响应，用于 proxy/provenance/调试。
- `agent_loop`、`tool_registry`、`logging`、`memory_store`、`context_packer` 只能依赖 `standard_output_items` 及规范化的 usage 元数据。

### 4.2 Provider Adapter

`responses_client.py` 内部允许使用 provider adapter，但对外语义必须统一。

建议结构：

```text
ResponsesClient
├── PrimaryResponsesAdapter(aixj)
├── FallbackResponsesAdapter(deepseek)
└── FailoverPolicy
```

Provider adapter 的职责：

- 发送 Responses 请求
- 解析供应商原始响应
- 生成 `provider_output_items`
- 归一化为 `standard_output_items`
- 产出统一 `ResponsesTurn`

禁止让 provider adapter 对上层暴露各自的字段名、错误结构、工具调用差异。

## 5. DeepSeek 备用通道

### 5.1 角色定义

DeepSeek 是 `Responses` 语义下的备用 provider，不是第二条上层协议。

上层只看到：

- 同一种 `ResponsesTurn`
- 同一种 `function_call`
- 同一种 `function_call_output`
- 同一种 tool execution 与 ledger 语义

### 5.2 触发条件

主链路为 `aixj`。当且仅当满足以下 provider 级故障时，允许切到 `DeepSeek`：

- 请求超时
- 网络错误
- 网关故障：`502/503/504/524`
- 明确的额度耗尽、余额不足、credit exhausted、quota exceeded
- 明确表示主 provider 当前不可继续的限流状态

以下情况禁止触发 failover：

- 参数错误
- 工具 schema 错误
- 模型名错误
- 认证错误
- 本地代码 bug
- 已经完成工具调用决策后的本地运行时失败

### 5.3 不跨工具边界

Failover 只能发生在“模型请求尚未产出可执行 tool call 的响应边界内”。

允许：

- 主链路请求失败，尚未返回可执行 `function_call`

禁止：

- 主链路已经返回 `function_call`
- 本地执行工具失败
- 状态更新失败
- 工件冻结失败
- validation/provenance/package 失败

这些问题属于 tool/runtime/state/engine 问题，必须进入 failure recovery 或 diagnosis，不能通过切 provider 重判。

### 5.4 Fallback 门槛

`primary` 必须始终 `formal-ready`。

`fallback` 是否是启动硬门槛取决于配置：

- 当 `require_fallback=true` 时：fallback 必须 `formal-ready`，否则 `run` 拒绝启动
- 当 `require_fallback=false` 时：fallback 不阻断 `run`，但系统必须记录当前为 `degraded-but-runnable`

## 6. Proxy 兼容与日志要求

### 6.1 双轨 Proxy

第三版保持 `scripts/proxy.py`，并要求可分别记录主链路与备用链路的 Responses 原始请求和响应。

推荐部署方式：

```text
primary path:  cozy_pde_v3 -> proxy A -> aixj
fallback path: cozy_pde_v3 -> proxy B -> deepseek
```

日志路径建议：

```text
workspace/proxy_logs/
├── aixj/
│   └── llm-YYYYMMDD.jsonl
└── deepseek/
    └── llm-YYYYMMDD.jsonl
```

### 6.2 日志内容

Proxy 必须记录：

- 完整 request body
- 完整 raw response body
- `provider`
- `target`
- `request_id`
- `timestamp`
- 必要的路由元数据

若是 streaming，proxy 仍需在内部重组后落一份完整 raw response。

### 6.3 脱敏要求

Proxy 日志必须脱敏，但不能破坏 provenance 所需内容。

至少需要 redaction：

- API key
- `Authorization` header
- 环境变量敏感值
- shell output 中疑似 token 的内容
- 私有路径中的用户名

允许保留：

- LLM 输出内容
- 工具调用结构
- 研究摘要
- 代码生成内容
- 与 provenance 有关的必要上下文

### 6.4 早期验收要求

`scripts/proxy.py` 必须作为第三版早期验收项，在最小主链路刚建好时就通过以下检查：

- primary text probe
- primary function call probe
- primary `function_call_output` continuation probe
- strict schema probe
- primary proxy raw log 写入
- fallback 对应 probe
- forced failover probe
- 双轨 raw log 可分别导出并合并

## 7. CLI 设计

第三版只保留以下稳定入口：

```text
cozy-pde run
cozy-pde check-provider
cozy-pde check-research
cozy-pde validate
cozy-pde package
cozy-pde status
```

### 7.1 `run`

职责：

- 正式单任务运行入口
- 读取 config 和 task spec
- 执行 capability check 结果验证
- 初始化或恢复状态
- 启动 `AgentLoop`
- 在确定性 finalize gate 通过后进入 package

约束：

- 只允许单 task
- 不支持多任务共享 session
- 不支持串行多个任务

### 7.2 `check-provider`

职责：

- 验收 primary provider
- 验收 fallback provider
- 验收 proxy
- 验收 strict schema
- 验收 `function_call_output` continuation
- 验收 forced failover

报告路径固定为：

`workspace/capabilities/provider_report.json`

报告必须包含：

- `config_hash`
- `tool_schema_hash`
- `proxy_version_hash`
- `primary` 能力结果
- `fallback` 能力结果
- `forced_failover` 结果
- `formal_ready`
- `checked_at`

仅比较 `config_hash` 不够，因为工具 schema 或 proxy 变化后也必须重测。

### 7.3 `check-research`

职责：

- 验证 local research 工具链
- 验证 research cache
- 验证 arXiv / GitHub / fetch / parse 能力

### 7.4 `validate`

职责：

- 校验 submission bundle
- 校验 logs
- 校验 provenance
- 校验 inference time
- 校验 package readiness

要求：

- 纯确定性
- 不调用 LLM
- 输出结构化 gate 对象

### 7.5 `package`

职责：

- 读取最近一次 finalize gate
- gate 不通过则拒绝打包
- 合并主备 proxy raw logs
- 导出正式日志
- 生成 manifest / code manifest / methodology
- 冻结 submission snapshot
- 生成最终 zip

### 7.6 `status`

职责：

- 展示当前 phase
- 展示最近失败与 blocker
- 展示当前最佳工件
- 展示 finalize gate 差距

## 8. AgentState 设计

`AgentState` 只保存调度事实，不保存大段自然语言历史。

建议字段：

```text
run_id
task
mode
started_at
elapsed_seconds
remaining_seconds
current_phase
current_objective
latest_error_type
latest_error_summary
latest_error_at
latest_tool_name
latest_tool_result_ok
last_llm_call_id
last_tool_call_id
best_artifact_path
best_artifact_version
best_metrics
latest_metrics
latest_checkpoint_path
submission_snapshot_id
latest_submission_ready
experiments_total
experiments_failed
consecutive_failures
cache_hit_ratio_latest
cache_hit_ratio_rolling
fallback_status
capability_status
finalize_gate_status
```

设计原则：

- state 面向路由和执行决策
- 审计细节进入 memory 和 logs
- state 必须可以持久化与恢复
- `last_llm_call_id`、`last_tool_call_id`、`best_artifact_version`、`submission_snapshot_id` 作为 state、memory、log、artifact 的稳定外键

## 9. Router 设计

`deterministic_router.py` 是纯 Python 决策器，只读结构化 `AgentState`。

输出建议：

```text
phase
profile
allowed_tools
allow_hosted_research
requires_llm
deterministic_action
reason_code
```

优先级顺序要求：

1. capability readiness
2. preflight / data inspection
3. baseline guard
4. failure recovery
5. implementation
6. train / validate / benchmark
7. diagnosis
8. reflection
9. finalization

明确要求：

- `failure recovery` 的优先级高于新的 `implementation`
- router 不能在刚发生 OOM / shape mismatch 后直接又让 LLM 写新代码，而应先走确定性恢复

`reason_code` 必须机器可读，例如：

- `baseline_missing`
- `shape_mismatch`
- `cuda_oom`
- `quota_failover`
- `finalize_ready`

## 10. MemoryStore 设计

第三版采用本地 SQLite。

建议文件：

```text
workspace/memory/
├── memory.sqlite
├── current_state.json
├── semantic_facts.jsonl
├── failure_patterns.yaml
└── cache/
```

建议表：

- `experiments`
- `experiment_events`
- `semantic_memory`
- `failure_patterns`
- `research_sources`
- `code_artifacts`
- `cache_metrics`
- `decision_records`

### 10.1 `decision_records`

必须新增 `decision_records`，记录 router 的选择轨迹。

建议字段：

- `state_hash`
- `reason_code`
- `route`
- `selected_profile`
- `selected_phase`
- `selected_tools`
- `outcome`
- `created_at`

目的：

- 调试“为什么 agent 一直绕圈”
- 回看不同 state 下 router 的路径选择
- 为后续规则修正提供证据

### 10.2 记忆职责

- `experiments`：一次实验的假设、配置、结果、结论
- `experiment_events`：训练、失败、回退、验证、冻结最佳工件等事件
- `semantic_memory`：跨实验抽象经验
- `failure_patterns`：症状 -> 诊断 -> 恢复动作 -> 预防检查
- `research_sources`：research 来源、摘要、缓存路径、许可信息
- `code_artifacts`：代码文件与生成调用的映射
- `cache_metrics`：prompt caching 指标
- `decision_records`：路由决策与结果

## 11. ContextPacker 设计

`ContextPacker` 只负责构造 cache-first 的最小上下文，不允许退化成长历史拼接器。

固定输出 6 段：

1. `developer_contract`
2. `task_spec`
3. `phase_tool_policy`
4. `compact_state`
5. `retrieved_memory`
6. `current_request`

### 11.1 Token Budget 硬限制

必须有硬 token budget。

建议上限：

- `retrieved_memory <= 1500 tokens`
- `log_summary <= 1000 tokens`
- `code_excerpt <= 5000 tokens`

超限时必须裁剪、压缩或改为引用 path/hash/summary，不允许无限扩展。

### 11.2 禁止项

禁止放入：

- 完整历史 prompt
- 完整训练日志
- 完整 shell stdout
- 与当前问题无关的长代码片段
- 多轮无关 reasoning

## 12. ExperimentEngine 设计

`experiment_engine.py` 是第三版执行闭环的核心。它负责：

- baseline guard
- preflight validator
- smoke check
- train
- validate
- benchmark
- failure recovery
- best artifact freeze
- finalization gate

建议状态迁移：

```text
init
-> capability_check
-> inspect_data
-> baseline_guard
-> planning_or_execute
-> implementation
-> smoke_check
-> train
-> validate
-> benchmark
-> reflect
-> recover_or_continue_or_finalize
```

关键规则：

- 每个 task 必须尽早得到可提交 baseline
- 正式训练前必须先做 smoke check
- 对 `loss nan / OOM / shape mismatch / inference timeout` 优先走规则恢复
- 连续恢复失败后才进入 LLM diagnosis
- 最佳工件必须显式冻结并版本化

## 13. LLM 使用边界

第三版中 LLM 只负责以下高价值认知节点：

- 方案生成
- 代码生成
- 无法规则化的失败诊断
- 实验反思

第三版中 LLM 不再负责：

- router
- shell 编排
- 记忆管理
- 简单失败恢复
- finalization 判定
- package 判定

## 14. E/F 能力迁移

### 14.1 E：日志审计与可追溯

保留能力，但迁移到新模块：

- `validation/logs.py`
- `validation/provenance.py`
- `logging.py`
- `proxy_logs.py`

保留内容：

- JSONL / Responses logs 校验
- proxy logs 合并导出
- code manifest / manifest
- secret leak scan
- run status 汇总

### 14.2 F：research

保留能力，但迁移到新模块：

- `research/cache.py`
- `research/tools.py`
- `research/providers.py`

保留内容：

- `ResearchCache`
- `search_arxiv`
- `search_github`
- `fetch_url`
- `fetch_pdf`
- `parse_pdf`
- `parse_html`

Research 结果必须进入 memory 和 cache，只允许以摘要和引用形式进入上下文。

## 15. Finalize Gate

`finalize_gate_status` 不得是单个 bool，必须是结构化对象。

建议字段：

- `prediction_ok`
- `time_csv_ok`
- `logs_ok`
- `provenance_ok`
- `inference_time_ok`
- `package_ok`
- `code_manifest_ok`
- `methodology_ok`
- `secret_scan_ok`
- `task_rule_ok`
- `overall_ok`
- `failures`
- `warnings`

要求：

- `overall_ok` 只是派生值
- formal finalization 必须按子 gate 规则执行
- finalization 不允许依赖 LLM 判断

## 16. Methodology 生成

第三版的 `methodology` 必须由结构化运行记录机械生成。

输入来源：

- `AgentState` snapshots
- `decision_records`
- `experiment_cards`
- `validation reports`
- `artifact metadata`
- `final package snapshot`

要求：

- 默认机械生成
- 只根据结构化记录产出
- 不允许自由补写事实
- 不允许与 logs 不一致
- 若后续引入 LLM，只允许做不改变事实的语言压缩

`methodology` 的本质是“结构化运行记录的人类可读视图”，不是赛末自由总结。

## 17. Capability Check 设计

Provider capability check 是正式运行前的重要门槛。

探针最少覆盖：

- text response
- single function call
- `function_call_output` continuation
- strict schema
- proxy raw log

此外需要：

- primary probes
- fallback probes
- forced failover probe

若 fallback 被要求为正式门槛，还必须显式通过同一组 probe，否则不能作为 formal fallback。

## 18. 验收矩阵

### 18.1 Provider / Proxy / Failover

- primary text probe 通过
- primary function call probe 通过
- primary continuation probe 通过
- primary strict schema probe 通过
- primary proxy raw log 通过
- fallback 对应 probe 通过
- forced failover 仅发生在合法边界
- `provider_report.json` 完整生成

### 18.2 Responses 协议一致性

- 仓库中无 `chat.completions`
- 仓库中无 `json_action`
- tool output 一律通过 `function_call_output`
- `provider_output_items` 与 `standard_output_items` 双层保留

### 18.3 State / Memory / Router / Engine

- state 可恢复
- `decision_records` 可追踪
- `failure recovery > implementation`
- finalize gate 结构化
- engine 能覆盖主路径与恢复路径

### 18.4 E/F

- `validate` 能校验 logs / submission / provenance / inference time
- `package` 能导出合并后的 proxy provenance
- `status` 能展示当前 blocker
- `check-research` 能走通本地 research 能力

### 18.5 Methodology / Provenance

- methodology 仅来自 structured records
- 不出现日志中不存在的实验或结论
- 能回指 experiment / decision / artifact

## 19. 迁移策略

建议迁移顺序：

### 阶段 1：建新核

- 创建 `cozy_pde_v3`
- 建立核心模块骨架
- 不接旧 CLI

### 阶段 2：打通最小主链路

- 打通 primary Responses 文本、tool call、continuation
- 升级 `scripts/proxy.py`
- 实现 `check-provider`

### 阶段 3：接入 fallback 与 E/F

- 接入 DeepSeek provider adapter
- 接入 provider capability report
- 迁移 validate/package/proxy logs/research 工具

### 阶段 4：接入状态机闭环

- 实现 AgentState、Router、Engine、Memory、ContextPacker
- 跑通单 task formal run

### 阶段 5：切默认入口并删旧层

- 启用第三版 CLI
- 删除第二版协议层、dry run、rehearsal、multi-task 逻辑与对应测试壳

## 20. 删除矩阵

明确删除：

- `JsonActionClient`
- `chat.completions` 兼容路径
- LLM router
- `json_action` 协议相关测试壳
- `autonomous_dry_run`
- `autonomous_rehearsal`
- rehearsal profile
- 多任务 formal session / 多任务导出 / 串行多任务运行

保留但迁移：

- 日志与 provenance 校验能力
- proxy logs 合并导出
- research cache 与 research tools

## 21. 风险与控制

主要风险：

- provider Responses 兼容性细节差异
- fallback 名义可用但 formal 能力不完整
- proxy 原始记录不完整或未脱敏
- context packer 退化为长上下文
- methodology 生成与真实运行记录脱节

控制方式：

- provider capability check 前置
- 双层 item 保留证据
- proxy 早验收
- token budget 硬限制
- methodology 结构化机械生成

## 22. 最终结论

Cozy PDE v3 必须被实现为一个单任务、Responses-only、Python 主导的科研状态机。它的上层只认统一 Responses 语义；provider 主备、proxy、research、provenance、fallback、raw response 都必须沉到下层实现。`DeepSeek` 是统一协议下的 formal fallback，`scripts/proxy.py` 是正式 provenance 链的重要组成部分，`methodology` 必须由结构化运行记录机械生成。

这一设计文档批准后，下一步是基于本规格编写第三版 implementation plan，并据此执行重构。
