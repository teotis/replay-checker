# replay_checker

Replay Checker 是一个 agent 评估工具——用真实的、历史发生过项目状况来检验 AI coding agent 的实际能力。

## 理念

**信证据，不信自述。** agent 声称"完成了"不等于真的完成了。Replay Checker 收集 diff、文件变更、验证命令输出等可观测证据，而非采信 agent 的自我报告。

**重结果，轻过程。** 默认评分权重——结果 80%，过程 20%。代码是否跑通、测试是否通过、边界是否覆盖，比 agent 的思考过程更重要。

**用历史说话。** 不从零构造玩具任务。Replay Checker 从你真实项目的 plan 文档、git 历史和本地会话记录中重建可重放的任务场景，让评估有据可查。

**平台中立。** 不绑定任何特定 agent 平台。你手动将任务包复制到 Claude Code、Codex、或任何接受 prompt 的平台。工具只准备契约，不替你决策。

**默认本地，按需触网。** 不用外部 LLM 也能完成 case 生成——所有分析默认为本地运行。LLM 调用需显式 opt-in，上传范围严格受限。

**执行与评分隔离。** 执行包（给被评估 agent）不含参考答案或评分标准。评分包匿名化 runner 身份。Oracle 数据对执行 agent 永远不可见。

## 安装

```bash
pip install -e .
```

Python >= 3.9，零运行时依赖。

## 快速开始

```bash
# 从任意项目目录自动生成一个 replay case
python3 tools/replay.py intake --project /path/to/project

# 为评估目标 agent 准备隔离工作区
python3 tools/replay.py prepare-run --case <case-id> --label "<model-name>"

# 将 runs/<run-id>/TASK.md 交给 agent 执行
# agent 在 runs/<run-id>/workspace 中工作
# 完成后写入 runs/<run-id>/completion_report.md

# 收集证据并评分
python3 tools/replay.py collect-run --run <run-id>
python3 tools/replay.py score-run --run <run-id>

# 比较同 case 下多次运行
python3 tools/replay.py compare --case <case-id>
```

## 3.0 — 自然语言 Case 发现

用自然语言描述范围，自动生成编排工具包：

```bash
python3 tools/replay.py wizard --project /path/to/project \
  --scope "authentication refactor and testing improvements"

bash <kit>/launchers/orchestrate.sh start
bash <kit>/launchers/orchestrate.sh status
```

## 证据门控

六项证据门控确保评分的最低证据质量。失败的门控触发评分上限或标记为无效：

| 门控 | 失败时限制 |
|---|---|
| 非空 diff.patch | 结果分 → 0 |
| completion_report.md 存在 | 过程分 → 0 |
| 有文件变更记录 | 结果分 → 0 |
| 验证步骤被引用 | 验证分 → 0 |
| 未访问 oracle 数据 | 总分 → invalid |
| 未暴露 runner 身份 | 总分 → invalid |

## 命令总览

| 命令 | 用途 |
|---|---|
| `wizard` | 自然语言生成发现工具包 |
| `intake` | 自动生成 case |
| `prepare-run` | 准备隔离执行包 |
| `collect-run` | 收集运行证据 |
| `score-run` | 生成匿名评分包 |
| `lint-task` / `lint-score` | 验证执行/评分包结构 |
| `compare` | 多次运行对比报告 |
| `inspect-sources` / `inspect-candidates` | 预览证据和候选 |

## 许可证

MIT License
