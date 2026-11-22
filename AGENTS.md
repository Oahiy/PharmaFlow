# AGENTS.md

## 项目

PharmaFlow Agent：围绕 KRAS G12C 靶点的多智能体系统，做三件事：临床安全性证据核查、候选分子类药性筛选、人工审批下的分子改造。

完整计划在 `docs/plan.md`：
- 第 4 节：架构和目录结构
- 第 5 节：关键技术决定，写代码时必须遵守
- 第 6 节：按 Task 拆分的开发计划，每个 Task 都有目标、做法、测试、演示

## 工作方式

- 用中文回复。
- 一次只做用户指定的一个 Task。开始前先读 `docs/plan.md` 里这个 Task 的全部内容和第 5 节。
- 计划没写清楚的设计选择，先问再做。需要偏离计划时，先说明原因。
- 不要执行 `git commit`、`git push`、`git reset --hard`。改动留给用户审查后提交。
- 做完后列出：改动的文件、新增的测试、跑过的命令和结果、没能运行的命令。

## 命令

- 安装依赖：`uv sync`。加依赖用 `uv add`，不用 pip。
- 单测：`uv run pytest tests/unit`
- 集成测试（Task 3 起，需要 Postgres）：先 `docker compose up -d postgres`，再 `uv run pytest tests/integration`
- 检查和格式化：`uv run ruff check .`、`uv run ruff format .`

## 代码约定

- Python 3.12。
- MCP 工具的输入输出用 Pydantic 模型定义，docstring 写清楚用途、参数和返回值。
- PubMed、ClinicalTrials.gov、openFDA 只通过 lit-mcp 的录制回放层访问，其他地方不直接请求这些接口。
- DeepSeek 调用一律关闭 thinking：`extra_body={"thinking": {"type": "disabled"}}`。
- MCP 客户端用 `client.session()` 开长期会话，不用 `get_tools()` 的默认行为。stdio 子进程需要的环境变量在 `env` 里显式传。
- State 字段标注具体类型，不把自定义对象放进 `Any` 或 `dict` 字段，因为检查点开启了严格反序列化。
- 模型不直接生成 SMILES，分子由 RDKit 工具枚举和校验。

## 测试规则

- 每个功能都按对应 Task 的"测试"一项写测试。
- 不许为了让测试通过而修改或删除已有断言。测试失败先查代码。
- 单测用假模型和 replay 数据，不调用真实 LLM，不联网。
- 连 Postgres 的测试用随机的 `thread_id` 和 `user_id`。

## 安全

- 不读、不写、不打印 `.env`。不把密钥写进代码、日志或测试数据。新增配置项时同步更新 `.env.example`。
- `data/uploads/` 不进仓库。录制的 HTTP 响应和 `data/candidates.csv` 要提交，CI 的 replay 测试依赖它们。

## 环境限制

Codex 的沙盒默认不联网。需要联网或 Docker 的命令跑不了时，不要想办法绕过，把命令列出来交给用户运行。这类命令包括 `uv sync`、`uv add`、录制 HTTP、调用 DeepSeek、`docker compose`，以及代码沙盒（Task 6）的测试。
