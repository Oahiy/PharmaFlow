# Task 5 评测运行手册

评测集包含 12 条用例，分别覆盖 6 条单工具、3 条多跳和 3 条异常场景。运行器为每轮生成 `run_id` 和 `user_id`，为每条用例生成独立的 `thread_id` 与 Langfuse trace ID。Agent 最多调用模型 10 次。两个 MCP 会话使用独立的 stdio 子进程，lit-mcp 固定从 `data/lit_recordings` 回放。

## 配置 Langfuse 凭据

打开你的 Langfuse Cloud 项目。在项目的 Settings → API Keys 创建一对项目密钥，把 Public Key、Secret Key 和项目所在区域的 Base URL 填进项目根目录的 `.env`。复用已有密钥时，确认两项来自同一项目。Secret Key 只在创建时显示一次。 [Langfuse 凭据设置](https://langfuse.com/docs/observability/get-started) · [Secret Key 说明](https://langfuse.com/faq/all/lost-secret-key)

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENABLED=true
```

上面的 Base URL 是欧区示例，按项目所在区域填写地址。每次正式评测会在创建 run 前校验凭据。缺少任一 key 时会退出，终端不会生成可供对比的 run ID。 [.env.example](../.env.example) · [Cloud 区域地址](https://langfuse.com/docs/api-and-data-platform/features/public-api)

## 跑两轮正式基线

在项目根目录准备 Python 3.12 和锁定依赖。在本地配置 `DATABASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`。Agent 和 judge 使用相同的模型与端点配置，各自发起模型调用。基线使用 `deepseek-v4-pro` 和 `https://api.deepseek.com`，温度为 0，关闭 thinking。

```sh
uv sync --locked
docker compose up -d postgres

LLM_MODEL=deepseek-v4-pro LLM_BASE_URL=https://api.deepseek.com \
  LLM_DISABLE_THINKING=true uv run python -m scripts.check_model
```

基线固定使用 `function_calling`。先确认探针中的 Pydantic 结构化输出检查通过。若该项显示「函数调用被拒绝，JSON 模式验证通过」，正式基线应等到统一确认两轮的 judge 模式后开始。探针在这个分支会以非零状态退出。

回放预检会按 YAML 中的约束实际调用工具。正向 PubMed 检索包含 ESearch 和 EFetch，零结果检索以成功响应校验。当前用例覆盖所需的 9 条录制已随仓库提供。

```sh
uv run --locked --no-env-file pytest tests/integration/test_eval_replay.py
```

将评测实现、用例和录制文件纳入同一个 Git 提交后，从该提交的干净工作区运行。用 `git status --short` 确认工作区干净，用 `git rev-parse HEAD` 获取两轮应记录的 commit。

配好 Langfuse 凭据后，用探针验证的模型和端点跑两次。第一轮命令成功退出后才会启动第二轮。每次保存终端的 `run_id` 和汇总表。

```sh
uv run python -m eval.run --tag v2 &&
  uv run python -m eval.run --tag v2
```

运行器逐条提交结果，并在所有计划用例均已入库后填写 `finished_at`。Agent 或 judge 的模型服务出现鉴权失败、权限错误、限流、连接失败、超时或 5xx 时，客户端抛错后运行中止。中断后用 `--summary` 查看已入库结果，运行状态显示未完成。将变量 `RUN_ID` 设为目标运行的 UUID 后执行。

```sh
uv run python -m eval.run --summary "$RUN_ID"
```

等两轮都完整结束，再执行下面的命令。按终端提示依次粘贴两次运行打印的 UUID，每个 ID 输入后按回车。对比要求用例摘要和录制摘要分别一致。逐用例差值为第 2 轮减第 1 轮。judge 分数和 token 任一侧为 `NULL` 时，相应差值为 `NULL`。成功状态按 0 和 1 求差，有 `replay_miss` 的用例成功差值显示 `NULL`。

```sh
printf '第一轮 run_id ' &&
  IFS= read -r FIRST_RUN_ID &&
  printf '第二轮 run_id ' &&
  IFS= read -r SECOND_RUN_ID &&
  uv run python -m eval.compare "$FIRST_RUN_ID" "$SECOND_RUN_ID"
```

两轮完成后，记录 `v2` 的提交 SHA、run ID 和对比结果。S04 和 S06 的规则修订会改变用例摘要，现有对比命令仅支持相同摘要的两轮。

## v1 工作区实测基线

2026 年 11 月 26 日对同一套 12 条自建用例运行两轮。Agent 和 judge 均使用 `deepseek-v4-pro`、温度 0、关闭 thinking。Agent 的调用上限为 10 次，lit-mcp 使用 replay，judge 使用 `function_calling`，通过阈值为 3 分。两轮的 Git HEAD 为 `335b66c`，均记录 `git_dirty=true`，包含当时工作区的 Task 5 改动。

| 轮次 | Run ID | 开始和结束时间（UTC+8） |
| --- | --- | --- |
| 第 1 轮 | `b199bdb4-c3b3-4bdc-a6bc-5486e39c0794` | 10:20:10–10:21:43 |
| 第 2 轮 | `a13afd33-6287-4d65-9ac9-7be91dc73dab` | 10:21:47–10:23:25 |

两轮完整结束，配置一致。用例摘要为 `7fe0f3cd5e1e63e5cae58ae387a79f71a3bd1b24572862469b8a59099a5db214`，录制摘要为 `b1c114e32203fdb72535fea39124a374d5008dc2d6156401efda1911ecf20a9e`。

Langfuse Cloud 验收通过。用户确认可在对应项目打开 trace，查看模型与工具记录。

### SQL 汇总

| 指标 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 计划数 / 已入库数 / 可评分数 | 12 / 12 / 12 | 12 / 12 / 12 |
| 回放未命中 / 缺 token | 0 / 0 | 0 / 0 |
| 工具选择准确率 | 100%（12/12） | 100%（12/12） |
| 参数合法率 | 100%（11/11） | 100%（11/11） |
| 任务成功率 | 83.3%（10/12） | 83.3%（10/12） |
| 平均输入 token | 5,724.17 | 5,724.33 |
| 平均端到端耗时（ms） | 6,559.2 | 7,122.3 |

两轮 E02 都直接指出 NCT 格式问题，参数合法记 `NULL`。启用 judge 的 E03、M02、M03 在两轮均为 3 分。平均耗时第二轮增加 563.2 ms，单例输入 token 只有 S06 增加了 2。

| 类别 | 任务成功（两轮相同） | 平均输入 token 1 → 2 | 平均耗时 1 → 2（ms） |
| --- | ---: | ---: | ---: |
| 单工具 | 4/6 | 5,327.67 → 5,328.00 | 5,475.8 → 5,728.3 |
| 多跳 | 3/3 | 8,935.67 → 8,935.67 | 11,310.7 → 12,755.0 |
| 异常 | 3/3 | 3,305.67 → 3,305.67 | 3,974.3 → 4,277.7 |

### 逐用例结果

| 用例 | 判分（两轮相同） | Judge（两轮相同） | 输入 token 1 → 2 | 耗时 1 → 2（ms） | 耗时差（ms） |
| --- | --- | ---: | ---: | ---: | ---: |
| E01 | 通过 | NULL | 4,002 → 4,002 | 3,025 → 4,384 | +1,359 |
| E02 | 通过 | NULL | 1,869 → 1,869 | 2,344 → 2,214 | −130 |
| E03 | 通过 | 3 | 4,046 → 4,046 | 6,554 → 6,235 | −319 |
| M01 | 通过 | NULL | 8,490 → 8,490 | 10,330 → 11,343 | +1,013 |
| M02 | 通过 | 3 | 9,632 → 9,632 | 8,364 → 10,701 | +2,337 |
| M03 | 通过 | 3 | 8,685 → 8,685 | 15,238 → 16,221 | +983 |
| S01 | 通过 | NULL | 3,891 → 3,891 | 4,561 → 3,658 | −903 |
| S02 | 通过 | NULL | 8,621 → 8,621 | 3,629 → 4,602 | +973 |
| S03 | 通过 | NULL | 5,090 → 5,090 | 3,208 → 3,342 | +134 |
| S04 | `rule_failed` | NULL | 4,720 → 4,720 | 6,450 → 7,244 | +794 |
| S05 | 通过 | NULL | 4,591 → 4,591 | 5,877 → 5,939 | +62 |
| S06 | `rule_failed` | NULL | 5,053 → 5,055 | 9,130 → 9,585 | +455 |

### 失败复核

- **S04** 两轮都答对总命中数 25、返回数 5 和前两项 NCT 号。回答随后附了完整列表。v1 的序列规则要求全文只包含这两项，额外编号使规则失败。修订后验证前两项的顺序，并允许后附完整列表。 [S04 当前规则](../eval/cases/S04.yaml)
- **S06** 两轮都答对 set ID、生效日期以及两个章节的有无。章节名后的中文括注和表格里的勾号使 v1 的存在性正则匹配失败。第二轮还在表格中写「内容完整返回」，页末又说内容被截断。工具记录的 `truncated=true` 支持页末表述。修订后的规则接受常见表格格式，并检查明确的完整返回断言。 [S06 当前规则](../eval/cases/S06.yaml)

| 用例 | 第 1 轮 trace ID | 第 2 轮 trace ID |
| --- | --- | --- |
| S04 | `b0dca21e3b0395c04955423add068991` | `34eeb44ebcaa6c2a4f478f87ab28c91b` |
| S06 | `7cbbcef754ba40a138c36cf1b0017885` | `cea97ed3bbe982328e6f945882bfec88` |

## 指标

Python 负责单例判分。Postgres 按整轮和类别分别汇总，每组显示计划数、已入库数、可评分数、回放未命中数和缺 token 数。计划清单来自运行启动时的快照。

| 指标 | 口径 |
| --- | --- |
| 工具选择准确率 | 工具名称、次数和依赖顺序符合某条接受路径的用例数，除以已入库数。 |
| 参数合法率 | 参数合法的用例数，除以参数可判定的用例数。没有可判定调用时记 `NULL`。与 YAML 相符的预期工具错误可通过。 |
| 任务成功率 | 工具路径与参数符合预期，规则全部通过，启用 judge 时分数达到阈值的用例数，除以可评分数。可评分数等于已入库数减去 `replay_miss` 数。 |
| 平均输入 token | 对输入 token 非空的用例求平均。单例累加所有 Agent 模型调用的 `AIMessage.usage_metadata.input_tokens`。如有调用失败或缺 usage，整例记 `NULL`。Judge 调用独立计费。 |
| 平均端到端耗时 | 对已入库用例求平均。从提交问题给图前到图返回最终状态或抛出异常为止，用单调时钟计毫秒。包含 Agent、工具和检查点耗时。规则、judge 和评测写库在计时结束后执行。 |

分母为零时显示 `NULL`。答案文本的正则搜索忽略大小写，序列规则逐项精确比较，以保留 SMILES 等标识的大小写。

Judge 按覆盖要求、证据可追溯、口径正确三个布尔项打分，每项 1 分，基线阈值为 3 分。Judge 与 Agent 使用同一模型，评分可能受共同偏好影响。计入单例的 judge 调用错误和输出解析错误保留错误详情，分数记 `NULL`。

同例有多个错误时依次取第一个成立的失败类型。

1. `replay_miss`
2. `invalid_args`
3. `agent_error`
4. `step_limit`
5. `tool_error`
6. `wrong_tool`
7. `rule_failed`
8. `judge_call_error`
9. `judge_parse_error`
10. `judge_below_threshold`

只有满足预期工具及参数约束的调用才能记为 `replay_miss`。真实零结果和分子批次中的 `invalid_smiles` 属于成功工具响应。计入单例的异常保存类型、消息和 traceback。数据库、MCP 会话和明确的模型服务故障会中止运行。

## Postgres 表

| 表 | 列与键 |
| --- | --- |
| `eval.runs` | `run_id uuid` 主键，`tag text`，`git_commit text`，`user_id uuid` 唯一，`config jsonb`，`started_at timestamptz`，`finished_at timestamptz` 可空。 |
| `eval.cases` | `case_id text` 主键，`category text` 取 `single_tool`、`multi_hop`、`exception`。每次运行从 YAML 同步。 |
| `eval.results` | `run_id uuid` 与 `case_id text` 为联合主键，分别引用前两张表。`thread_id uuid` 唯一，`trace_id text` 为 32 位十六进制，`success boolean`、`tools_correct boolean`、`params_valid boolean` 可空、`judge_score smallint` 可空、`failure_type text` 可空、`error_detail jsonb` 可空、`input_tokens bigint` 可空、`elapsed_ms bigint`。 |

失败类型、类别、分数范围、计数与耗时均由 SQL 约束。运行配置记录模型、端点、温度、thinking、步数、回放设置、judge 细则版本与阈值、Git commit、工作区状态、用例清单及两份 SHA-256 摘要。摘要对各文件的相对路径和原始内容计算。端点记录省略 URL 中的用户信息、查询参数和片段，配置中无密钥。 [建表 SQL](../eval/schema.sql) · [汇总 SQL](../eval/summary.sql)

`schema.sql` 可重复初始化。已有表新增列或变更具名 `CHECK` 约束时，需执行对应的 `ALTER TABLE` 迁移。

## 打开 Langfuse trace

正式评测首先检查追踪已开启、两个 key 已填写，随后调用 `auth_check()`。退出时用同一个默认客户端 flush。API 默认也在启动时验证凭据，关闭 API 追踪可设置 `LANGFUSE_TRACING_ENABLED=false`。两者的 session 都对应 `thread_id`。

API trace 带 API 版本标签。代码版本优先使用 Langfuse SDK 的 `LANGFUSE_RELEASE`。镜像构建时可将源码提交 SHA 注入这个变量。本地 Git 工作区也可以直接提供 HEAD。没有版本来源时，API 正常启动并使用 API 版本标签。

查某轮的 trace ID 时，先进入 Postgres。

```sh
docker compose exec postgres psql -U pharmaflow -d pharmaflow
```

在 psql 中把示例 UUID 换成实际的 run ID。

```sql
\set run_id '00000000-0000-0000-0000-000000000000'
PREPARE find_eval_traces(uuid) AS
    SELECT case_id, thread_id, trace_id, failure_type
    FROM eval.results
    WHERE run_id = $1
    ORDER BY case_id;
EXECUTE find_eval_traces(:'run_id');
```

在相同的 Cloud 项目打开 `<LANGFUSE_BASE_URL>/project/<project_id>/traces/<trace_id>`，逐条核对 Agent 的模型与工具 observation。失败的用例也会保存预分配的 trace ID，Cloud 验收以页面能实际打开且包含 observation 为准。
