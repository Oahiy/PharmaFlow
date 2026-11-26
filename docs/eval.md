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

等两轮都完整结束，再执行下面的命令。按终端提示依次粘贴两次运行打印的 UUID，每个 ID 输入后按回车。对比要求用例摘要和录制摘要分别一致。系统提示词和工具定义摘要各显示「一致」「不同」或「缺少记录」，支持比较改动前后的模型输入。逐用例差值为第 2 轮减第 1 轮。judge 分数和 token 任一侧为 `NULL` 时，相应差值为 `NULL`。成功状态按 0 和 1 求差，有 `replay_miss` 的用例成功差值显示 `NULL`。

```sh
printf '第一轮 run_id ' &&
  IFS= read -r FIRST_RUN_ID &&
  printf '第二轮 run_id ' &&
  IFS= read -r SECOND_RUN_ID &&
  uv run python -m eval.compare "$FIRST_RUN_ID" "$SECOND_RUN_ID"
```

两轮完成后，记录 `v2` 的提交 SHA、run ID 和对比结果。S04 和 S06 的规则修订会改变用例摘要，现有对比命令仅支持相同摘要的两轮。

## v2 正式基线

2026 年 11 月 26 日在提交 `802e4b465c3764a927f66a6de6ea0c17df530a24` 的干净工作区运行两轮。两轮均记录 `git_dirty=false`。同一套 12 条自建用例固定使用 replay。Agent 与 judge 均为 `deepseek-v4-pro`、温度 0、关闭 thinking。Agent 调用上限为 10 次，judge 使用 `function_calling`，阈值为 3 分。

| 轮次 | Run ID | 开始和结束时间（UTC+8） |
| --- | --- | --- |
| 第 1 轮 | `78f28e64-2c5c-4bdf-b4ee-1f532d9cd6d5` | 11:26:52–11:28:30 |
| 第 2 轮 | `84dfc269-4256-45fd-8c81-de7c5ec198b9` | 11:28:33–11:30:08 |

两轮完整结束，配置一致，用例摘要为 `54e9f689f88c9704189d918bd1a407824cd341b625115645789215938bdf3968`，录制摘要为 `b1c114e32203fdb72535fea39124a374d5008dc2d6156401efda1911ecf20a9e`。两份摘要均与提交时的文件匹配。v1 和 v2 的用例摘要不同，各自按当时规则报告。

### SQL 汇总

| 指标 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 计划数 / 已入库数 / 可评分数 | 12 / 12 / 12 | 12 / 12 / 12 |
| 回放未命中 / 缺 token | 0 / 0 | 0 / 0 |
| 工具选择准确率 | 100%（12/12） | 100%（12/12） |
| 参数合法率 | 100%（11/11） | 100%（11/11） |
| 任务成功率 | 100%（12/12） | 100%（12/12） |
| 平均输入 token | 5,723.08 | 5,723.00 |
| 平均端到端耗时（ms） | 6,983.6 | 6,924.1 |

E02 两轮都直接指出 NCT 格式错误，参数合法记 `NULL`。E03、M02 和 M03 两轮 judge 均得 3 分。每轮 12 条结果均有 trace ID。

| 类别 | 任务成功（两轮相同） | 平均输入 token 1 → 2 | 平均耗时 1 → 2（ms） |
| --- | ---: | ---: | ---: |
| 单工具 | 6/6 | 5,324.83 → 5,324.83 | 5,807.8 → 5,622.8 |
| 多跳 | 3/3 | 8,937.00 → 8,936.67 | 11,993.7 → 12,306.3 |
| 异常 | 3/3 | 3,305.67 → 3,305.67 | 4,325.0 → 4,144.3 |

### 逐用例对比

12 条用例两轮均通过，judge 分数一致。差值为第 2 轮减第 1 轮。

| 用例 | Judge（两轮相同） | 输入 token 1 → 2 | token 差值 | 耗时 1 → 2（ms） | 耗时差（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| E01 | NULL | 4,002 → 4,002 | 0 | 3,370 → 3,366 | −4 |
| E02 | NULL | 1,869 → 1,869 | 0 | 2,620 → 2,638 | +18 |
| E03 | 3 | 4,046 → 4,046 | 0 | 6,985 → 6,429 | −556 |
| M01 | NULL | 8,490 → 8,490 | 0 | 10,937 → 12,426 | +1,489 |
| M02 | 3 | 9,632 → 9,636 | +4 | 9,318 → 9,253 | −65 |
| M03 | 3 | 8,689 → 8,684 | −5 | 15,726 → 15,240 | −486 |
| S01 | NULL | 3,891 → 3,891 | 0 | 3,674 → 3,342 | −332 |
| S02 | NULL | 8,602 → 8,602 | 0 | 5,039 → 4,079 | −960 |
| S03 | NULL | 5,090 → 5,090 | 0 | 3,944 → 3,560 | −384 |
| S04 | NULL | 4,720 → 4,720 | 0 | 6,809 → 6,620 | −189 |
| S05 | NULL | 4,591 → 4,591 | 0 | 5,988 → 6,485 | +497 |
| S06 | NULL | 5,055 → 5,055 | 0 | 9,393 → 9,651 | +258 |

E01 的 RDKit 解析日志来自故意传入的 `not-a-smiles`。两轮工具都成功返回，记录一条 `invalid_smiles` 和两条有效结果。S04 两轮都先列出正确的前两项 NCT，并附完整列表。S06 两轮的答复均标明说明书摘录被截断。

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
4. `step_limit`（Agent 达到图的模型调用上限）
5. `repair_budget_exceeded`（自愈评测实际修正超过 3 次）
6. `tool_error`
7. `wrong_tool`
8. `rule_failed`
9. `judge_call_error`
10. `judge_parse_error`
11. `judge_below_threshold`

只有满足预期工具及参数约束的调用才能记为 `replay_miss`。真实零结果和分子批次中的 `invalid_smiles` 属于成功工具响应。计入单例的异常保存类型、消息和 traceback。数据库、MCP 会话和明确的模型服务故障会中止运行。

## Postgres 表

| 表 | 列与键 |
| --- | --- |
| `eval.runs` | `run_id uuid` 主键，`tag text`，`git_commit text`，`user_id uuid` 唯一，`config jsonb`，`started_at timestamptz`，`finished_at timestamptz` 可空。 |
| `eval.cases` | `case_id text` 主键，`category text` 取 `single_tool`、`multi_hop`、`exception`、`self_healing`。每次运行从 YAML 同步。 |
| `eval.results` | `run_id uuid` 与 `case_id text` 为联合主键，分别引用前两张表。`thread_id uuid` 唯一，`trace_id text` 为 32 位十六进制，`success boolean`、`tools_correct boolean`、`params_valid boolean` 可空、`judge_score smallint` 可空、`failure_type text` 可空、`error_detail jsonb` 可空、`input_tokens bigint` 可空、`elapsed_ms bigint`、`repair_attempts smallint` 可空。 |

失败类型、类别、分数范围、计数与耗时均由 SQL 约束。运行配置记录模型、端点、温度、thinking、步数、回放设置、judge 细则版本与阈值、Git commit、工作区状态、用例清单及四份 SHA-256 摘要。用例和回放摘要对各文件的相对路径和原始内容计算。`system_prompt_sha256` 对 Agent 系统提示词文本计算，`tool_schema_sha256` 对传给 Agent 的工具定义计算，保留工具列表顺序并对 JSON 对象键排序。端点记录省略 URL 中的用户信息、查询参数和片段，配置中无密钥。 [建表 SQL](../eval/schema.sql) · [汇总 SQL](../eval/summary.sql)

`schema.sql` 可重复初始化。已有表新增列或变更具名 `CHECK` 约束时，需执行对应的 `ALTER TABLE` 迁移。

## Task 6 自愈评测

`eval/cases_self_healing` 单独运行并拥有独立的 `cases_sha256`。全新评测库由当前 `schema.sql` 初始化。已有 Task 5 评测库先执行 006，增加 `self_healing` 类别和可空的 `eval.results.repair_attempts`。已有结果的修正次数为 `NULL`。 [006 迁移 SQL](../eval/migrations/006_sandbox_self_healing.sql)

```sh
docker compose exec -T postgres psql -U pharmaflow -d pharmaflow -v ON_ERROR_STOP=1 \
  < eval/migrations/006_sandbox_self_healing.sql
```

已应用 006 的评测库执行 007，支持记录 `repair_budget_exceeded` 和超过 3 次的实际执行次数。 [007 迁移 SQL](../eval/migrations/007_repair_budget_exceeded.sql)

```sh
docker compose exec -T postgres psql -U pharmaflow -d pharmaflow -v ON_ERROR_STOP=1 \
  < eval/migrations/007_repair_budget_exceeded.sql
```

构建镜像并按前文配置 DeepSeek、数据库和 Langfuse 后运行。

```sh
docker build -t pharmaflow-sandbox:1 sandbox
uv run python -m eval.run --tag v3-self-healing --cases eval/cases_self_healing
```

每轮有 3 条用例。运行器在首个模型调用前执行预置步骤，并核对实际错误类型。列名错误还须命中指定的 traceback。预置步骤和结果进入模型上下文，评分和计时从后续修正开始。Docker 基础设施故障或预置步骤未产生指定错误时，运行中止。

| 用例 | 预置失败 | 恢复验收 |
| --- | --- | --- |
| H01 | 在原始 CSV 中读取大写 `IC50_NM` | 返回正确的非空记录数和最小值，并生成直方图。[H01 规则](../eval/cases_self_healing/H01.yaml) |
| H02 | 重复计算同一组记录的中位数，触发 15 秒超时 | 在执行时间内返回全部 50 条记录的精确中位数 77.50 nM。[H02 规则](../eval/cases_self_healing/H02.yaml) |
| H03 | 调用 chem-mcp 的 `export_properties`，用它返回的数据集执行脚本，读取大写 `QED` 触发 `KeyError` | 修正脚本沿用本轮派生 ID，输出十条按 QED 降序排列的 ChEMBL 记录和 MW–LogP 散点图。排名检查逐行核对 ID 和四位小数的 QED，允许另外打印保存提示。用例内的 `exported_properties` 在执行和判分前绑定到真实 ID。[H03 规则](../eval/cases_self_healing/H03.yaml) |

运行时从当前用户消息后的可修复失败推导额度。首次失败后最多尝试 3 次修正。成功的诊断脚本不消耗失败额度。`dataset_ids` 为必填参数，无 CSV 输入时显式传 `[]`。缺参时 ToolNode 将校验错误回给模型，调用在启动容器前结束。

评测从预置失败之后开始计数。模型实际执行的诊断脚本、成功的修正和可修复失败都计入 `repair_attempts`。任务成功要求在 3 次执行内传入指定的 `dataset_ids` 并生成符合规则的结果。超过 3 次计为 `repair_budget_exceeded`，`repair_attempts` 记录实际次数。汇总表 `self_healing` 行的「任务成功」即 3 轮内的恢复率。报告时同时列出成功数、用例数和运行轮数。OOM 的专项覆盖在执行器和 Docker 测试中。

Task 5 的 v2 正式基线使用另一份用例摘要。使用自愈目录运行的新结果可和相同摘要的自愈运行比较。

### 开发期自愈评测与复核

2026-11-26 完成首轮开发期自愈评测，运行 ID 为 `6b0b6c95-4f68-45f5-8636-ddb701cede67`。3 条用例均已入库并可评分，按当时规则入库的恢复率为 **1/3（33.3%）**。工具选择和参数合法率均为 100%，平均输入 7,440.3 token，修正阶段平均耗时 7,951.7 ms。该轮的用例摘要为 `978acd1473c2864ae22ca5f46e56fb402c118b4be7d88cf95a105a17e4a2c406`。

| 用例 | 检查点复核 | 修正次数 |
| --- | --- | ---: |
| H01 | 第一次执行列出 CSV 列名，第二次生成正确的统计和直方图。原规则把成功的诊断执行记成工具错误。 | 2 |
| H02 | 超时后一次执行算出精确中位数 77.50 nM。 | 1 |
| H03 | 十条排名和四位小数的 QED 全部匹配，散点图已生成。原规则按整段 stdout 比对，被末行 `saved` 触发。 | 1 |

按修正规则只读重判上述检查点，**3/3 通过**。重判采用的用例摘要为 `2969662b07b523a255aa1ff4d009cfb08d90c7140cfc1b788df91337db4ac2e1`。这次重判复用了同一批模型输出，Postgres 中的原运行记录对应当时的规则与摘要。

### 修正规则后的开发期自愈复跑

2026-11-26 用修正后的规则重新调用模型完成一轮评测，标签为 `v3-self-healing-score-fix`，运行 ID 为 `668accc8-f5f2-4ed8-803a-56ad48aa0cec`。本轮用例摘要与上面的重判一致，3 条用例全部入库并可评分，**3 轮内恢复率为 3/3（100%）**。工具选择和参数合法率均为 100%，平均输入 7,441.3 token，修正阶段平均耗时 7,583.0 ms。

| 用例 | 修正次数 | 输入 token | 修正阶段耗时（ms） |
| --- | ---: | ---: | ---: |
| H01 | 2 | 9,803 | 8,129 |
| H02 | 1 | 5,566 | 4,692 |
| H03 | 1 | 6,955 | 9,928 |

### 提示词修订后的开发期自愈复跑

2026-11-26 修订 E01 相关的输出格式提示词并应用 007 迁移后再次运行，运行 ID 为 `9379a15c-c5cd-4728-abf0-6d9b95efe6c4`。与前一轮相比，模型服务设置、用例摘要和回放摘要匹配，系统提示词经历了上述修订。这两轮早于系统提示词和工具定义的摘要入库。本轮 3 条结果均已入库并可评分。工具选择、参数合法率和 **3 轮内恢复率均为 2/3（66.7%）**，平均输入 11,671.0 token，修正阶段平均耗时 12,431.7 ms。

| 用例 | 入库结果 | 修正次数 | 输入 token | 修正阶段耗时（ms） |
| --- | --- | ---: | ---: | ---: |
| H01 | `invalid_args` | 3 | 18,443 | 21,666 |
| H02 | 通过 | 1 | 5,607 | 4,815 |
| H03 | 通过 | 2 | 10,963 | 10,814 |

H01 的预置脚本成功读取 `candidates` 并触发预期的列名错误。模型随后两次调用漏传 `dataset_ids`，分别得到文件不存在和输入目录为空的结果。模型又调用 `export_properties` 生成了派生数据集，第三次沙盒执行得到了正确的统计值和直方图。评分要求直接使用原始 `candidates`，额外导出和三个沙盒请求的参数都偏离了用例路径。该例的 trace ID 为 `6d3d361e8867e4b116f414e34f144b6d`。

根据这轮 H01 的失败，`run_python` 明确每次调用使用新容器，并将 `dataset_ids` 改为必填。诊断和修正调用都须传入脚本读取的 CSV ID。缺参会在启动容器前返回校验错误，因此不会增加实际修正执行次数。 [工具定义](../sandbox/tool.py) · [参数定义](../sandbox/models.py)

### Task 6 基础用例回归

使用与 v2 相同的模型和评测配置。默认目录包含 12 条基础用例。对比使用 v2 第二轮作为基准，并校验两轮的用例和回放摘要。输入复跑完成后打印的 run ID。

```sh
uv run --no-sync python -m eval.run --tag v3-baseline &&
  printf 'v3 run_id ' &&
  IFS= read -r task6_v3_run_id &&
  uv run --no-sync python -m eval.compare \
    84dfc269-4256-45fd-8c81-de7c5ec198b9 "$task6_v3_run_id"
```

逐例检查成功状态和输入 token 差值。S01 应使用 `compute_properties`，S02 应使用 `filter_druglike`。如遇工具选择失败，查看该例的 `error_detail` 或 trace。

2026-11-26 完成两轮开发期基础回归，均早于 `dataset_ids` 必填修订。首次在 `3ebab3f` 的 Task 6 工作区运行，`git_dirty=true`，运行 ID 为 `d650cd8f-8413-44ab-abf5-683d115c2910`。提示词修订后复跑的运行 ID 为 `de5ceede-b02f-4875-96ae-86a10bc09e63`。两轮与 v2 第二轮的模型、端点、温度、thinking、调用上限、judge 配置及用例与回放摘要一致。

| 指标 | v2 第 2 轮 | v3 初次回归 | v3 提示词修订后 |
| --- | ---: | ---: | ---: |
| 计划数 / 已入库数 / 可评分数 | 12 / 12 / 12 | 12 / 12 / 12 | 12 / 12 / 12 |
| 回放未命中 / 缺 token | 0 / 0 | 0 / 0 | 0 / 0 |
| 工具选择准确率 | 100% | 100% | 100% |
| 参数合法率 | 100% | 100% | 100% |
| 任务成功率 | 12/12（100%） | 11/12（91.7%） | 12/12（100%） |
| 平均输入 token | 5,723.0 | 6,992.8 | 7,046.0 |
| 平均端到端耗时（ms） | 6,924.1 | 6,874.1 | 5,377.6 |

检查点确认初次回归与 v2 每例的 Agent 模型调用次数相同。初次回归每例首个调用多 586 个输入 token，平均每例增加 1,269.8 个（22.2%）。提示词修订后复跑，平均每例比 v2 增加 1,323.0 个（23.1%）。两次回归的 S01 均使用 `compute_properties`，S02 均使用 `filter_druglike`，两例均通过。

初次回归唯一失败是 E01（`rule_failed`，trace ID `164483ff6dee5759d298ee4fbb2c7891`）。`compute_properties` 返回两条正确分子量与一条 `invalid_smiles`。最终答复先按顺序给出三行表格，随后在说明中逐条复述同一批输入。E01 要求每条输入只列一行，实际六次标识匹配触发 `answer_sequence`。

根据 E01 的失败，系统提示词增加了每条输入只列一次的格式要求，并将报告被排除行的说明放在性质导出场景下。复跑时，E01 在三行内分别给出 CCO 的 46.069 Da、`not-a-smiles` 的解析错误和 CC 的 30.07 Da。原规则通过，trace ID 为 `15e6eed89432c1a8c2924dca42e3786e`。上表的 12/12 是包含该调试用例的基础集回归结果。

### Task 6 提交后正式评测

从包含本次修改的提交在干净工作区复跑。基础与自愈用例各跑两轮，每轮保存 run ID、Git commit、提示词和工具定义摘要及汇总表。上述开发期数据保留用于解释规则和提示词的调整。

```sh
git status --short
git rev-parse HEAD

uv run --no-sync python -m eval.run --tag v3 &&
  uv run --no-sync python -m eval.run --tag v3

uv run --no-sync python -m eval.run --tag v3-self-healing --cases eval/cases_self_healing &&
  uv run --no-sync python -m eval.run --tag v3-self-healing --cases eval/cases_self_healing
```

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
