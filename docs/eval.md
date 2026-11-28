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
docker compose up -d --wait postgres qdrant
uv run --locked --no-env-file python -m rag.download_models
uv run --offline --locked --no-sync --no-env-file python -m rag.index

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

等两轮都完整结束，再执行下面的命令。按终端提示依次粘贴两次运行打印的 UUID，每个 ID 输入后按回车。对比要求用例摘要和录制摘要分别一致。系统提示词、工具定义和检索语料摘要各显示「一致」「不同」或「缺少记录」，支持比较改动前后的模型输入。逐用例差值为第 2 轮减第 1 轮。judge 分数和 token 任一侧为 `NULL` 时，相应差值为 `NULL`。成功状态按 0 和 1 求差，有 `replay_miss` 的用例成功差值显示 `NULL`。

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

失败类型、类别、分数范围、计数与耗时均由 SQL 约束。运行配置记录模型、端点、温度、thinking、步数、回放设置、judge 细则版本与阈值、Git commit、工作区状态、用例清单及 SHA-256 摘要。用例和回放摘要对各文件的相对路径和原始内容计算。`system_prompt_sha256` 对 Agent 系统提示词文本计算，`tool_schema_sha256` 对传给 Agent 的工具定义计算，保留工具列表顺序并对 JSON 对象键排序。Task 7 起，`corpus_sha256` 对 `chunks.jsonl` 原始字节计算。端点记录省略 URL 中的用户信息、查询参数和片段，配置中无密钥。 [建表 SQL](../eval/schema.sql) · [汇总 SQL](../eval/summary.sql)

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

2026-11-26 在提交 `09c1684e08f993160952fb78cda50600b5e692fa` 完成四轮。Postgres 中四轮均记录 `git_dirty=false`。Agent 使用 `deepseek-v4-pro`，温度 0，关闭 thinking，模型调用上限为 10 次。提示词、工具定义和回放摘要在四轮中一致，用例摘要在各自套件的两轮中一致。

| 输入 | SHA-256 摘要 |
| --- | --- |
| 系统提示词 | `1b07eb1ed4c1230bee7199dda63d9b841f8d33343c6800d180826c6369fcb38f` |
| 工具定义 | `f27571c4c58da09477f2dbac2ed8f1f761e85f33776d1c8410cf2afeea61cc73` |
| 回放录制 | `b1c114e32203fdb72535fea39124a374d5008dc2d6156401efda1911ecf20a9e` |
| 基础用例 | `54e9f689f88c9704189d918bd1a407824cd341b625115645789215938bdf3968` |
| 自愈用例 | `2969662b07b523a255aa1ff4d009cfb08d90c7140cfc1b788df91337db4ac2e1` |

| 轮次 | Run ID | 开始至结束（UTC+8） | 任务成功 | 平均输入 token | 平均耗时（ms） |
| --- | --- | --- | ---: | ---: | ---: |
| 基础第 1 轮 | `0bd743d8-2d38-4470-a657-e6dd03f48ff4` | 16:35:53–16:37:05 | 12/12 | 7,152.3 | 5,068.9 |
| 基础第 2 轮 | `25fb9fa3-6ac1-4039-aeb5-3dc864641969` | 16:37:08–16:38:21 | 11/12 | 7,152.0 | 5,066.1 |
| 自愈第 1 轮 | `1b013693-ce82-4b1e-aae9-6b393f8bdfdd` | 16:38:24–16:38:59 | 3/3 | 7,624.3 | 6,085.7 |
| 自愈第 2 轮 | `4565550d-8255-44a9-b4e9-c6f5f878474d` | 16:39:02–16:39:42 | 3/3 | 8,852.3 | 7,474.0 |

四轮都完整入库，回放未命中和缺 token 均为 0。工具选择和参数合法率均为 100%。基础评测计端到端耗时，自愈评测计预置失败之后的修正耗时。

**基础用例两轮合计通过 23/24（95.8%）。** E01 两轮均通过。S01 和 S02 分别使用 `compute_properties` 和 `filter_druglike`。第二轮 E02 正确指出编号格式问题，随后附上了示例编号 `NCT01234567`，因此触发禁止示例编号的规则（`rule_failed`，trace ID `7294fe383e89a7b3522cff6946deb281`）。基础用例与回放摘要匹配 v2 第二轮，v3 两轮平均每例输入 7,152.2 token，比 v2 第二轮增加 1,429.2 个（25.0%）。 [E02 用例](../eval/cases/E02.yaml)

**自愈评测共 3 条用例、2 轮，3 轮内恢复率为 6/6（100%）。** H01 两轮均在 2 次执行内完成，诊断和修正都传入原始 `candidates` ID。H02 两轮各用 1 次，H03 两轮分别用 1 次和 2 次。

## Task 7 检索评测

证据语料由 `data/corpus_recordings/` 回放构建。基础 Agent 回归继续使用 `data/lit_recordings/` 的摘要作为对比条件。 [语料构建](lit-mcp.md#task-7-证据语料)

复核题目后，用已建成的 `kras_evidence` 生成候选池。先按业务目的拟中文问题，再翻译为英文检索词。六词连续重合检查在单测中比较规范化的英文检索词和全部语料片段。Recall@8 和 MRR 衡量给定人工英文检索词时的检索质量。Agent 实际生成的英文查询在中文演示中核对。 [查询集](../eval/retrieval/queries.yaml) · [中文演示](../README.md#演示证据检索)

```sh
uv run --offline --locked --no-sync --no-env-file python -m eval.retrieval pool
```

候选池合并 dense、hybrid 和 hybrid + rerank 的前 10 条，按片段 ID 去重，固定种子打乱后写入 `eval/retrieval/pool.jsonl`。每行记录查询、相关性单元和片段。原方法、名次和得分在池内隐藏。所有方法都漏召的证据可能在池中缺失，使 Recall@8 的分母小于语料中实际相关单元数。标注可补充池外相关单元并标明来源。

初稿候选池包含 259 个查询与片段对。按复核后的英文检索词重新生成后，共 255 对，覆盖全部 15 条查询。R03、R05 和 R06 的候选集合随检索词更新。 [候选池](../eval/retrieval/pool.jsonl)

### qrels 标注规则

确认语料中是否有答案以及判断相关性时，都检查文献、试验和说明书。查询类别按相关证据的实际来源填写。片段能回答问题的一部分即算相关，其所属单元可以标为相关。同一文献可为多条查询提供相关证据。

`qrels.yaml` 以查询 ID 为键。每个查询至少需要一条标注，`unit_id` 对应语料中一个或多个片段，`reason` 写一句相关理由。`source: pool` 表示候选池，`source: corpus` 表示从池外语料补充。复核后共 20 个查询与相关单元对，其中 R01 的 CodeBreaK 200 主论文和 R04 的 KRYSTAL-1 单药论文从池外补充。 [相关性标注](../eval/retrieval/qrels.yaml)

R03 按 sotorasib 说明书 5.1 的全等级肝毒性口径提问。R05 只标 CodeBreaK 300 主论文 `pubmed:37870968`。R06 改问晚期 NSCLC 两个单药剂量组的常见治疗期间不良事件，标注 `pubmed:39029295`。题目和英文检索词都采用复核后的口径。 [查询集](../eval/retrieval/queries.yaml)

### 两轮检索评测

复核标注后，从包含 Task 7 的干净工作区运行。复用现有 Qdrant 索引和模型缓存，并沿用上次评测的 Postgres 配置。先跑集成测试，再将两轮检索结果写入 Postgres，并比较两轮报告。

```zsh
(
  set -e -o pipefail
  if [ -n "$(git status --porcelain=v1)" ]; then
    git status --short
    printf '%s\n' '请在包含 Task 7 的干净工作区运行。' >&2
    exit 1
  fi

  git rev-parse HEAD
  task7_reports=$(mktemp -d)
  printf '评测日志目录 %s\n' "$task7_reports"
  uv run --offline --locked --no-sync --no-env-file pytest tests/integration
  uv run --offline --locked --no-sync --no-env-file python -m eval.retrieval run |
    tee "$task7_reports/run1.txt"
  uv run --offline --locked --no-sync --no-env-file python -m eval.retrieval run |
    tee "$task7_reports/run2.txt"

  diff -u \
    <(sed -E -e '/^run_id /d' -e 's/^运行 [0-9a-f-]+ /运行 /' "$task7_reports/run1.txt") \
    <(sed -E -e '/^run_id /d' -e 's/^运行 [0-9a-f-]+ /运行 /' "$task7_reports/run2.txt")
)
```

验收时保留提交 SHA、集成测试摘要和两个 `run_id`。两轮都应显示「完整」，每轮覆盖 15 条查询与 3 种方法。末尾 `diff` 无输出表示 Recall@8、MRR 和逐查询的引用顺序一致。查询历史汇总时将 `TASK7_RETRIEVAL_RUN_ID` 设为目标 `run_id`。

```sh
uv run --offline --locked --no-sync --no-env-file python -m eval.retrieval summary "$TASK7_RETRIEVAL_RUN_ID"
```

Recall@8 用前八条命中的不同相关单元数除以该查询相关单元总数。MRR 取各查询首个相关结果名次的倒数后平均，前八条没有相关结果时倒数排名为 0。`eval.retrieval_runs` 保存 Git commit、起止时间、三份输入摘要、模型名称、候选上限、别名和同分排序规则。`eval.retrieval_results` 保存每个查询和方法的名次、相关单元计数及排名后的片段引用 ID。命令打印各方法的 SQL 平均值及逐条查询结果。两次同配置运行可核对逐条的引用 ID 与指标。 [建表 SQL](../eval/schema.sql) · [检索汇总 SQL](../eval/retrieval/summary.sql)

### 调整同分排序前的两轮实测

第 1 轮 `07383caf-5f4d-42fe-baa6-4b3d442dad01` 和第 2 轮 `ed9f1067-5a85-4c5b-8714-06bab77bbfe0` 均完整结束，每轮覆盖 15 条查询与 3 种方法。按当时的语料和初稿标注逐条复算，两轮的 90 条明细及汇总均校验通过。

| 方法 | 第 1 轮 Recall@8 | 第 2 轮 Recall@8 | 第 1 轮 MRR | 第 2 轮 MRR |
| --- | ---: | ---: | ---: | ---: |
| `dense` | 0.5667 | 0.5667 | 0.4883 | 0.4883 |
| `hybrid` | 0.8667 | 0.8667 | 0.7000 | 0.6889 |
| `hybrid_rerank` | 0.8667 | 0.8667 | 0.8333 | 0.8333 |

`dense` 和 `hybrid_rerank` 的逐条引用顺序一致。`hybrid` 的 R02、R03、R04、R10、R12 和 R15 的前八条出现顺序变化。R02 的第八名还更换了片段。所有查询的相关单元命中数一致，R02 的首个相关结果从第 2 名变为第 3 名，导致 `hybrid` 的平均 MRR 下降约 0.0111。

对同一语料索引只读复查 R02，`pubmed:40783289:2` 与说明书 5.4 片段的 RRF 分数均为 0.5，两者顺序在请求之间互换。第 7 至第 9 名的三个片段同得 0.2，顺序变化会改变第 8 名截断处的引用。三种方法两轮都漏召 R01 和 R04 的相关单元。

增加稳定排序后，Qdrant 返回两路各 30 条预取所产生的全部融合候选。客户端以 RRF 分数降序、点 ID 升序取前 30 条，处理跨截断位置的同分。运行配置记录 `rrf_tie_break=point_id_ascending`。

提交 `27a6b29` 的两轮初稿标注结果已入库，运行 ID 为 `d2099067-0495-4c6e-bca3-2dbc2d6f8f37` 和 `74420e6d-e2e6-45db-aac9-2c17127f8be7`。两轮的查询、标注和语料摘要一致，每轮 45 条结果的指标和引用顺序相同。R02 在 `hybrid` 中首个相关结果位于第 2 名。三组指标与上表第 1 轮列一致。

### 复核标注后的两轮开发评测

使用 `aa1fd81` 加未提交的查询和标注改动，在同一个 Qdrant 索引上跑了两轮，运行 ID 为 `43be5fa0-27c9-4788-b70b-dfeef0bcb9f8` 和 `2e7a2d17-2a23-4734-9f2e-bc76a8dcc701`。两轮共入库 90 条明细，配置摘要、逐条指标和前八条引用顺序一致。

| 方法 | Recall@8 | MRR |
| --- | ---: | ---: |
| `dense` | 0.6333 | 0.5217 |
| `hybrid` | 0.8667 | 0.7083 |
| `hybrid_rerank` | 0.8667 | 0.7778 |

未修改的 12 条查询与稳定排序后的初稿运行逐条一致。重排的 MRR 从 0.8333 降到 0.7778，变化来自 R05 在新口径下的首个相关结果排第 6。R05 的相关单元现在只有主论文。

### 基础用例回归

在包含 Task 7 改动的干净工作区，沿用 Task 6 正式基础评测的 DeepSeek 和 Langfuse 配置运行两轮默认的 12 条基础用例。命令先检查工作区。两轮完成后，将打印的两个 `run_id` 粘贴到对比命令的提示中。

```sh
if [ -n "$(git status --porcelain=v1)" ]; then
  git status --short
  printf '%s\n' '请在包含 Task 7 的干净工作区运行。' >&2
else
  git rev-parse HEAD &&
    uv run --offline --locked --no-sync --no-env-file python -m eval.run --tag v4 &&
    uv run --offline --locked --no-sync --no-env-file python -m eval.run --tag v4
fi
```

```sh
printf 'Task 7 第 1 轮 run_id ' &&
  IFS= read -r TASK7_AGENT_RUN_1 &&
  printf 'Task 7 第 2 轮 run_id ' &&
  IFS= read -r TASK7_AGENT_RUN_2 &&
  uv run --offline --locked --no-sync --no-env-file python -m eval.compare \
    25fb9fa3-6ac1-4039-aeb5-3dc864641969 "$TASK7_AGENT_RUN_1" &&
  uv run --offline --locked --no-sync --no-env-file python -m eval.compare \
    25fb9fa3-6ac1-4039-aeb5-3dc864641969 "$TASK7_AGENT_RUN_2"
```

### 开发期基础评测

贴回的两轮 `v4` 在 `53c3c2a635c6e7ecb07d20e488f9e48a1f221826` 的有改动工作区执行。两轮各有 12 条结果入库并可评分，回放未命中和缺 token 均为 0，工具选择和参数合法率均为 100%。

| 轮次 | Run ID | 任务成功 | 平均输入 token | 平均耗时（ms） |
| --- | --- | ---: | ---: | ---: |
| Task 6 正式第 2 轮 | `25fb9fa3-6ac1-4039-aeb5-3dc864641969` | 11/12 | 7,152.0 | 5,066.1 |
| Task 7 开发期第 1 轮 | `ebeade95-cef9-45bd-bd7e-03a728e3cffc` | 12/12 | 7,553.2 | 9,960.6 |
| Task 7 开发期第 2 轮 | `39aa38a3-a334-42bb-b2cf-4bc22538e1a0` | 12/12 | 7,554.3 | 5,246.0 |

两轮开发期评测的耗时差集中在单工具组，该组平均耗时分别为 14,082.0 和 4,422.3 ms。使用实际 `run_id` 对 Postgres 中的结果逐例比较，两轮均有 12 条匹配用例。相对 Task 6 第 2 轮，两轮的 E02 均从失败变为成功，其余 11 条均继续通过。

```sh
uv run --offline --locked --no-sync --no-env-file python -m eval.compare \
  25fb9fa3-6ac1-4039-aeb5-3dc864641969 ebeade95-cef9-45bd-bd7e-03a728e3cffc
uv run --offline --locked --no-sync --no-env-file python -m eval.compare \
  25fb9fa3-6ac1-4039-aeb5-3dc864641969 39aa38a3-a334-42bb-b2cf-4bc22538e1a0
```

## Task 8 多智能体与上下文压缩

`AGENT_GRAPH_MODE` 在 CLI、API 和评测中选择 `single` 或 `multi`，默认 `single`。两个图的 `AGENT_TRIM_TOOL_FIELDS` 和 `AGENT_SUMMARIZE_HISTORY` 都默认关闭。字段裁剪只影响 Agent 送入模型的副本，例如省略请求回显、扩展检索式和重排分数。`AGENT_STRUCTURED_WORKER_RESULTS` 默认开启，仅影响多图。关闭时 Reporter 同时接收分支结论和完整消息记录，记录作为转义后的 JSON 数据块传入。

Task 12 对比字段裁剪时使用多智能体模式，固定其他开关，只切换 `AGENT_TRIM_TOOL_FIELDS`。多图在两个状态下都使用数据块。单 Agent 启用裁剪也会改变系统提示词和历史消息的呈现方式。数据块将内容中的 `</` 写成 JSON 中合法的 `<\/`，说明书 HTML 的起始标签保持可读。

Planner 每轮至多生成一次共享的历史摘要。各 worker 与 Reporter 使用同一份摘要和保留的历史，worker 各自压缩本轮的 ReAct 记录。以一条消息及其全部工具结果为一个保留单位。每个 worker 的输入依次为系统提示词、摘要块、历史数据块、本轮原始用户消息和本轮保留单位。Critic 重派时在末尾添加一条临时用户消息，写明反馈，不写入分支记录。

原始用户文本即使早于摘要的位置也照样送入模型。摘要末尾由代码附上 PMID、NCT、说明书、片段和数据集 ID，包括 `candidates` 和 `upload_` 开头的 ID。多图跨轮传递对话历史，历史答复中已经给出的 `export_properties` 派生 ID 可以复用。只出现在工具结果里的 ID 随当轮分支记录归档。

开启摘要时填写 `LLM_CHEAP_MODEL`，沿用主模型的端点和密钥。两个模型都通过同一个客户端配置创建，DeepSeek 关闭 thinking。下例使用 `.env.example` 中的模型名，若配置了其他便宜模型，将命令中的值改为实际模型。

```sh
LLM_MODEL=deepseek-flash uv run --offline --locked --no-sync --no-env-file python -m scripts.check_model
```

### Token 预算与估算误差

预算为代码常量 **6,144**，保留最近 **2** 个可压缩单位，完整保留原始用户消息。估算对序列化后的模型输入计算，包含系统提示词、消息封装和实际绑定的工具定义。每个非 ASCII 字符计 1 token，每个 ASCII 字符计 0.3 token，使用固定系数。固定输入与保留单位仍超预算时照常调用，模型 trace 中的 `context_budget_exceeded` 标记该次调用，旁边记录估算值和预算。

从 Task 7 两轮开发期的 24 个本地 Postgres 检查点逐次重建模型输入，使用当时的单 Agent 系统提示词和完整工具定义。重建的工具定义摘要与两轮入库值相同，52 次调用的实际用量是检查点里各 `AIMessage.usage_metadata.input_tokens`。误差是估算值减实际值。

| 样本 | 实际中位数 | 实际 P95 | 实际最大值 | 平均误差 | 平均绝对误差 | 平均绝对百分比误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 次 Agent 调用 | 2,908 | 5,331 | 7,576 | +234.81 | 320.19 | 8.28% |

误差范围为 −1,110 至 +908 token，绝对误差 P95 为 906.9 token。以 6,144 为界，估算超预算 4 次，实际超预算 2 次。两次 S02 的实际值为 7,576、估算值为 6,466，两次 M02 的估算值超预算而实际值低于预算。预算来自携带全部工具定义的单 Agent 调用。多图 worker 只绑定子集，12 条基础用例又是单轮，即使开启摘要也可能没有一次实际压缩。Task 12 测量摘要效果时需加入能触发摘要、且有完整旧单位可压缩的用例。

### 同提交的两组基础评测

在收录 Task 8 改动的同一个提交上完成四轮评测。沿用相同的模型、回放、证据索引和 Langfuse 配置。关闭字段裁剪和历史摘要，保留多图默认的结构化回传。关闭结构化回传的实验留到 Task 12。先跑两轮单 Agent，每轮保存 `run_id`。默认目录有 12 条基础用例，多图暂不支持独立的自愈用例目录。

```sh
git rev-parse HEAD
export AGENT_TRIM_TOOL_FIELDS=false
export AGENT_STRUCTURED_WORKER_RESULTS=true
export AGENT_SUMMARIZE_HISTORY=false
AGENT_GRAPH_MODE=single uv run --offline --locked --no-sync --no-env-file python -m eval.run --tag task8-single-1 &&
  AGENT_GRAPH_MODE=single uv run --offline --locked --no-sync --no-env-file python -m eval.run --tag task8-single-2
```

把刚输出的两轮 ID 分别赋给 `TASK8_SINGLE_1` 和 `TASK8_SINGLE_2`。先核对两轮的「系统提示词摘要」和「工具定义摘要」均与 Task 7 `39aa38a3` 显示一致。它和 `ebeade95` 都是有未提交改动时的开发运行，仅用于这项核对。正式对比用本节新跑的单 Agent 两轮。Task 6 的 `25fb9fa3` 早于 `search_evidence`，两项摘要与 Task 7/8 的对应值有差异。

```sh
uv run --offline --locked --no-sync --no-env-file python -m eval.compare \
  39aa38a3-a334-42bb-b2cf-4bc22538e1a0 "$TASK8_SINGLE_1" &&
  uv run --offline --locked --no-sync --no-env-file python -m eval.compare \
    39aa38a3-a334-42bb-b2cf-4bc22538e1a0 "$TASK8_SINGLE_2"
```

在同一提交上沿用上述三个开关值，再跑多图两轮。把新输出的 ID 分别赋给 `TASK8_MULTI_1` 和 `TASK8_MULTI_2`。

```sh
git rev-parse HEAD
AGENT_GRAPH_MODE=multi uv run --offline --locked --no-sync --no-env-file python -m eval.run --tag task8-multi-1 &&
  AGENT_GRAPH_MODE=multi uv run --offline --locked --no-sync --no-env-file python -m eval.run --tag task8-multi-2
```

记录四轮汇总的「任务成功」和「平均输入 token」，再查看两组逐例差异。用量包含 Planner、worker、Reporter 和启用时的摘要调用。Judge 单独计费。多图按 Planner 的子任务顺序拼接各分支原始记录，最后接 Reporter 答复。

```sh
uv run --offline --locked --no-sync --no-env-file python -m eval.compare "$TASK8_SINGLE_1" "$TASK8_MULTI_1" &&
  uv run --offline --locked --no-sync --no-env-file python -m eval.compare "$TASK8_SINGLE_2" "$TASK8_MULTI_2"
```

运行配置记录图模式、各节点模型、三个压缩开关、预算和保留单位数、各节点系统提示词 SHA-256，以及各 worker 可用工具定义的 SHA-256。单 Agent 的 `system_prompt_sha256` 取实际系统提示词模板。关闭字段裁剪和历史摘要时，取值与基线一致。Critic 的模型为 `null`。`eval.compare` 汇报各项是否一致。旧运行缺失的字段会显示「缺少记录」。用 Langfuse 查看实际的并行时间区间见 README 的混合文献与分子问题。 [并行演示](../README.md#演示文献和分子并行分析)

Reporter 把未核验的编号追加到答复末尾，评测同时保存到 `error_detail.reporter_unverified_ids`。代码识别 `NCT` 加 8 位数字、带 `PMID` 前缀的数字、片段 `citation_id` 和 UUID 格式的说明书 `set_id`。用户原文已有的编号不计入未核验项。E02 规定答复中不得出现有效格式的 NCT 示例，正文或末尾追加说明出现编造的 `NCT` 加 8 位数字时，此例按现有规则失败。

### 正式评测结果

2026-11-27 在干净提交 `5f6b01b3de24e00ab1987efa5a407fd75219b368` 完成四轮基础评测。四轮使用 `deepseek-v4-pro` 和同一份回放及语料，启用结构化回传，关闭字段裁剪和历史摘要。48 条结果全部入库且可评分，输入 token 记录齐全。两轮单图的系统提示词、工具定义和语料摘要均与 Task 7 的 `39aa38a3` 一致。

| 轮次 | Run ID | 通过 | 平均输入 token | 平均耗时（ms） |
| --- | --- | ---: | ---: | ---: |
| 单图第 1 轮 | `1b7282ca-12ac-4363-82af-54beed67e397` | 12/12 | 7,553.8 | 5,652.1 |
| 单图第 2 轮 | `5b1084b0-576a-416d-a93f-245839fc3c82` | 12/12 | 7,553.3 | 7,483.0 |
| 多图第 1 轮 | `d1d7315d-2307-4fdc-8884-35b6cbd66b42` | 8/12 | 6,242.5 | 13,932.8 |
| 多图第 2 轮 | `05ed0e2f-2ff4-459f-9f58-ecea12963191` | 8/12 | 6,703.2 | 12,830.4 |

合并两轮后，按全部 12 条用例计算的平均输入 token 为单图 7,553.5、多图 6,472.8，多图低 14.31%。该指标包含失败用例的调用开销。成功率从 100%（24/24）降至 66.7%（16/24），平均耗时从 6.57 秒增至 13.38 秒。

只比较四轮均通过的同一组 8 条用例（E02、M01、S01–S06），每种模式各有 16 条结果。单图平均输入 6,978.4 token，多图为 4,413.6 token，多图低 36.75%。这组用例包含六条单工具用例，比较结果适用于这个共同通过的子集。

归因来自两轮多图检查点的 `tasks` 和对应的分支工具记录。Planner 的工具分配决定各 worker 可以执行哪些工具。并行分支的输入中没有同轮其他分支的工具结果。

| 两轮均失分的用例 | 入库原因 | 检查点核对结果 |
| --- | --- | --- |
| E01 | `wrong_tool` | Planner 两轮都只分配 `run_python`。MolAnalyst 调用了该工具，计划中缺少应使用的 `compute_properties`。 |
| E03 | `wrong_tool` | Planner 正确为 PubMed 分支分配 `search_pubmed`，给 openFDA 分支的唯一工具是 `search_evidence`。计划缺少所需的 `get_label_section`。 |
| M02 | `invalid_args` | Planner 保留「第 5 条书目标题的药名」这一目标，将 PubMed 和说明书查询拆为两个并行分支。PubMed 的第 5 条确为 Adagrasib，说明书分支拿不到该结果。两轮中该分支尝试未分配的 `search_evidence` 均被拒绝，随后首次查询都传入 `sotorasib`。第二轮又查询了 `adagrasib`。 |
| M03 | `wrong_tool` | Planner 两轮都将试验查询与依赖其结果的说明书查询拆开，并为说明书分支额外分配 `search_evidence`，两轮均调用了它。第二轮还给试验分支分配了 `search_trials`，两次额外调用均未命中回放。两轮的说明书查询实际都用了 `adagrasib`。 |

`deepseek-flash` 通过强制工具调用、结构化输出、温度采样和带工具的多轮对话探针。README 的混合问题通过真实 API 执行，`search_evidence` 和 `compute_properties` 均成功。Langfuse trace `3f90c3a424c872eeaf1325d16b209359` 中，`LitResearcher` 与 `MolAnalyst` 的执行时间重叠 5.612 秒，随后依次执行 Critic 和 Reporter。

核对两轮多图的 24 个检查点及混合演示的检查点，`critique_passes` 均为 1。反馈作为最后一条用户消息传入的顺序由单测覆盖。这批真实运行未触发重派，因此尚未验证模型收到反馈后是否会修正结论。

### 工具分配优化的开发复测

Planner 按请求分配专用工具。如果后一步的参数来自上一步结果，Planner 把调用链放到同一个子任务，并写明顺序和参数来源。LitResearcher 从上一步的实际返回值取得参数，并直接报告检索成功返回的零结果。MolAnalyst 用默认条件筛选时省略阈值参数，从返回的 `passes_filter` 统计通过数。

补充默认筛选规则前，两轮开发复测各通过 11/12，E01、E03、M02 和 M03 均通过。S02 的两个检查点显示，MolAnalyst 先用工具默认值完成筛选，又显式传入相同阈值跑了一次，额外调用导致 `wrong_tool`。

最终在 `aa1fd81` 加未提交提示词改动的工作区重跑两轮。用例、模型、工具定义、回放、语料和压缩配置与正式多图一致。两轮的 E01 和 E03 走专用工具，M02 和 M03 各在同一分支完成两次调用，S02 各调用一次 `filter_druglike`。24 条均通过，检查点的 `critique_passes` 均为 1。

| 轮次 | Run ID | 通过 | 平均输入 token | 平均耗时（ms） |
| --- | --- | ---: | ---: | ---: |
| 多图开发复测第 1 轮 | `7e74e3ed-5c24-480e-8059-b3c01d8fe0c1` | 12/12 | 5,079.3 | 8,028.9 |
| 多图开发复测第 2 轮 | `ea65b4ee-cce2-4116-9c6d-e31e2f12fc31` | 12/12 | 5,057.8 | 7,470.8 |

两轮正式单图与两轮开发多图都通过全部 12 条，平均输入分别为 7,553.5 和 5,068.5 token，多图低 32.90%。按优化前后多图均通过的 8 条用例计算，平均输入从 4,413.6 增至 4,534.9 token，增加 2.75%。

### 工具分配优化后的正式评测

2026-11-28 在干净提交 `125b5988bb5444e578ca023cb4ccb0a84795cb08` 上先跑两轮单图，确认系统提示词、工具定义和语料摘要均与 `39aa38a3` 一致，再跑两轮多图。四轮使用 `deepseek-v4-pro` 和同一份用例、回放及语料，开启结构化回传，关闭字段裁剪和历史摘要。48 条结果全部入库且可评分，输入 token 记录齐全。

| 轮次 | Run ID | 通过 | 平均输入 token | 平均耗时（ms） |
| --- | --- | ---: | ---: | ---: |
| 单图第 1 轮 | `2a29a831-f104-4a9a-93a4-75593ebef035` | 12/12 | 7,552.9 | 4,229.8 |
| 单图第 2 轮 | `3daa9836-caff-4a3c-b057-e72c546d61da` | 11/12 | 7,847.0 | 4,329.0 |
| 多图第 1 轮 | `0794d7bb-26d4-48dd-b6ff-d8007aa0dc45` | 12/12 | 5,088.6 | 8,521.2 |
| 多图第 2 轮 | `60c0453c-7b1b-40e6-a70f-167a638bc564` | 12/12 | 5,096.4 | 9,232.2 |

单图通过 23/24（95.8%），多图通过 24/24（100%）。全部样本的平均输入为单图 7,700.0、多图 5,092.5 token，多图低 33.86%。平均耗时分别为 4.28 秒和 8.88 秒。按四轮共同通过的 11 条用例计算，每种模式统计 22 条结果，平均输入为单图 7,659.3、多图 5,187.5 token，多图低 32.27%。

单图第 2 轮的 S04 首次调用 `search_trials` 时将参数放在顶层，缺少必填的 `request` 对象。第二次调用正确封装后成功，评测记录为 `invalid_args`，Langfuse trace 为 `23ca14226dc0ad942edff07466abe9f1`。

两轮多图的检查点均显示，E01 和 E03 使用专用工具，M02 和 M03 在同一分支完成有依赖的两次调用，S02 只调用一次 `filter_druglike`。24 条多图结果的 `critique_passes` 均为 1。

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
