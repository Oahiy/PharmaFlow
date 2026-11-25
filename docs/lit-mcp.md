# lit-mcp 本地录制与回放

从项目根目录启动 stdio 服务。测试和演示使用已录制的数据。

```sh
FASTMCP_ENV_FILE=/dev/null LIT_MCP_MODE=replay \
  LIT_MCP_RECORDINGS_DIR="$PWD/data/lit_recordings" \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.server
```

stdio 客户端启动子进程时，在 `env` 中显式传入配置。服务在启动时读取一次环境变量，四个工具共享一个 `httpx.AsyncClient`，退出时关闭。

| 配置 | 默认值 | 用途 |
|---|---|---|
| `LIT_MCP_MODE` | `replay` | `live` 访问上游，`record` 保存最终响应，`replay` 读取录制文件。 |
| `LIT_MCP_RECORDINGS_DIR` | 项目内的 `data/lit_recordings/` | 指定录制目录。显式传相对路径时以子进程工作目录为准。回放要求目录已经存在。 |
| `NCBI_API_KEY` | 空 | 可选的 PubMed API key。录制命令要求此变量为空或未设置。 |

## 录制

先试录 PubMed 的 KRAS G12C 检索和 ClinicalTrials.gov 的 CodeBreaK 200 不良事件模块。`--pilot pubmed`、`--pilot trials` 和 `--pilot labels` 可以分别试录一个请求。命令会打印每次请求的脱敏 URL、录制文件路径和大小。

```sh
env -u NCBI_API_KEY LIT_MCP_MODE=record \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.record --pilot
```

检查试录的结构和大小后补齐余下请求。`--remaining` 使用已录制的 PubMed 检索结果取得 PMID，并跳过目录里已有的请求。中途失败后可用同一命令继续。

```sh
env -u NCBI_API_KEY LIT_MCP_MODE=record \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.record --remaining
```

需要刷新整批响应时，用 `--source all` 指定一个新的录制目录。也可以按来源使用 `--source pubmed`、`--source trials` 或 `--source labels`。录制正命中的 PubMed 搜索时会自动追加 EFetch。要单独复录当前快照的摘要，可按原顺序传 PMID。

```sh
env -u NCBI_API_KEY LIT_MCP_MODE=record \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.record \
    --pubmed-ids 31666701 38072173 32430388 37782744 37956237
```

每个请求对应一个 UTF-8 JSON 文件，包含方法、去掉密钥后的 URL、HTTP 状态码、内容类型和响应正文。文件名是大写方法加一个空格再加规范化 URL 的 SHA-256，查询参数按名称和值排序。改动上游路径、查询或分页参数后，新的请求需要重新录制。

PubMed 每秒最多发 3 次请求，配置 key 时为 10 次。HTTPX 的连接、读、写和连接池超时各为 15 秒。429、5xx、超时和连接失败最多重试 3 次。`Retry-After` 有效时按上游指定的时间等待，否则依次等待 1、2、4 秒。openFDA 说明书接口的 `404 NOT_FOUND` 表示查询无结果，可以回放。回放只使用文件 transport，录制缺失会返回 `回放未命中` 和请求方法及脱敏 URL。

终止请求的 HTTP 错误会附带最多 2,048 个字符的脱敏响应正文，方便定位上游参数错误。错误消息保留原始状态码，正文读取失败时也会附上读取异常。

### 已录制的请求

以下 9 个 `GET` 请求共 492,638 字节。表中前两行是试录，其余通过 `--remaining` 补录。文件名前缀对应 `data/lit_recordings/` 中唯一的 JSON，完整 URL 可在文件的 `url` 字段查看。

| 请求 | 查询参数 | 状态 | 文件名前缀 | 字节 |
|---|---|---:|---|---:|
| PubMed `/entrez/eutils/esearch.fcgi` | `term="KRAS G12C"[Title/Abstract]` | 200 | `6d65bc0c9f` | 563 |
| ClinicalTrials.gov `/api/v2/studies/NCT04303780` | 不良事件模块投影 | 200 | `e412923543` | 74,493 |
| PubMed `/entrez/eutils/efetch.fcgi` | `db=pubmed`，`id=31666701,38072173,32430388,37782744,37956237`，`retmode=xml` | 200 | `6e2340ae39` | 91,788 |
| PubMed `/entrez/eutils/esearch.fcgi` | `term="pharmaflowtask2nomatchxq9z"[Title/Abstract]` | 200 | `a91359f994` | 694 |
| ClinicalTrials.gov `/api/v2/studies` | `query.cond=NSCLC`，`query.term=KRAS G12C`，`filter.advanced=AREA[Phase]PHASE3`，`countTotal=true`，`pageSize=5` | 200 | `5eed85e5e7` | 3,351 |
| ClinicalTrials.gov `/api/v2/studies/NCT04685135` | 不良事件模块投影 | 200 | `1cf02c7795` | 83,592 |
| openFDA `/drug/label.json` | `search=openfda.generic_name.exact:"SOTORASIB"` | 200 | `668805add1` | 120,388 |
| openFDA `/drug/label.json` | `search=openfda.generic_name.exact:"ADAGRASIB"` | 200 | `268662597e` | 117,407 |
| openFDA `/drug/label.json` | `search=openfda.generic_name.exact:"PHARMAFLOW-TASK2-NO-MATCH-XQ9Z"` | 404 | `12862f8f03` | 362 |

ESearch 共用 `db=pubmed`、`retmax=5`、`retmode=json`、`sort=relevance`。不良事件请求使用 `fields=NCTId,HasResults,AdverseEventsModule`。试验检索使用 `fields=NCTId,BriefTitle,Condition,InterventionName,Phase,OverallStatus`。说明书请求共用 `limit=1` 和 `sort=effective_time:desc`。

## 工具契约与裁剪

四个工具都接受单个 `request`。检索响应返回原始查询条件 `request`、上游总数 `total_count`、实际返回数 `returned_count` 和 `results`。成功的零结果使用计数 0 和空列表。HTTP、回放未命中或响应校验错误会传给调用方。

| 工具 | 选择与裁剪规则 | 修改后需要重录的情况 |
|---|---|---|
| `search_pubmed` | ESearch 默认查 5 条，上限 10 条，按相关性排序。合并 EFetch 摘要，保留 PMID、标题及期刊论文或图书条目类型，输出顺序与 ESearch 一致。 | 改检索式、`retmax`、排序或所取 PMID。 |
| `search_trials` | 按疾病、可选药物、分期及关键词检索。默认 5 条，上限 10 条。每条保留 NCT 号、标题、状态和最多各 20 项的疾病、干预和分期。 | 改条件、分页参数或上游 `fields`。 |
| `get_trial_adverse_events` | 保留源数据的严重／其他口径、`frequencyThreshold`、`timeFrame` 和报告说明。可用 `term` 筛选源事件术语，再按原顺序每类取前 50 个事件。保留所有组及各组报告的分母和计数。人数缺失为 `null`。`has_results=false` 时 `adverse_events=null`。 | 改 NCT 号或上游 `fields`。本地选择 `category`、`term` 或调整行数沿用同一份录制。 |
| `get_label_section` | 用精确通用名匹配，按生效日期降序取第一份，同日沿用上游排序。保留 `set_id`、`effective_time` 和请求的章节。普通章节最多取 5 段；缺失章节返回 `status=missing` 与 `content=null`。 | 改药名、上游排序或 `limit`。选择其他本地支持的章节沿用同一份录制。 |

PubMed 摘要、试验时间范围和说明书普通章节文本最多返回 4,000 个字符，附 `truncated` 和 `original_length`。`adverse_reactions_table` 的所有 HTML 块原样返回，带原始长度与 `truncated=false`，方便 Task 9 解析完整表格。修改本地文本长度或列表裁剪规则可复用录制文件。

`term` 按 ClinicalTrials.gov 的事件原文做大小写无关的子串匹配。查 ALT 升高时可用完整术语 `Alanine aminotransferase increased`。结果原样回显 `term`，每类事件分别提供 `total_count`（源表行数）、`matched_count`（匹配行数）和 `returned_count`（裁剪后的行数）。省略 `term` 时，匹配行数等于源表行数。`matched_count=0` 表示本次文字匹配没有命中。缺失或未请求的类别为 `null`。

## Inspector 回放

从项目根目录运行以下命令，分别查看 CodeBreaK 200 的严重不良事件与 sotorasib 说明书的原始 HTML 不良反应表。[MCP Inspector CLI 文档](https://github.com/modelcontextprotocol/inspector/blob/main/docs/cli-smoke-testing.md)

```sh
npx --yes @modelcontextprotocol/inspector@2.16.0 --cli \
  env FASTMCP_ENV_FILE=/dev/null LIT_MCP_MODE=replay LIT_MCP_RECORDINGS_DIR="$PWD/data/lit_recordings" \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.server -- \
  --method tools/call --tool-name get_trial_adverse_events \
  --tool-args-json '{"request":{"nct_id":"NCT04303780","category":"serious"}}' --format json

npx --yes @modelcontextprotocol/inspector@2.16.0 --cli \
  env FASTMCP_ENV_FILE=/dev/null LIT_MCP_MODE=replay LIT_MCP_RECORDINGS_DIR="$PWD/data/lit_recordings" \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.server -- \
  --method tools/call --tool-name get_label_section \
  --tool-args-json '{"request":{"drug":"sotorasib","fields":["adverse_reactions_table"]}}' --format json
```
