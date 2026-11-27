# lit-mcp 本地录制与回放

准备 FastEmbed 缓存和 `kras_evidence` 索引后，从项目根目录启动 stdio 服务。测试和演示使用已录制的数据。 [本地准备命令](../README.md#在本地运行)

```sh
FASTMCP_ENV_FILE=/dev/null LIT_MCP_MODE=replay \
  LIT_MCP_RECORDINGS_DIR="$PWD/data/lit_recordings" \
  uv run --offline --locked --no-sync --no-env-file python -m mcp_servers.lit.server
```

stdio 客户端启动子进程时，在 `env` 中显式传入配置。服务在启动时读取环境变量，四个上游工具共享一个 `httpx.AsyncClient`。检索工具共享 Qdrant 客户端及模型，退出时关闭客户端。

| 配置 | 默认值 | 用途 |
|---|---|---|
| `LIT_MCP_MODE` | `replay` | `live` 访问上游，`record` 保存最终响应，`replay` 读取录制文件。 |
| `LIT_MCP_RECORDINGS_DIR` | 项目内的 `data/lit_recordings/` | 指定录制目录。显式传相对路径时以子进程工作目录为准。回放要求目录已经存在。 |
| `NCBI_API_KEY` | 空 | 可选的 PubMed API key。录制命令要求此变量为空或未设置。 |
| `QDRANT_URL` | `http://127.0.0.1:6333` | `kras_evidence` 所在的 Qdrant HTTP 地址。 |
| `FASTEMBED_CACHE_PATH` | `.cache/fastembed` | 三个检索模型的缓存目录。相对路径以项目根目录为基准。 |

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

## Task 7 证据语料

从项目根目录执行下列命令。录制命令将试验检索、两项试验的不良事件和两份说明书的共享响应逐字节复制到 `data/corpus_recordings/`，然后联网录制 PubMed。ESearch 按相关性取最多 200 条，EFetch 沿用该顺序，每 50 个 PMID 请求一次。录制和生成的 `data/corpus/chunks.jsonl` 一起纳入版本控制。

```sh
env -u NCBI_API_KEY LIT_MCP_MODE=record \
  uv run --locked --no-sync --no-env-file python -m rag.record_corpus
uv run --locked --no-sync --no-env-file python -m rag.download_models
uv run --offline --locked --no-sync --no-env-file python -m rag.build_corpus
```

`FASTEMBED_CACHE_PATH` 默认为项目根目录下的 `.cache/fastembed`。下载和构建都接受此环境变量，指定相对路径时以项目根目录为基准。构建只读取本地录制和模型缓存。两次回放使用相同的录制和 tokenizer 时，输出字节一致。终端报告包含 `chunk_count`、`corpus_sha256`、缺摘要的 PMID 和缺失的说明书章节。两份快照的 `boxed_warning` 均缺失。

每块正文含首行上下文，使用 `BAAI/bge-small-en-v1.5` 的 tokenizer 计数，含特殊 token 最多 448 个，为重排时拼接的扩展查询留约 60 个 token。构建使用 FastEmbed 0.9.1 的 ONNX 后端 `_ensure_tokenizer()` 读取缓存，并关闭 tokenizer 的截断和填充。摘要超长时先按结构化小节拆分，长文本继续按句子拆分。试验事件及说明书表格按行拆分，表格标题与表头在每块重复。单个句子或行加上必要上下文仍超出上限时，构建会报出相关 ID 和 token 数。

PubMed 优先使用 `PubDate/Year`，其次使用 `MedlineDate` 开头的四位年份。试验的共享录制仅投影不良事件模块，因此年份为 `null`，疗效终点未纳入语料。试验事件按严重／其他和器官系统分组，每行保留各组的原始人数分子和分母，并为每项试验生成报告概况。

说明书收录 `warnings_and_cautions`、`adverse_reactions` 和 `adverse_reactions_table`，年份取 `effective_time` 的前四位。HTML 表格按源 `<tr>` 展开，单元格用 ` | ` 连接。`rowspan` 和 `colspan` 的跨行、跨列关系会丢失，需结合原表表头阅读。JSONL 中每块的 `chunk_id` 可用于引用，去掉末尾序号即对应检索评测的相关单元。

## 索引与启动校验

运行 `uv run --offline --locked --no-sync --no-env-file python -m rag.index`，用 `chunks.jsonl` 重建 `kras_evidence`。命令先编码片段，再删除旧 collection 并写入新索引。点 ID 由固定命名空间和 `chunk_id` 计算 UUIDv5。稠密向量名为 `dense`，使用余弦距离。稀疏向量名为 `bm25`，由 Qdrant 计算 IDF。

`bm25_avg_len` 使用 FastEmbed 0.9.1 的 `Bm25` 内部接口。建索引先用已加载的 BM25 实例按 `raw_embed` 的流程移除标点并调用 `tokenizer.tokenize` 和 `_stem`，按去停用词、去超长词及词干化后的词数取平均。`_term_frequency` 使用同一份词列表的长度。文档编码由设置了这个 `avg_len` 的第二个 `SparseTextEmbedding` 实例执行。查询编码的词权重为 1。

lit-mcp 在同一个工作线程顺序从缓存加载 dense、BM25 和重排模型。启动检查 collection 是否存在、三个模型名、语料文件 SHA-256 及 `count(exact=True)`。失配或加载出错会中止启动。启动就绪时在 stderr 记录 `lit-mcp ready in … s`。

2026-11-27 的本地验收确认 `qdrant/qdrant:v1.20.0` 提供 `/usr/bin/bash`，Compose 健康检查通过。索引包含 437 个点，`bm25_avg_len` 为 `106.98627002288329`。在 macOS 上使用本地模型缓存跑集成测试，日志记录 14 次 stdio 启动，耗时为 0.777–0.842 秒，中位数 0.788 秒。39 项集成测试在 24.34 秒内全部通过。

## 工具契约与裁剪

五个工具都接受单个 `request`。上游列表响应返回原始查询条件 `request`、上游总数 `total_count`、实际返回数 `returned_count` 和 `results`。成功的零结果使用计数 0 和空列表。HTTP、回放未命中或响应校验错误会传给调用方。

| 工具 | 选择与裁剪规则 | 修改后需要重录的情况 |
|---|---|---|
| `search_pubmed` | ESearch 默认查 5 条，上限 10 条，按相关性排序。合并 EFetch 摘要，保留 PMID、标题及期刊论文或图书条目类型，输出顺序与 ESearch 一致。 | 改检索式、`retmax`、排序或所取 PMID。 |
| `search_trials` | 按疾病、可选药物、分期及关键词检索。默认 5 条，上限 10 条。每条保留 NCT 号、标题、状态和最多各 20 项的疾病、干预和分期。 | 改条件、分页参数或上游 `fields`。 |
| `get_trial_adverse_events` | 保留源数据的严重／其他口径、`frequencyThreshold`、`timeFrame` 和报告说明。可用 `term` 筛选源事件术语，再按原顺序每类取前 50 个事件。保留所有组及各组报告的分母和计数。人数缺失为 `null`。`has_results=false` 时 `adverse_events=null`。 | 改 NCT 号或上游 `fields`。本地选择 `category`、`term` 或调整行数沿用同一份录制。 |
| `get_label_section` | 用精确通用名匹配，按生效日期降序取第一份，同日沿用上游排序。保留 `set_id`、`effective_time` 和请求的章节。普通章节最多取 5 段；缺失章节返回 `status=missing` 与 `content=null`。 | 改药名、上游排序或 `limit`。选择其他本地支持的章节沿用同一份录制。 |

PubMed 摘要、试验时间范围和说明书普通章节文本最多返回 4,000 个字符，附 `truncated` 和 `original_length`。`adverse_reactions_table` 的所有 HTML 块原样返回，带原始长度与 `truncated=false`，方便 Task 9 解析完整表格。修改本地文本长度或列表裁剪规则可复用录制文件。

`term` 按 ClinicalTrials.gov 的事件原文做大小写无关的子串匹配。查 ALT 升高时可用完整术语 `Alanine aminotransferase increased`。结果原样回显 `term`，每类事件分别提供 `total_count`（源表行数）、`matched_count`（匹配行数）和 `returned_count`（裁剪后的行数）。省略 `term` 时，匹配行数等于源表行数。`matched_count=0` 表示本次文字匹配没有命中。缺失或未请求的类别为 `null`。

`search_evidence` 的 `request` 只有英文 `query`。Agent 先把中文问题转为英文检索词。工具展开查询中的药名别名，再让 Qdrant 用 RRF 融合 dense 和 BM25 各自的前 30 条。客户端按 RRF 分数降序排列全部融合候选，同分按稳定点 ID 升序排列，取前 30 条。cross-encoder 重排后返回最多 8 条。结果包含原始与扩展查询、`citation_id`、来源及原文链接、片段正文和 `rerank_score`。检索由工作线程执行，Qdrant 或模型错误会传给调用方。

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
