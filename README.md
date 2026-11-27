# PharmaFlow Agent

## 构建 Task 7 证据语料

从项目根目录录制 PubMed 摘要、准备 FastEmbed 缓存，再回放构建 `data/corpus/chunks.jsonl`。随仓库提供的试验和说明书录制也会用于构建。 [录制与构建命令](docs/lit-mcp.md#task-7-证据语料)

## 在本地运行

在项目根目录安装依赖，启动 Postgres 和 Qdrant。使用仓库中的 `chunks.jsonl` 建好证据索引。API、CLI 和自愈评测使用本机 Docker daemon 执行脚本。

```sh
uv sync --locked
docker compose up -d --wait postgres qdrant
uv run --locked --no-env-file python -m rag.download_models
uv run --offline --locked --no-sync --no-env-file python -m rag.index
docker build -t pharmaflow-sandbox:1 sandbox
```

按 `.env.example` 配置模型和数据库。在两个终端分别启动 API 和 Streamlit。 [模型和评测设置](docs/eval.md)

```sh
uv run uvicorn api.main:app
uv run streamlit run ui/app.py
```

每次 `run_python` 使用一个一次性容器，关闭网络，以非 root 用户运行。CPU 配额为 1 核，内存上限为 512 MiB，额外 swap 配额为 0，进程数上限为 64。启动后等待脚本最多 15 秒。CSV 通过 Docker archive API 传入 `/work/input/`，PNG 从 `/work/output/` 取回。每次最多 5 张图，单张最多 5 MiB，合计最多 10 MiB。成功取回的 PNG 存在本地 `data/artifacts/`，会话刷新时按 UUID 取图。

容器可写层共享 Docker Desktop 的磁盘空间，单次脚本运行可能耗尽可用空间。产物大小限制在从容器取回 PNG 时生效。

如果将 API 容器接入宿主机的 `/var/run/docker.sock`，API 将取得足以控制宿主机的 Docker 权限。这个部署方式仅用于本地演示。生产环境可采用 gVisor、Firecracker 或独立的托管沙盒服务。

## 演示证据检索

在聊天中发送下列问题。展开执行详情，查看 Agent 传给 `search_evidence` 的英文检索词和工具结果预览。用侧栏的会话 ID 请求 `GET /api/threads/{thread_id}`，可查看展开后的查询和完整证据片段。

> 跨文献、试验和说明书查一下 sotorasib 的肝毒性证据。请附上你引用的英文原文片段、片段 ID 和原文链接。

核对答复里的片段 ID 和链接与工具返回一致。演示时附上 dense、混合召回和混合召回加重排的 Recall@8 和 MRR。 [检索评测](docs/eval.md#task-7-检索评测)

## 演示文献和分子并行分析

CLI、API 和评测默认使用 `AGENT_GRAPH_MODE=single`。启动 API 时设为 `multi`，在 Streamlit 新建会话。已有的 Postgres、Qdrant 证据索引和 Langfuse 配置继续使用。

```sh
AGENT_GRAPH_MODE=multi uv run --offline --locked --no-sync --no-env-file uvicorn api.main:app
uv run --offline --locked --no-sync --no-env-file streamlit run ui/app.py
```

> 请用 `search_evidence` 查 sotorasib 的肝毒性证据，给出英文片段、片段 ID 和原文链接。同时计算 `CCO` 和 `CC(=O)O` 的分子量及 TPSA，按输入顺序单列结果。

聊天的执行详情显示子任务 ID 和工具调用，答复由 Reporter 流式输出。在 Langfuse 按侧栏的会话 ID 过滤 session，打开该轮 trace，核对 `LitResearcher` 和 `MolAnalyst` 的执行时间有重叠，并在两个分支之后看到 `Critic` 和 `Reporter`。 [追踪设置和评测命令](docs/eval.md#task-8-多智能体与上下文压缩)

多智能体父图的 `step_count` 是本轮已回传的 worker ReAct 模型调用数，包含 Critic 重派，不含 Planner、Reporter 和摘要。每次派发的 worker 各自使用请求的 `max_steps`。后续轮次可以复用历史答复中已给出的派生数据集 ID。仅存在于分支工具结果里的 ID 随该轮分支记录归档。

刷新时，历史接口从各轮已完成的父图检查点恢复工具记录和图表，按 Planner 的子任务顺序展示，每轮用户消息只出现一次。上一轮待执行或中途出错时，仍可看到该轮原始问题和已回传的工具结果。恢复执行后 Reporter 才会补上答复。Streamlit 重跑会再次读取历史接口，服务端在有历史分支时会扫描该会话的父图检查点，读取成本随会话长度增加。

## 演示代码自愈与图表

在 Streamlit 侧栏上传 `data/candidates.csv`，发送下面的演示消息。给定起始脚本里的大写 `QED` 列用于触发 `KeyError`，派生 CSV 提供的是小写 `qed`。

> 按 QED 降序取前 10，列出分子名称与 QED，并画前 10 的 MW–LogP 散点图。先用 export_properties 处理本轮上传的 CSV。首次执行请使用下面的脚本，只替换 DERIVED 为派生数据集 ID，并把该 ID 放入 dataset_ids。看到执行错误后，自行修正脚本，打印前 10，保存散点图为 /work/output/mw-logp.png。

```python
import pandas as pd

data = pd.read_csv("/work/input/DERIVED.csv")
print(data["QED"].nlargest(10))
```

展开聊天里的执行详情查看首次代码、traceback 和修正版。最终图表显示在答复内，刷新同一个会话仍可查看。

## 验收与计时

`pytest` 默认排除带 `docker` 标记的测试。显式传入 `-m docker` 运行容器测试。集成测试使用已启动的 Postgres、构建好的 Qdrant 索引和 FastEmbed 缓存。

```sh
uv run --no-sync --no-env-file pytest tests/unit
uv run --no-sync --no-env-file pytest tests/integration
uv run --no-sync --no-env-file pytest -m docker tests/docker
```

启动 Docker Desktop 后，在项目根目录运行下面的命令。脚本顺序执行 5 次空脚本，逐次报告创建、删除及六阶段合计耗时，再输出创建与删除之和、六阶段合计的中位数。

```sh
uv run --no-sync --no-env-file python -m scripts.benchmark_sandbox
```

### Docker Desktop 实测

使用已构建的 `pharmaflow-sandbox:1` 镜像顺序运行 5 次空脚本 `pass`。

| 样本 | 创建 (ms) | 删除 (ms) | 创建＋删除 (ms) | 六阶段合计 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.8 | 3.7 | 23.5 | 77.7 |
| 2 | 10.3 | 3.3 | 13.6 | 62.1 |
| 3 | 11.7 | 3.3 | 15.0 | 67.5 |
| 4 | 22.1 | 3.2 | 25.3 | 82.6 |
| 5 | 11.9 | 3.2 | 15.1 | 63.0 |

五次采样的创建与删除合计中位数为 **15.1 ms**，六阶段合计中位数为 **67.5 ms**。

开发期一轮自愈评测中，3 条用例共执行 4 次修正，修正阶段平均耗时 **7.583 s**。用五次空脚本采样的中位数粗估，4 次创建与删除共约 60.4 ms，占修正阶段总耗时 22.749 s 的 **0.27%**。实际占比以及模型调用、脚本执行和检查点分别用了多久，需要在同一轮评测中采集分段数据。 [开发期自愈评测](docs/eval.md#修正规则后的开发期自愈复跑)

已有评测库的迁移和独立自愈评测命令见评测手册。 [Task 6 自愈评测](docs/eval.md#task-6-自愈评测)
