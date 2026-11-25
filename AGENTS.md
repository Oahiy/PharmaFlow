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

## 错误处理和防御性代码

出错时让错误尽早暴露，并带上原始信息，不要用默认值或兜底逻辑盖住。这个项目里静默兜底比崩溃更糟：检索失败时如果返回空列表，下游会把"没查到"当成"没有不良事件证据"，评测里也看不出是工具出了错。

- 校验放在数据进入系统的地方，只做一次，优先用 Pydantic 模型。这些地方是：MCP 工具和 FastAPI 的输入、上传文件和 CSV、环境变量、外部接口的响应（在 lit-mcp 里）、LLM 的结构化输出、HITL 恢复时传回的数据。校验通过后，内部代码直接使用，不再重复检查。
- 外部数据里可能缺失的字段，在模型里标成 `X | None`，下游显式处理 None。不要用 0、空字符串或空列表代替缺失值：不良事件发生率缺失时填 0，看起来就成了"没有发生"。
- 只捕获能处理的具体异常，其余的让它抛出去。MCP 工具出错直接 raise，`langchain-mcp-adapters` 会把错误回给模型。包装异常时用 `raise ... from exc`，并在新消息里带上原始错误。
- 不要用 `except Exception` 包住一段逻辑，再打日志、返回 None 或空结果继续跑。宽泛捕获只放在最外层（FastAPI 的异常处理器、SSE 生成器的最外层），捕获后记录完整 traceback，并把错误返回给调用方。
- 不要用 `isinstance`、`hasattr`、`getattr(obj, name, None)` 重复检查类型标注或校验已经保证的事，也不要用 `x or []`、`d.get(key, default)` 填补按约定不会缺失的值。类型是 `X | None` 但按约定不会是 None 时，用 `assert x is not None` 收窄，或者改模型让类型里不含 None。
- 图节点读取上游节点一定会写入的 State 字段时直接取值，不写 `state.get("x", [])`。真正可选的字段在 State 类型里标成可选。
- 修 KeyError、AttributeError、`NoneType` 这类报错时，先查清值为什么缺失或类型不对，在源头修。不要在报错的那一行加判断、默认值或 try/except 让报错消失。
- 边界校验照常写测试。内部函数不为校验已经排除的输入写测试，也不写断言"出错时返回默认值"的测试，这类测试会把兜底行为固定下来。
- Python 固定 3.12，依赖版本由 `uv.lock` 锁定。不写 `try: import ... except ImportError`、版本判断或新旧 API 两套写法。不确定某个 API 在当前版本里有没有，就去看 `.venv` 里的源码。
- 项目还没有外部调用方。改函数签名、模型字段或工具参数时，直接改掉所有调用处，不保留旧名字、别名或废弃参数。
- 不加计划里没有的重试、降级路径和配置开关。`docs/plan.md` 设计好的超时、重试、限流和自愈照做，只放在计划指定的位置：lit-mcp 的 HTTP 客户端、LLM 客户端、代码沙盒。

下面这些是正常逻辑，要保留：库函数按文档会返回 None 的情况（如 `Chem.MolFromSmiles` 遇到非法 SMILES 返回 None）；批量接口按设计逐条返回的错误项（如 `InvalidMolecule`）；路径穿越检查、沙盒的资源和网络限制这类安全措施。

拿不准某个值会不会为空、某个异常会不会发生时，先看类型定义和调用方，不要先加一层判断。确实认为需要额外防御的，在做完后的列表里单独列出位置和它防的具体场景。这一节只约束你新写和修改的代码，不要顺手改无关文件里已有的写法。


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
