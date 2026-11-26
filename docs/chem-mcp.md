# chem-mcp 本地演示

在项目根目录安装锁定的依赖，然后启动 stdio 服务。

```sh
uv sync --locked
FASTMCP_ENV_FILE=/dev/null uv run --locked --no-env-file python -m mcp_servers.chem.server
```

服务等待 MCP 客户端通过 stdin 发送请求。演示命令需要 Node.js 和 `npx`，由 MCP Inspector 启动服务子进程。以下两条命令分别计算候选集性质和筛选候选集。[MCP Inspector CLI 文档](https://github.com/modelcontextprotocol/inspector/blob/main/docs/cli-smoke-testing.md)

```sh
npx --yes @modelcontextprotocol/inspector@2.16.0 --cli \
  env FASTMCP_ENV_FILE=/dev/null uv run --locked --no-env-file python -m mcp_servers.chem.server -- \
  --method tools/call --tool-name compute_properties \
  --tool-args-json '{"request":{"dataset_id":"candidates"}}' --format json

npx --yes @modelcontextprotocol/inspector@2.16.0 --cli \
  env FASTMCP_ENV_FILE=/dev/null uv run --locked --no-env-file python -m mcp_servers.chem.server -- \
  --method tools/call --tool-name filter_druglike \
  --tool-args-json '{"request":{"dataset_id":"candidates","thresholds":{"max_mw":600}}}' --format json
```

两个工具也接受 `{"request":{"smiles":["CCO"]}}`。内置数据集 ID 对应项目 `data/` 下的 CSV 文件名，去掉 `.csv` 后缀。上传接口返回的 ID 带 `upload_` 前缀，chem-mcp 从 `UPLOAD_DIR/<dataset_id>.csv` 读取，默认目录是 `data/uploads/`。CSV 使用 UTF-8 编码，列名必须唯一，且须包含 `smiles` 列。每条数据记录的字段数必须与表头一致。`molecule_chembl_id` 列可选。如果表头为 `smiles,molecule_chembl_id`，`CCC,` 表示 ID 为空，`CCC` 则缺少一个字段。两个工具均按输入顺序返回原始 SMILES 和 ChEMBL ID。直接传入 SMILES 列表、省略 ID 列或显式留空 ID 时，结果中的 `molecule_chembl_id` 为 `null`。

字段数不符时，工具停止处理该数据集，报告数据集 ID、CSV 行号和缺失列名或多余字段数。例如在 `name,smiles` 表头下，名称中带逗号应写为 `"batch,C",CCO`。写成 `batch,C,CCO` 会触发行宽错误。字段齐全但 SMILES 为空或非法时，仅该行返回 `invalid_smiles`，同批其他行继续计算。未知数据集返回 `Unknown dataset ID: <id>`。编码和 CSV 解析错误保留原始原因。

`filter_druglike` 默认上限为 MW 500 Da、LogP 5、HBD 5、HBA 10，允许 0 项超限。两个工具都使用 RDKit 的 `Lipinski.NumHDonors` 和 `Lipinski.NumHAcceptors` 计算氢键供体和受体数。可在 `thresholds` 中调整 `max_mw`、`max_logp`、`max_hbd`、`max_hba` 和 `max_violations`。`passes_filter` 要求通过 Lipinski 阈值检查且未命中 PAINS。

## 导出性质数据集

`export_properties` 接收 `{"request":{"dataset_id":"candidates"}}`，在 `UPLOAD_DIR` 下发布一个派生 CSV 并返回新 ID。有效行保留原始列及顺序，追加 `mw`、`logp`、`hbd`、`hba`、`tpsa`、`qed`。响应的 `invalid_rows` 列出被排除的非法 SMILES，以第一条数据行记为 1。源 CSV 含同名性质列时工具报错。全部非法时，新数据集 ID 为 `null`。可把新 ID 传给 `run_python` 做排序、统计和画图。 [沙盒演示](../README.md#演示代码自愈与图表)

## 候选集出处

`data/candidates.csv` 含 50 个去重化合物。2026-11-22 从 ChEMBL 37 导出，靶点为 `CHEMBL2189121`，变体为 `G12C`。入选记录的标准活性类型为 `IC50`、单位为 `nM`、关系为 `=`、标准化标志为 `1`，并带可解析的 ChEMBL SMILES。每行保留化合物、活性、assay 和文献 ID。[ChEMBL 活性查询](https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL2189121&assay_variant_mutation=G12C&standard_type=IC50&standard_units=nM&standard_relation=%3D) · [数据库版本](https://www.ebi.ac.uk/chembl/api/data/status?format=json)

先保留 sotorasib 和 adagrasib 各一条活性记录，再从 `CHEMBL5335102`、`CHEMBL6126514`、`CHEMBL4612347` 各选 16 个，按活性 ID 升序取样并按分子 ID 和 RDKit 规范化 SMILES 去重。IC50 数值保留各自 assay 的测定条件，适用于按原始记录追溯。

数据来源为 ChEMBL，许可为 CC BY-SA 3.0。[ChEMBL 数据许可](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions) · [ChEMBL 数据库论文](https://doi.org/10.1093/nar/gkad1004)
