import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_servers.chem import tools as chem_tools
from mcp_servers.chem.models import (
    DruglikeMolecule,
    DruglikeResult,
    ExportPropertiesResult,
    InvalidMolecule,
    MoleculeProperties,
    PropertiesResult,
)
from mcp_servers.chem.server import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def test_chem_tools_over_one_stdio_session(tmp_path):
    client = MultiServerMCPClient(
        {
            "chem": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_servers.chem.server"],
                "cwd": str(tmp_path),
                "env": {
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "FASTMCP_ENV_FILE": os.devnull,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "UPLOAD_DIR": str(tmp_path),
                },
            }
        }
    )

    async with asyncio.timeout(30), client.session("chem") as session:
        tools = await session.list_tools()
        assert {tool.name for tool in tools.tools} == {
            "compute_properties",
            "export_properties",
            "filter_druglike",
        }
        assert all(tool.outputSchema for tool in tools.tools)

        computed = await session.call_tool(
            "compute_properties", {"request": {"dataset_id": "candidates"}}
        )
        screened = await session.call_tool(
            "filter_druglike",
            {"request": {"dataset_id": "candidates", "thresholds": {"max_mw": 600}}},
        )
        mixed = await session.call_tool(
            "compute_properties", {"request": {"smiles": ["not-smiles", "CCO"]}}
        )
        exported = await session.call_tool(
            "export_properties", {"request": {"dataset_id": "candidates"}}
        )

    assert not computed.isError, computed.content
    properties = PropertiesResult.model_validate(computed.structuredContent).results
    assert len(properties) == 50
    assert all(isinstance(row, MoleculeProperties) for row in properties)
    assert properties[0].mw == pytest.approx(560.605, abs=0.01)
    assert properties[0].molecule_chembl_id == "CHEMBL4535757"

    assert not screened.isError, screened.content
    filters = DruglikeResult.model_validate(screened.structuredContent).results
    assert [row.smiles for row in filters] == [row.smiles for row in properties]
    assert all(isinstance(row, DruglikeMolecule) for row in filters)
    assert filters[0].passes_lipinski
    assert filters[0].molecule_chembl_id == "CHEMBL4535757"

    assert not mixed.isError, mixed.content
    assert mixed.structuredContent["results"][0]["error"] == "invalid_smiles"
    invalid, ethanol = PropertiesResult.model_validate(mixed.structuredContent).results
    assert invalid == InvalidMolecule(smiles="not-smiles")
    assert isinstance(ethanol, MoleculeProperties)
    assert ethanol.mw == pytest.approx(46.069, abs=0.01)

    assert not exported.isError, exported.content
    export = ExportPropertiesResult.model_validate(exported.structuredContent)
    assert export.row_count == 50
    assert {"mw", "logp", "qed"} <= set(export.columns)
    assert (tmp_path / f"{export.dataset_id}.csv").is_file()


@pytest.mark.parametrize(
    ("csv_bytes", "reason"),
    [
        (b"smiles\n\xff\n", "invalid start byte"),
        (b'smiles\n"CCO', "unexpected end of data"),
    ],
)
async def test_csv_read_error_reaches_mcp_client(
    tmp_path, monkeypatch, csv_bytes, reason
):
    monkeypatch.setattr(chem_tools, "_DATA_DIR", tmp_path)
    (tmp_path / "broken.csv").write_bytes(csv_bytes)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "compute_properties",
            {"request": {"dataset_id": "broken"}},
            raise_on_error=False,
        )

    assert result.is_error
    assert "broken" in result.content[0].text
    assert reason in result.content[0].text
