"""Expose the chem-mcp tools over stdio."""

from fastmcp import FastMCP

from .tools import compute_properties, export_properties, filter_druglike

mcp = FastMCP("chem-mcp")
mcp.tool(compute_properties)
mcp.tool(export_properties)
mcp.tool(filter_druglike)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
