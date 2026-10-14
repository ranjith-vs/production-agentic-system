import os

from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://mcp-server:9000")


async def load_mcp_tools():
    client = MultiServerMCPClient(
        {
            "demo": {
                "url": f"{MCP_SERVER_URL}/mcp",
                "transport": "streamable_http",
            }
        }
    )
    return await client.get_tools()
