from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-tools", host="0.0.0.0", port=9000, stateless_http=True)


@mcp.tool()
def get_current_time() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
