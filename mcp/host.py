import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_scraping(targets):
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.scraping_server"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("scrape_all", {"targets": targets})
            return result.content[0].text