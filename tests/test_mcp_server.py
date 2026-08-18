"""测试三个 MCP Server 是否能正常 list_tools + call_tool"""
import asyncio
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv;

load_dotenv()
from mcp_servers.mcp_client import MCPClient


async def main():
    client = MCPClient()
    await client.connect()

    # 列出所有工具
    tools = client.get_tool_meta()
    print(f"发现 {len(tools)} 个 MCP 工具：")
    for t in tools:
        print(f"  - {t['name']}: {t['description'][:60]}...")

    # 测试 search_documents
    result = await client.call_tool("search_documents", {"query": "CS-Pro SLA"})
    print(f"\nsearch_documents 返回 {len(result)} 字符")
    print(result[:200])

    # ★ 显式关闭所有连接——避免 Windows 上 anyio cancel scope 清理时报 RuntimeError
    await client.close()


asyncio.run(main())