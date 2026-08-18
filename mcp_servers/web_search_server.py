"""MCP Server —— Web 搜索工具（基于 Tavily API）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from mcp.server.fastmcp import FastMCP
from langchain_community.tools.tavily_search import TavilySearchResults
from backend.config import settings

mcp = FastMCP("web-search")


@mcp.tool()
async def web_search(query: str) -> str:
    """在互联网上搜索最新信息。

    适用场景：新闻事件、实时数据、最新动态等知识库中不存在的信息。

    Args:
        query: 搜索关键词
    """
    tavily = TavilySearchResults(max_results=3, tavily_api_key=settings.tavily_api_key)
    results = tavily.invoke({"query": query})
    formatted = "\n\n---\n\n".join([
        f"[{r.get('title', '无标题')}]({r.get('url', '')})\n{r.get('content', '')[:300]}"
        for r in (results if isinstance(results, list) else [])
    ])
    return formatted or "未找到相关结果"


if __name__ == "__main__":
    mcp.run()