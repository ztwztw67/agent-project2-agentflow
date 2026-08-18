"""MCP Server —— 数据库查询工具（基于 SQL Agent）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from mcp.server.fastmcp import FastMCP
from backend.services.agent_tools import _get_sql_agent

mcp = FastMCP("query-database")


@mcp.tool()
async def query_database(question: str) -> str:
    """查询云枢科技业务数据库。

    适用场景：统计数据、用户信息、订单记录等结构化数据查询。
    输入自然语言问题，自动转换为 SQL 并执行。

    Args:
        question: 自然语言问题，如 '上个月新增了多少用户？'
    """
    result = _get_sql_agent().invoke({"input": question})
    output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
    return output


if __name__ == "__main__":
    mcp.run()