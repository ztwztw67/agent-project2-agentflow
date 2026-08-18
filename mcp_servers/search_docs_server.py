"""MCP Server —— 本地知识库检索工具

将 DeepRAG 的 search_v3() 包装为标准 MCP 工具。
运行方式：mcp dev mcp_servers/search_docs_server.py（开发调试）
         或 python mcp_servers/search_docs_server.py（生产 stdio 通信）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from backend.services.rag_service import search_v3  # 复用 DeepRAG 的完整检索链路

mcp = FastMCP("search-documents")


@mcp.tool()
async def search_documents(query: str) -> str:
    """在云枢科技产品知识库中检索相关文档片段。

    适用场景：产品规格、定价、SLA条款、功能说明、更新日志等。
    返回形式：文档片段 + 来源 chunk_id 标注。

    Args:
        query: 搜索查询，如 'CS-Pro 的 SLA 可用性是多少？'
    """
    results = search_v3(query)  # ★ 复用 DeepRAG 的 rewrite→ensemble→rerank 完整链路
    # results 格式：[(page_content, metadata), ...]
    formatted = "\n\n---\n\n".join([
        f"[来源: {meta.get('chunk_id', 'unknown')}]\n{content[:500]}"
        for content, meta in results
    ])
    return formatted


if __name__ == "__main__":
    mcp.run()  # FastMCP 自动处理 stdio transport
