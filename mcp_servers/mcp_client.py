"""MCP Client —— Agent 通过此模块统一调用所有 MCP 工具

关键设计：
1. 维持持久 session（不是每次调用都重新启动 Server 进程）
2. 用工厂函数模式解决闭包陷阱（每个 tool 的函数闭包捕获自己对应的 tool_name）
3. 提供 build_langchain_tools() 将 MCP 工具转为 LangChain 兼容格式
4. 用 AsyncExitStack 管理嵌套 async context manager 的生命周期——
   避免 Windows 上 anyio cancel scope 的跨 task 退出异常
"""
import sys
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 每个 MCP Server 的启动配置
SERVER_CONFIGS = {
    "search_docs": [sys.executable, "mcp_servers/search_docs_server.py"],
    "web_search":  [sys.executable, "mcp_servers/web_search_server.py"],
    "query_db":    [sys.executable, "mcp_servers/query_db_server.py"],
}


class MCPClient:
    """统一 MCP Client —— 管理到所有 MCP Server 的持久连接"""

    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self._tool_meta: list[dict] = []  # [{name, description, inputSchema, server}, ...]

    async def connect(self):
        """启动时调用一次：连接所有 MCP Server 并发现工具"""
        for name, cmd in SERVER_CONFIGS.items():
            server_params = StdioServerParameters(command=cmd[0], args=cmd[1:])
            # ★ AsyncExitStack.enter_async_context 等价于 async with，
            #    但保持上下文存活直到 close() 调用
            transports = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = transports
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[name] = session

            # 发现此 Server 提供的所有工具
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                self._tool_meta.append({
                    "server": name,
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                })

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用指定工具 —— 自动找到对应的 MCP Server"""
        for session in self.sessions.values():
            try:
                result = await session.call_tool(tool_name, arguments)
                return "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
            except Exception:
                continue
        raise ValueError(f"Tool '{tool_name}' not found on any connected server")

    def get_tool_meta(self) -> list[dict]:
        """返回所有已发现工具的元数据（用于注入 LLM system prompt）"""
        return self._tool_meta

    async def close(self):
        """关闭所有 MCP Server 连接（LIFO：先关 session 再关 transport）"""
        await self._exit_stack.aclose()
        self.sessions.clear()


# ====== 工厂函数 —— 修复闭包陷阱 ======
# ❌ 错误的写法（for 循环闭包捕获引用，所有 tool 最终指向最后一次迭代的值）：
#   for tool_meta in mcp_client.get_tool_meta():
#       @tool(name=tool_meta["name"])  # ← tool_meta 是引用，循环结束后全是最后一个
#       async def mcp_bridge(query): ...
#
# ✅ 正确的写法（工厂函数捕获参数值，每次迭代独立绑定）：
def _make_mcp_bridge(mcp_client: MCPClient, tool_name: str, tool_desc: str):
    """工厂函数：为指定 tool 创建独立的 async callable"""
    async def _bridge(query: str) -> str:
        return await mcp_client.call_tool(tool_name, {"query": query})
    _bridge.__name__ = tool_name
    _bridge.__doc__ = tool_desc
    return _bridge


def build_langchain_tools(mcp_client: MCPClient) -> list:
    """将所有 MCP 工具转换为 LangChain 兼容的 Tool 列表

    每个 Tool 的 callable 通过 _make_mcp_bridge 工厂函数创建，
    确保闭包中捕获的是正确的 tool_name（而非循环引用的最后一个）。
    """
    tools = []
    for tool_meta in mcp_client.get_tool_meta():
        tool_name = tool_meta["name"]
        tool_desc = tool_meta["description"]

        # 创建 Pydantic 输入 schema
        class ToolInput(BaseModel):
            query: str = Field(description=tool_desc)

        langchain_tool = StructuredTool(
            name=tool_name,
            description=tool_desc,
            args_schema=ToolInput,
            coroutine=_make_mcp_bridge(mcp_client, tool_name, tool_desc),
        )
        tools.append(langchain_tool)

    return tools


# ====== 全局单例（FastAPI lifespan 中初始化） ======
_mcp_client: MCPClient | None = None

async def get_mcp_client() -> MCPClient:
    """获取全局 MCP Client（延迟初始化，首次调用时自动连接）"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        await _mcp_client.connect()
    return _mcp_client