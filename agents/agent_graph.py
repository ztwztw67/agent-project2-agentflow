"""编译完整的 Agent 协作图

这是整个 AgentFlow Pro 的编排核心 —— 组装所有节点、边和条件路由。
build_agent_graph() 返回的 compiled graph 可被 FastAPI / CLI / 测试调用。

架构说明：
- research_node 内部直接调用 MCP 工具（而非通过 LangChain Tool Calling），
  因此不需要 ToolNode。Agent 间通过 CollaborationState 共享数据。
- 图结构简洁：research → report → END
- route_tool_use 保留条件边，当前仅路由到 report（为后续扩展预留 call_tools 分支）
"""
from langgraph.graph import StateGraph, END
from agents.state import CollaborationState
from agents.research_agent import research_node
from agents.report_agent import report_node
from agents.router import route_tool_use


async def build_agent_graph():
    """构建并编译 Agent 协作图

    Returns:
        编译后的 LangGraph graph（可用 .ainvoke() 或 .invoke() 执行）

    图结构：
        research ──→ report → END

    数据流：
        research_node 写入 state: research_plan, search_results, search_sources, analysis
        report_node 读取上述字段，写入: info_sufficient, final_answer, sources
    """
    builder = StateGraph(CollaborationState)
    builder.add_node("research", research_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("research")

    # 条件边：route_tool_use 判断 research → 下一步
    # 当前 research_node 内联完成检索，始终走向 report；
    # "call_tools" 分支保留用于后续扩展（如引入 ToolNode 或 Agent 循环）
    builder.add_conditional_edges(
        "research",
        route_tool_use,
        {"call_tools": "report", "report": "report"}
    )
    builder.add_edge("report", END)

    return builder.compile()
