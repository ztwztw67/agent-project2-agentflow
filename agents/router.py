"""LangGraph 条件边：判断 ResearchAgent 下一步走向

这是旧版 README 缺失的关键函数 —— add_conditional_edges 需要此函数返回值
来决定走 "call_tools"（继续检索）还是 "report"（交给 ReportAgent）。
"""
from agents.state import CollaborationState


def route_tool_use(state: CollaborationState) -> str:
    """判断当前应继续检索还是交给 ReportAgent 生成回答

    决策逻辑：
    - 有 search_results + analysis → 交给 report
    - 否则 → 继续调用工具
    """
    # 用 is not None 而非 truthiness 检查：
    # search_results 可能是空字符串 ""（知识库为空），但检索确实执行了
    if state.get("search_results") is not None and state.get("analysis") is not None:
        return "report"
    return "call_tools"