"""Agent 对话的 LangFuse 追踪封装

每个用户请求创建一个 LangFuse trace，记录完整调用链和最终结果。
graph 作为参数传入（而非引用全局变量），便于测试时传入 mock graph。

⚠️ SDK v3 变更（2025）：
   - langfuse.trace() 已移除，改为 start_as_current_span() 上下文管理器
   - trace 级属性（session_id / tags / input / output）通过 span.update_trace() 设置
   - CallbackHandler 创建的 LLM span 会自动嵌套在此 span 下
   - 参考：https://langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3
"""
from langfuse import Langfuse
from backend.config import settings
from observability.langfuse_callback import langfuse_handler

langfuse = Langfuse(
    secret_key=settings.langfuse_secret_key,
    public_key=settings.langfuse_public_key,
    host=settings.langfuse_host,
)


async def traced_chat(graph, query: str, session_id: str) -> dict:
    """执行 Agent 对话 + 全链路 LangFuse 追踪

    Args:
        graph: 编译后的 LangGraph agent graph（由 build_agent_graph() 返回）
        query: 用户问题
        session_id: 会话 ID（LangFuse 按 session 聚合展示）

    Returns:
        Agent 执行结果 dict，含 final_answer / sources 等字段
    """
    # SDK v3: start_as_current_span() 进入时创建 trace（顶层 span），
    #        退出时自动关闭并上报。CallbackHandler 的 LLM span 会自动嵌套在此 span 下
    with langfuse.start_as_current_span(name="agent-chat") as span:
        # 设置 trace 级别属性（在 Dashboard 中按这些字段筛选/聚合）
        span.update_trace(
            session_id=session_id,
            tags=["agent-agentflow-pro"],
            input={"query": query},
        )

        # ★ ainvoke + callbacks 传入 LangFuse handler
        #   CallbackHandler 自动拦截 LLM 调用，生成子 span 挂在此 trace 下
        result = await graph.ainvoke(
            {"query": query},
            config={"callbacks": [langfuse_handler]},
        )

        # 补上 trace 级别的业务结果（CallbackHandler 不感知业务语义）
        span.update_trace(
            output=result.get("final_answer", ""),
            metadata={
                "sources_count": len(result.get("sources", [])),
                "info_sufficient": result.get("info_sufficient"),
            },
        )

    return result