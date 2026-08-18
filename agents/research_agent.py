"""ResearchAgent: 制定检索策略 → 调用 MCP 工具 → 初步分析结果"""
import re
from backend.services.rag_service import llm  # 复用 DeepRAG 的 llm 客户端
from mcp_servers.mcp_client import get_mcp_client
from agents.state import CollaborationState


async def research_node(state: CollaborationState) -> dict:
    """ResearchAgent 节点函数

    三步流程：
    ① LLM 制定检索策略（plan）
    ② 通过 MCP 协议调用 search_documents 工具
    ③ LLM 对检索结果做初步分析
    """
    client = await get_mcp_client()

    # 第一步：制定检索策略
    plan_prompt = (
        f"用户问题：{state['query']}\n"
        f"请制定检索策略：需要查哪些资料？用什么关键词？最多 5 句话。"
    )
    plan = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": plan_prompt}],
    ).choices[0].message.content

    # 第二步：调用 MCP 工具执行检索
    # ⚠️ 当前硬编码只调 search_documents；web_search / query_database 的动态选择
    #    是下一阶段改造（改造后由 LLM 决定调哪些工具，并循环调用、逐条记录）。
    tools_called: list[str] = []
    search_results = await client.call_tool(
        "search_documents", {"query": state['query']}
    )
    tools_called.append("search_documents")
    # 从格式化文本中提取所有 chunk_id（格式：[来源: xxx]）
    sources = re.findall(r'\[来源:\s*([^\]]+)\]', search_results)

    # 第三步：LLM 对结果做初步分析
    analysis_prompt = (
        # DeepSeek v4-pro 上下文 128K token，MCP 检索结果通常仅 3K~5K 字符，
        # 远不到窗口零头，直接全量传入即可，无需截断。
        #
        # 若未来文档库膨胀、检索返回数万字符，取消下方注释启用安全截断：
        # （按最后一个换行符切，避免切断 [来源: xxx] 标记或半截句子）
        #
        # _raw = search_results
        # _limit = 8000  # 字符上限，按实际模型窗口调整
        # if len(_raw) > _limit:
        #     _cut = _raw.rfind('\n', 0, _limit)
        #     _text = _raw[:_cut] + "\n...(内容超出限制，后续已省略)"
        # else:
        #     _text = _raw
        #
        f"检索结果：\n{search_results}\n\n"
        f"请分析：信息是否充分？关键发现是什么？最多 10 句话。"
    )
    analysis = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": analysis_prompt}],
    ).choices[0].message.content

    return {
        "research_plan": plan,
        "search_results": search_results,     # str（格式化文本，供 Report 阅读）
        "search_sources": sources,            # list[str]（chunk_id，供 Report 提取引用）
        "analysis": analysis,
        "tools_called": tools_called,         # list[str]（Level 2 工具选择评估依据）
    }