"""ReportAgent: 评估信息充分性 → 生成最终回答（或追加检索）"""
import re
from backend.services.rag_service import llm
from agents.state import CollaborationState


def extract_followup(llm_output: str) -> str:
    """从 LLM 的评估输出中提取追加检索方向

    使用正则匹配"还需要xxx"模式，失败时回退到最后一句有意义的话。
    """
    match = re.search(
        r'还需要[查询了解确认]{1,3}[：:]\s*(.+?)(?:[。！\n]|$)',
        llm_output
    )
    if match:
        return match.group(1).strip()
    # 回退：取最后一句足够长的句子
    sentences = re.split(r'[。！\n]', llm_output)
    for s in reversed(sentences):
        s = s.strip()
        if len(s) > 5:
            return s[:200]
    return llm_output[-200:]


async def report_node(state: CollaborationState) -> dict:
    """ReportAgent 节点函数

    两步流程：
    ① LLM 评估现有信息是否足够回答用户问题
    ② 足够 → 生成最终回答 + 引用来源；不够 → 提取追加检索方向
    """
    # 第一步：评估信息充分性
    check_prompt = (
        f"用户问题：{state['query']}\n\n"
        # 同上，全量传入（当前项目检索量远小于 DeepSeek 128K 窗口），无需截断
        f"检索结果：\n{state['search_results']}\n\n"
        f"初步分析：{state['analysis']}\n\n"
        f"请判断：现有信息是否足够回答用户问题？\n"
        f"注意：只判断能否回答用户【实际问的那个问题】。用户问数量/价格/名称等具体点，"
        f"检索结果里已给出对应答案就应判 sufficient=true，不要额外要求用户没问的细节"
        f"（例如版本具体名称、SLA、私有化范围等）。\n"
        f"- 如果足够，回复 sufficient=true\n"
        f"- 如果不够，回复 sufficient=false 并说明还缺什么信息（用'还需要查询：xxx'的格式）"
    )

    check_result = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": check_prompt}],
    ).choices[0].message.content

    # 第二步：根据评估结果做不同处理
    if "sufficient=true" in check_result.lower():
        # 信息足够 → 生成最终回答
        answer_prompt = (
            f"基于以下检索结果回答用户问题。请用中文直接回答，"
            f"回答中不要输出【来源: xxx】这类内部编号。\n\n"
            f"用户问题：{state['query']}\n\n"
            f"检索结果：\n{state['search_results']}"
        )
        answer = llm.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": answer_prompt}],
        ).choices[0].message.content

        return {
            "info_sufficient": True,
            "final_answer": answer,
            "sources": state.get("search_sources", []),
        }
    else:
        # 信息不够 → 提取追加检索方向
        followup = extract_followup(check_result)
        return {
            "info_sufficient": False,
            "followup_query": followup,
            "sources": state.get("search_sources", []),  # ★ 补：即使 insufficient 也回传检索来源
        }