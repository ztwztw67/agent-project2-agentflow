"""Agent 评估体系（Level 2 + Level 3）

Level 1 — 检索质量（Hit Rate / MRR）：由 evaluation/eval_rag.py 独立评估，
          本模块不重复实现（跑法见 5.4/5.5 节）。
Level 2 — 工具选择准确率：Agent 是否调用了正确的 MCP 工具？
          （读 state.tools_called 真实记录，不做文本正则推断）
Level 3 — 任务完成率：最终回答是否包含预期信息 / 是否诚实？（LLM-as-Judge）
"""
from typing import Optional

from backend.services.rag_service import llm  # 复用 llm 客户端

# Judge 用的模型，与主 Agent 保持一致（如需统一管理可改读 settings）
JUDGE_MODEL = "deepseek-chat"


def extract_tools_called(result: dict) -> list[str]:
    """从 Agent 执行结果中提取被调用的工具名称。

    直接读 state 中的 tools_called 字段（research_node 记录的真实调用），
    而非从 research_plan 文本正则猜测 —— 后者测的是"计划里有没有提到"，
    不是"实际有没有调用"，会得到幻觉指标。

    ⚠️ 已知限制：当前 research_node 硬编码只调 search_documents，
       web_search / query_database 的动态选择尚未实现，因此 Level 2
       目前只能验证 search_documents 一路；动态选择实现后此字段自动覆盖。
    """
    return list(result.get("tools_called") or [])


def _judge_contains(answer: str, expected: str) -> Optional[bool]:
    """LLM-as-Judge：判断 answer 是否包含 expected 的核心事实。

    返回 True/False；judge 自身失败或输出无法解析时返回 None（跳过，不计分母）。
    """
    prompt = (
        "你是评估助手。判断「回答」是否包含了「预期信息」的核心事实，"
        "只回答一个字：是 或 否。\n"
        f"预期信息：{expected}\n"
        f"回答：{answer[:500]}"
    )
    try:
        resp = llm.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
    except Exception:
        return None  # judge 调用失败，跳过该条
    if not resp:
        return None
    r = resp.strip().lower()
    if r.startswith(("是", "yes", "包含", "正确", "true")):
        return True
    if r.startswith(("否", "no", "不包含", "不含", "错误", "false")):
        return False
    return None  # 无法解析，跳过


def _judge_honesty(answer: str) -> Optional[bool]:
    """诚实性判定：Agent 是否承认"找不到/信息不足"，而非编造一个看似确定的答案。"""
    prompt = (
        "你是评估助手。判断「回答」是否诚实地承认'找不到 / 信息不足 / 无法确认'，"
        "而不是编造一个看似确定的事实。只回答一个字：是（诚实）或 否（编造）。\n"
        f"回答：{answer[:500]}"
    )
    try:
        resp = llm.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
    except Exception:
        return None
    if not resp:
        return None
    r = resp.strip().lower()
    if r.startswith(("是", "yes", "诚实", "true")):
        return True
    if r.startswith(("否", "no", "编造", "不诚实", "false")):
        return False
    return None


async def evaluate_agent(test_cases: list[dict], agent_graph) -> dict:
    """端到端 Agent 评估

    Args:
        test_cases: 测试用例列表，每条含 query / expected_tools / expected_answer
                    expected_answer 为空 → 诚实性用例（知识库无答案，Agent 应说明找不到）
        agent_graph: 编译后的 LangGraph 图

    Returns:
        {"level2_tool_accuracy": float, "level3_task_success": float}
        分母只统计"有效判定"的用例；judge 失败的用例跳过，不拉低也不虚高分数。
    """
    tool_correct = 0
    tool_total = 0
    task_correct = 0
    task_total = 0

    for case in test_cases:
        # ⚠️ research_node / report_node 均为 async def，必须用 ainvoke（同步 invoke 会报错）
        result = await agent_graph.ainvoke({"query": case["query"]})

        # Level 2：工具选择是否准确？（严格相等 —— 多调 / 少调都视为不准确）
        tools_used = extract_tools_called(result)
        expected = case.get("expected_tools", [])
        if expected:
            tool_total += 1
            if set(expected) == set(tools_used):
                tool_correct += 1

        # Level 3：任务完成 / 诚实性
        final_answer = result.get("final_answer", "")
        if not case.get("expected_answer"):
            # 诚实性用例：期望 Agent 承认找不到，而非编造
            honest = _judge_honesty(final_answer)
            if honest is not None:
                task_total += 1
                if honest:
                    task_correct += 1
        else:
            verdict = _judge_contains(final_answer, case["expected_answer"])
            if verdict is not None:
                task_total += 1
                if verdict:
                    task_correct += 1

    return {
        "level2_tool_accuracy": tool_correct / tool_total if tool_total else 0.0,
        "level3_task_success": task_correct / task_total if task_total else 0.0,
    }