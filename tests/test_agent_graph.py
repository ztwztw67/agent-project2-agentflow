# -*- coding: utf-8 -*-
"""端到端测试：验证 Agent Graph 完整链路

测试步骤（逐层递进，每步通过才进入下一步）：

Step 1 — 节点级测试：research_node 单独验证
Step 2 — 节点级测试：report_node 单独验证
Step 3 — 路由测试：route_tool_use 条件分支
Step 4 — 图级测试：完整图编译 + ainvoke

用法：
    cd F:/python-project/agent-agentflow
    .venv/Scripts/python tests/test_agent_graph.py
"""
import sys
import os

# 强制 UTF-8 输出，避免 Windows GBK 终端 emoji 报错
sys.stdout.reconfigure(encoding='utf-8')
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from agents.state import CollaborationState


SEPARATOR = "=" * 60


async def step1_test_research_node():
    """验证 research_node 能：检索策略 → MCP 调用 → 初步分析"""
    print(f"\n{SEPARATOR}")
    print("Step 1: 测试 research_node")
    print(SEPARATOR)

    from agents.research_agent import research_node

    state: CollaborationState = {"query": "云枢科技 CS-Pro 的 SLA 是什么？"}
    result = await research_node(state)

    assert "research_plan" in result, "缺少 research_plan"
    assert "search_results" in result, "缺少 search_results"
    assert "search_sources" in result, "缺少 search_sources"
    assert "analysis" in result, "缺少 analysis"

    print(f"  ✅ research_plan: {result['research_plan'][:100]}...")
    print(f"  ✅ search_results: {len(result['search_results'])} 字符")
    print(f"  ✅ search_sources: {result['search_sources']}")
    print(f"  ✅ analysis: {result['analysis'][:150]}...")

    return result


async def step2_test_report_node(research_output: dict):
    """验证 report_node 能：评估充分性 → 生成回答（或追加检索）"""
    print(f"\n{SEPARATOR}")
    print("Step 2: 测试 report_node")
    print(SEPARATOR)

    from agents.report_agent import report_node

    state: CollaborationState = {
        "query": "云枢科技 CS-Pro 的 SLA 是什么？",
        **research_output,
    }
    result = await report_node(state)

    if result.get("info_sufficient"):
        print(f"  ✅ info_sufficient: True")
        print(f"  ✅ final_answer: {result['final_answer'][:200]}...")
        print(f"  ✅ sources: {result.get('sources', [])}")
    else:
        print(f"  ⚠️  info_sufficient: False（信息不足，触发追加检索）")
        print(f"  ⚠️  followup_query: {result.get('followup_query', '')[:100]}")

    return result


def step3_test_router():
    """验证 route_tool_use 条件路由逻辑"""
    print(f"\n{SEPARATOR}")
    print("Step 3: 测试 route_tool_use 路由器")
    print(SEPARATOR)

    from agents.router import route_tool_use

    # 有结果 → 应走 report
    state_with_results: CollaborationState = {
        "query": "test",
        "search_results": "some results",
        "analysis": "some analysis",
    }
    assert route_tool_use(state_with_results) == "report", \
        f"有结果时应返回 'report'，实际返回 '{route_tool_use(state_with_results)}'"
    print("  ✅ 有 search_results + analysis → 路由到 'report'")

    # 无结果 → 应走 call_tools
    state_empty: CollaborationState = {"query": "test"}
    assert route_tool_use(state_empty) == "call_tools", \
        f"无结果时应返回 'call_tools'，实际返回 '{route_tool_use(state_empty)}'"
    print("  ✅ 无 search_results → 路由到 'call_tools'")


async def step4_test_full_graph():
    """验证完整图：编译 → ainvoke → 拿到 final_answer"""
    print(f"\n{SEPARATOR}")
    print("Step 4: 完整 Agent Graph 端到端测试")
    print(SEPARATOR)

    from agents.agent_graph import build_agent_graph
    from mcp_servers.mcp_client import get_mcp_client

    graph = await build_agent_graph()
    print(f"  ✅ 图编译成功，节点: {list(graph.nodes.keys())}")

    result = await graph.ainvoke({
        "query": "云枢科技 CS-Pro 的 SLA 是什么？"
    })

    print(f"\n  📋 最终状态中的字段: {list(result.keys())}")
    print(f"  📋 info_sufficient: {result.get('info_sufficient')}")
    print(f"  📋 sources: {result.get('sources', [])}")

    final_answer = result.get("final_answer", "")
    if final_answer:
        print(f"\n  🎯 最终回答（前 300 字）:\n{final_answer[:300]}")
    else:
        print(f"  ⚠️  未生成 final_answer（可能信息不足，followup_query={result.get('followup_query', '')[:100]}）")

    # 关闭 MCP 连接
    mcp_client = await get_mcp_client()
    await mcp_client.close()

    return result


# ===== 主入口 =====

async def main():
    print("🧪 AgentFlow Pro — 端到端验证")
    print(f"   项目根目录: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    try:
        # Step 1-2: 节点级测试（使用同一个 research 输出，避免重复调 MCP）
        research_output = await step1_test_research_node()
        await step2_test_report_node(research_output)

        # Step 3: 路由器测试（纯逻辑，不需要 MCP）
        step3_test_router()

        # Step 4: 完整图测试
        await step4_test_full_graph()

        print(f"\n{SEPARATOR}")
        print("✅ 全部 4 步验证通过！")
        print(SEPARATOR)

    except Exception as e:
        print(f"\n❌ 验证失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


asyncio.run(main())
