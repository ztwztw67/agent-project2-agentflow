"""LangFuse 集成验证 —— 确认全链路追踪正常工作

验证步骤：
  Step 1 — 配置加载检查：.env 密钥 → config.py → CallbackHandler
  Step 2 — 连接验证：LangFuse client 能否连通云端
  Step 3 — Trace 发送：发送一次 Agent 对话，确认 trace 出现在 Dashboard
  Step 4 — Dashboard 解读：看懂调用链中的每个 Span

用法：
    cd F:/python-project/agent-agentflow
    .venv/Scripts/python tests/test_langfuse.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

SEP = "=" * 60


def step1_check_config():
    """Step 1 — 配置加载：确认 .env 密钥被正确读入"""
    print(f"\n{SEP}")
    print("Step 1: 配置加载检查")
    print(SEP)

    from backend.config import settings

    pk = settings.langfuse_public_key
    sk = settings.langfuse_secret_key
    host = settings.langfuse_host

    assert pk and pk.startswith("pk-lf-"), \
        f"❌ langfuse_public_key 未配置或格式异常: {pk}"
    assert sk and sk.startswith("sk-lf-"), \
        f"❌ langfuse_secret_key 未配置或格式异常: {sk}"
    assert host, "❌ langfuse_host 为空"

    print(f"  ✅ langfuse_public_key: {pk[:12]}...（已脱敏）")
    print(f"  ✅ langfuse_secret_key: {sk[:12]}...（已脱敏）")
    print(f"  ✅ langfuse_host: {host}")


def step2_check_callback():
    """Step 2 — CallbackHandler 初始化：确认可正常创建"""
    print(f"\n{SEP}")
    print("Step 2: CallbackHandler 初始化检查")
    print(SEP)

    from observability.langfuse_callback import langfuse_handler

    assert langfuse_handler is not None, "❌ langfuse_handler 为 None"
    print(f"  ✅ CallbackHandler 初始化成功")
    print(f"  ✅ 类型: {type(langfuse_handler).__name__}")


def step3_check_tracer():
    """Step 3 — traced_chat 可用性：确认 tracer 封装可导入"""
    print(f"\n{SEP}")
    print("Step 3: traced_chat 导入检查")
    print(SEP)

    from observability.langfuse_tracer import traced_chat, langfuse

    assert traced_chat is not None
    assert langfuse is not None
    print(f"  ✅ traced_chat 导入成功")
    print(f"  ✅ langfuse client 已初始化")
    print(f"  ⚠️  Trace 发送需在 Agent 对话请求后验证 —— 见 Step 4")


async def step4_send_trace():
    """Step 4 — 发送 Trace：通过一次 Agent 对话验证端到端追踪"""
    print(f"\n{SEP}")
    print("Step 4: 端到端 Trace 发送验证")
    print(SEP)

    from agents.agent_graph import build_agent_graph
    from mcp_servers.mcp_client import get_mcp_client
    from observability.langfuse_tracer import traced_chat

    # 初始化 graph（与 main.py lifespan 中的逻辑一致）
    mcp_client = await get_mcp_client()
    tools = mcp_client.get_tool_meta()
    print(f"  MCP: {len(tools)} 个工具已连接 ({[t['name'] for t in tools]})")

    graph = await build_agent_graph()
    print(f"  Agent Graph: 已编译 (nodes={list(graph.nodes.keys())})")

    # ★ 发送带 LangFuse 追踪的对话请求
    import uuid
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    query = "云枢科技 CS-Pro 的 SLA 是什么？"

    print(f"\n  📤 发送请求: query='{query}', session_id='{session_id}'")
    result = await traced_chat(graph, query, session_id)

    print(f"  📥 info_sufficient: {result.get('info_sufficient')}")
    print(f"  📥 sources: {result.get('sources', [])}")
    final_answer = result.get('final_answer', '')
    if final_answer:
        print(f"  📥 final_answer: {final_answer[:150]}...")

    print(f"\n  🎯 关键验证 —— 打开浏览器访问:")
    print(f"     https://cloud.langfuse.com")
    print(f"     登录后应在 Traces 页面看到一条名为 'agent-chat' 的 trace")
    print(f"     session_id: {session_id}")
    print(f"     Tags: agent-agentflow-pro")

    # 清理
    await mcp_client.close()

    print(f"\n  💡 如果 Dashboard 中看不到 trace：")
    print(f"     1. 等待 10-30 秒（LangFuse 有异步上报延迟）")
    print(f"     2. 刷新页面（F5）")
    print(f"     3. 确认 .env 中 LANGFUSE_HOST 是 https://cloud.langfuse.com（不是自建地址）")
    print(f"     4. 在 cloud.langfuse.com → Settings → API Keys 中确认密钥未过期")


# ===== 主入口 =====

async def main():
    print("🧪 AgentFlow Pro — LangFuse 集成验证")
    print(f"   项目根目录: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    try:
        step1_check_config()
        step2_check_callback()
        step3_check_tracer()
        await step4_send_trace()

        print(f"\n{SEP}")
        print("✅ LangFuse 集成验证完成！")
        print(SEP)
        print()
        print("📋 下一步操作：")
        print("   1. 打开 https://cloud.langfuse.com")
        print("   2. 左侧菜单 → Traces")
        print("   3. 找到名为 'agent-chat' 的 trace（tag=agent-agentflow-pro）")
        print("   4. 点开 trace，确认能看到完整调用链：")
        print("      agent-chat (trace)")
        print("      └── ChatOpenAI / DeepSeek (LLM 调用)")
        print("          ├── 输入 token 数")
        print("          ├── 输出 token 数")
        print("          ├── latency（延迟）")
        print("          └── 实际 prompt 内容（可展开查看）")

    except Exception as e:
        print(f"\n❌ 验证失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


import asyncio
asyncio.run(main())