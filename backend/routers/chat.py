"""Agent 对话路由 —— POST /api/chat

接收用户问题 → 传入 Agent Graph（ResearchAgent → MCP tools → ReportAgent）→ 返回最终回答。
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field


router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(description="用户问题", min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    query: str
    final_answer: str | None = None
    sources: list[str] = []
    info_sufficient: bool | None = None
    followup_query: str | None = None
    error: str | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Agent 对话接口

    完整链路：query → ResearchAgent（制定检索策略 + 调用 MCP + 初步分析）
                      → ReportAgent（评估信息充分性 + 生成回答 + 引用来源）

    示例请求：
        POST /api/chat
        {"query": "云枢科技 CS-Pro 的 SLA 是多少？"}
    """
    graph = req.app.state.agent_graph

    try:
        # invoke 传入的 dict 匹配 CollaborationState 的字段
        result = await graph.ainvoke({"query": request.query})

        return ChatResponse(
            query=request.query,
            final_answer=result.get("final_answer"),
            sources=result.get("sources", []),
            info_sufficient=result.get("info_sufficient"),
            followup_query=result.get("followup_query"),
        )

    except Exception as e:
        return ChatResponse(
            query=request.query,
            error=f"{type(e).__name__}: {str(e)}",
        )
