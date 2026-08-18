# AgentFlow Pro

> 企业知识库问答系统 —— RAG 检索 × 多 Agent 协作 × MCP 协议 × 全链路可观测

一个从零构建的 Agent 应用，以**企业知识库问答**为场景，打通「文档上传 → 向量化 → 混合检索 + 重排 → 多 Agent 协作回答 → 三级评估 → LangFuse 追踪」的完整链路。

## ✨ 核心特性

- **RAG 检索链路演进 v1 → v3**
  - v1 纯向量检索（BGE 嵌入 + ChromaDB）
  - v2 多工具 Agent（检索 + Web 搜索 + 数据库查询）
  - v3 混合检索 + 重排：`Query 重写 → Dense+BM25 融合 → Cross-Encoder Rerank → Top-3`
- **多 Agent 协作**（LangGraph 编排）：ResearchAgent 负责检索与初步分析，ReportAgent 负责信息充分性判断与回答生成
- **MCP 协议工具调用**：`search_documents` / `web_search` / `query_database` 三个 MCP Server
- **三级评估体系**：Level 1 检索质量（Hit Rate / MRR）、Level 2 工具选择准确率、Level 3 任务完成率 + 诚实性
- **全链路可观测**：LangFuse tracing，每次对话的检索 / 工具调用 / 生成全链路可追溯
- **FastAPI 服务化 + Docker 部署**

## 🏗️ 系统架构

```
用户 Query
   │
   ▼
FastAPI /api/chat
   │
   ▼
Agent Graph（LangGraph）
   ├── ResearchAgent ──制定检索策略──► MCP Client
   │                                      ├── search_documents ──► RAG v3 链路
   │                                      │      Query 重写 → Dense+BM25 混合 → Rerank → Top-3
   │                                      ├── web_search
   │                                      └── query_database
   └── ReportAgent ──信息充分性判断 + 生成回答 + 引用来源
   │
   ▼
final_answer + sources
   │
   ▼
LangFuse 全链路追踪
```

## 🛠️ 技术栈

| 分类 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM 框架 | LangChain 0.3 + LangGraph 0.2 |
| 向量数据库 | ChromaDB 0.6 |
| Embedding / Rerank | BGE（bge-small-zh-v1.5 / bge-reranker-base） |
| 混合检索 | BM25（rank-bm25）+ 语义检索 |
| MCP 协议 | mcp 1.x |
| LLM 调用 | OpenAI SDK（DeepSeek，可换 OpenAI） |
| 可观测性 | LangFuse 3 |
| 数据库 / 缓存 | MySQL + Redis（SQLAlchemy 2.0） |
| 认证 | JWT（python-jose + passlib） |

## 📁 目录结构

```
agent-agentflow/
├── backend/
│   ├── main.py              # FastAPI 入口（lifespan 初始化 MCP + Agent Graph）
│   ├── config.py            # pydantic-settings 配置（读取 .env）
│   ├── db.py                # 数据库连接
│   ├── middleware/log.py    # 请求日志中间件
│   ├── models/              # Pydantic 数据模型
│   ├── routers/             # auth / chat / rag 路由
│   └── services/
│       ├── rag_service.py   # RAG 核心（v1 向量检索 / v3 混合+重排）
│       └── agent_tools.py   # v2 多工具 Agent
├── agents/                  # LangGraph 多 Agent
│   ├── state.py             # CollaborationState
│   ├── research_agent.py    # ResearchAgent（检索策略 + MCP 调用）
│   ├── report_agent.py      # ReportAgent（充分性判断 + 生成）
│   ├── router.py            # 条件路由
│   └── agent_graph.py       # build_agent_graph()
├── mcp_servers/             # MCP Server（search_docs / web_search / query_db）
├── evaluation/              # 三级评估体系
│   ├── eval_rag.py          # Level 1：Hit Rate / MRR
│   ├── agent_eval.py        # Level 2/3：工具选择 + 任务完成
│   └── test_cases.py        # 26 条测试用例
├── observability/           # LangFuse 追踪
├── tests/                   # 单元测试
├── Dockerfile / docker-compose.yml
├── requirement.txt
└── text-agent.py            # LangChain ReAct Agent 学习示例（入门脚本）
```

## 🚀 快速开始

### 1. 环境准备

需要 Python 3.11。创建虚拟环境并安装依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirement.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`（字段定义见 `backend/config.py`），核心必填项：

```ini
OPENAI_API_KEY=sk-xxx        # DeepSeek / OpenAI 的 Key
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
TAVILY_API_KEY=tvly-xxx      # v2 Web 搜索用
LANGFUSE_SECRET_KEY=...      # 可观测性（可选）
LANGFUSE_PUBLIC_KEY=...
```

> ⚠️ `.env` 已被 `.gitignore` 忽略，**切勿**提交包含真实 Key 的 `.env`。

### 3. 准备 Embedding / Rerank 模型

`rag_service.py` 使用本地 BGE 模型，需先下载并把路径配置为你的本地路径：

```bash
# bge-small-zh-v1.5（向量化）
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir models/bge-small-zh-v1.5
# bge-reranker-base（重排）
modelscope download --model BAAI/bge-reranker-base --local_dir models/bge-reranker-base
```

### 4. 启动服务

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 5. 上传文档 + 对话

```bash
# 上传知识库文档（txt / pdf）
curl -X POST http://localhost:8000/rag/upload \
  -F "file=@test_doc.pdf"

# Agent 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "云枢科技 CS-Pro 的 SLA 是多少？"}'
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务状态 |
| GET | `/health` | 健康检查 |
| POST | `/auth/*` | 认证（注册 / 登录） |
| POST | `/api/chat` | Agent 对话 |
| POST | `/rag/upload` | 上传文档 → 切分 → 向量化 |

## 📊 三级评估体系

| 级别 | 指标 | 评估内容 | 位置 |
|------|------|----------|------|
| Level 1 | Hit Rate / MRR | 检索质量（正确 chunk 是否被召回且靠前） | `evaluation/eval_rag.py` |
| Level 2 | 工具选择准确率 | Agent 是否调用了正确的 MCP 工具 | `evaluation/agent_eval.py` |
| Level 3 | 任务完成率 + 诚实性 | 回答是否包含预期信息 / 是否诚实承认不知道 | `evaluation/agent_eval.py` |

```bash
# Level 1：检索质量（从项目根目录运行，否则 ./chroma_db 相对路径会读空库）
.venv\Scripts\python -m evaluation.eval_rag
```

Level 2 / 3 通过 `evaluate_agent()` 异步函数调用，需要一个入口脚本：

```python
# evaluation/run_agent_eval.py
import asyncio
from evaluation.agent_eval import evaluate_agent
from evaluation.test_cases import agent_test_cases
from agents.agent_graph import build_agent_graph

async def main():
    graph = await build_agent_graph()
    scores = await evaluate_agent(agent_test_cases, graph)
    print(f"工具选择准确率: {scores['level2_tool_accuracy']:.1%}")
    print(f"任务完成率: {scores['level3_task_success']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📈 可观测性

基于 LangFuse，完整追踪每次对话链路：Query 重写 → 检索 → MCP 工具调用 → 生成，可在 LangFuse Dashboard 查看每次对话的全链路得分。

## 🧪 Dify 低代码对照实验

用 Dify 搭建了同场景的 PoC 版本，从开发速度 / 定制化能力 / 性能 / 可观测性四个维度对比，验证「核心业务需自研、快速验证可用低代码」的技术选型判断。
