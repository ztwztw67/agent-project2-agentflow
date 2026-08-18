"""RAG 上传路由 —— AgentFlow Pro 独立知识库的数据入口

与 DeepRAG 不同：本项目问答走 MCP 链路（/api/chat → agent_graph → MCP search_documents），
因此这里只提供 /upload，不再提供 /chat（避免与 chat.py 的 agent_graph 重复）。
"""
import os
import tempfile

from fastapi import APIRouter, UploadFile, File
from backend.models.response import APIResponse
from backend.services.rag_service import upload_document, EmbeddingError

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传文档（txt / pdf）→ 切分 → 向量化 → 存入本项目独立 chroma_db"""
    content = await file.read()
    original_filename = file.filename or "upload"
    suffix = os.path.splitext(original_filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunk_count = upload_document(tmp_path)
        return APIResponse(
            message=f"文档 {original_filename} 已处理完成",
            data={"filename": original_filename, "chunks": chunk_count},
        )
    except EmbeddingError as e:
        return APIResponse(code=500, message=f"文档处理失败: {str(e)}")
    finally:
        os.unlink(tmp_path)  # 清理临时文件