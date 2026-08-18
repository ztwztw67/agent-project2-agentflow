"""LangFuse CallbackHandler —— 自动追踪 LangChain/LangGraph 内部调用

LangFuse 提供 LangChain 集成：只需在 graph.invoke() 时传入此 callback，
所有 LLM 调用、Tool 调用、Chain 步骤都会被自动追踪，无需手动在每个函数上加 @observe。
"""
from langfuse.langchain import CallbackHandler

# SDK v3: CallbackHandler() 无参构造，自动读取环境变量
langfuse_handler = CallbackHandler()