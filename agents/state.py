"""Agent 协作的全局状态 —— 所有节点通过此 State 交换数据"""
from typing import TypedDict, List, Optional

class CollaborationState(TypedDict):
    query: str                          # 用户原始问题
    research_plan: Optional[str]        # ResearchAgent: 检索策略（LLM 生成）
    search_results: Optional[str]       # ResearchAgent: 检索结果（格式化文本）
    search_sources: Optional[List[str]] # ResearchAgent: 引用来源 chunk_id 列表
    tools_called: Optional[List[str]]  # ResearchAgent: 实际调用的 MCP 工具名（Level 2 评估依据）
    analysis: Optional[str]             # ResearchAgent: 初步分析
    info_sufficient: Optional[bool]     # ReportAgent: 信息是否足够
    followup_query: Optional[str]       # ReportAgent: 如不够，追加的检索方向
    final_answer: Optional[str]         # ReportAgent: 最终回答
    sources: Optional[List[str]]        # ReportAgent: 最终引用的来源