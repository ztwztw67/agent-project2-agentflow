"""Agent 评估测试用例（26 条）

每条包含：
- query: 用户问题
- expected_tools: 期望 Agent 调用的 MCP 工具（当前 research_node 只调 search_documents）
- expected_answer: 期望回答中包含的关键信息（用于 LLM-as-Judge）
                   为空 → 诚实性用例（知识库无答案，Agent 应说明找不到）

⚠️ 用例基于 3 份测试 PDF（均已生成）：
   - test_doc.pdf  （v3.0 说明书，基础）
   - test_doc2.pdf （v3.1 更新 + 迁移手册）
   - test_doc3.pdf （第三方竞品白皮书）
   跑评估前先把三份 PDF 通过 /rag/upload 上传进知识库。expected_answer 已与
   这三份文档逐条对齐，无需再校准。跨文档/流程类用例见第 8、9 组。
"""
agent_test_cases = [
    # === 1. 精准事实直查（3 条） ===
    {"query": "云枢科技的 CTO 是谁？",
     "expected_tools": ["search_documents"], "expected_answer": "林远舟"},
    {"query": "CS-Pro 一年多少钱？",
     "expected_tools": ["search_documents"], "expected_answer": "128,000"},
    {"query": "星言 Pro 一年多少钱？",
     "expected_tools": ["search_documents"], "expected_answer": "96,000"},

    # === 2. 语义替换（2 条，验证 embedding 对同义改写的鲁棒性） ===
    {"query": "云枢科技的首席技术官是哪位？",
     "expected_tools": ["search_documents"], "expected_answer": "林远舟"},
    {"query": "CS-Pro 每年要花多少钱？",
     "expected_tools": ["search_documents"], "expected_answer": "128,000"},

    # === 3. 表格 / 易混实体（2 条，验证精确检索而非模糊匹配） ===
    {"query": "CS-Lite 和 CS-Pro 的存储容量有什么区别？",
     "expected_tools": ["search_documents"], "expected_answer": "100GB 和 1TB"},
    {"query": "CS-Enterprise 一年多少钱？",
     "expected_tools": ["search_documents"], "expected_answer": "面议"},

    # === 4. 否定 / 排除（3 条，验证不被语义相似性误导） ===
    {"query": "SLA 补偿可以退现金吗？",
     "expected_tools": ["search_documents"], "expected_answer": "不可以"},
    {"query": "CS-Pro 支持私有化部署吗？",
     "expected_tools": ["search_documents"], "expected_answer": "不支持"},
    {"query": "迁移窗口内的停机算不算 SLA 违约？",
     "expected_tools": ["search_documents"], "expected_answer": "不算"},

    # === 5. 时间 / 版本限定（3 条，含跨文档版本冲突：取最新，别答旧值） ===
    {"query": "最新版 CS-Pro 的 SLA 可用性是多少？",
     "expected_tools": ["search_documents"], "expected_answer": "99.98%"},
    {"query": "v3.0 时期 CS-Pro 的 SLA 是多少？",
     "expected_tools": ["search_documents"], "expected_answer": "99.95%"},
    {"query": "单文档上传上限现在是多少？",
     "expected_tools": ["search_documents"], "expected_answer": "50MB"},

    # === 6. 长尾 / 冷门细节（2 条，验证低频信息召回） ===
    {"query": "云枢科技成立于哪一年？",
     "expected_tools": ["search_documents"], "expected_answer": "2019"},
    {"query": "星阑科技成立于哪一年？",
     "expected_tools": ["search_documents"], "expected_answer": "2020"},

    # === 7. 多跳 / 综合推理（2 条，信息需跨 chunk 拼接） ===
    {"query": "CS-Pro 一年的费用是含税价吗？",
     "expected_tools": ["search_documents"], "expected_answer": "含税"},
    {"query": "智言平台中存储容量比 CS-Pro 更大的版本有哪些？",
     "expected_tools": ["search_documents"], "expected_answer": "CS-Max、CS-Enterprise"},

    # === 8. 多步骤 / 流程性操作（2 条，来自 test_doc2 迁移手册） ===
    {"query": "把知识库从 v3.0 迁移到 v3.1 需要哪几步？",
     "expected_tools": ["search_documents"], "expected_answer": "备份、升级、校验、切换"},
    {"query": "迁移升级失败怎么处理？",
     "expected_tools": ["search_documents"], "expected_answer": "回滚"},

    # === 9. 跨文档 / 全局聚合（4 条，必须检索多份 PDF 才能答对） ===
    {"query": "云枢科技一共有几个产品版本？",
     "expected_tools": ["search_documents"], "expected_answer": "四个"},
    {"query": "智言平台的竞品有哪些？",
     "expected_tools": ["search_documents"], "expected_answer": "星言、聆语"},
    {"query": "智言 CS-Pro 和星言 Pro 哪个更便宜？",
     "expected_tools": ["search_documents"], "expected_answer": "星言"},
    {"query": "智言 CS-Pro 和聆语企业版哪个 SLA 更高？",
     "expected_tools": ["search_documents"], "expected_answer": "聆语"},

    # === 10. 诚实性测试（3 条，知识库无答案 → Agent 应说明找不到，而非编造） ===
    {"query": "云枢科技的 CEO 是谁？",
     "expected_tools": ["search_documents"], "expected_answer": ""},
    {"query": "云枢科技的创始人是谁？",
     "expected_tools": ["search_documents"], "expected_answer": ""},
    {"query": "智言平台怎么申请退款？",
     "expected_tools": ["search_documents"], "expected_answer": ""},
]