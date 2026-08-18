import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ⚠️ 先加载 .env，再 import rag_service。
# rag_service 模块顶层会初始化 llm = OpenAI(api_key=...)，未加载 .env 时报 Missing credentials。
# 评估只测检索（无需 LLM），但模块顶层的 llm 初始化绕不过去——生产改进：把 llm 改为懒加载。
from dotenv import load_dotenv
load_dotenv()

from backend.services.rag_service import search, search_v3


# ====== Chunk ID 发现工具 ======
def show_chunks():
    """打印知识库中所有 chunk 的 ID 和内容预览。

    上传 test_doc.pdf 后运行一次，根据输出把 chunk_id 填入下方 test_queries 的
    relevant_chunk_ids 字段，替换 3b9cd67e 占位符。
    运行方式：python -c "from backend.evaluation.eval_rag import show_chunks; show_chunks()"
    """
    from backend.services.rag_service import vectorstore
    data = vectorstore.get()
    for i, (chunk_id, content) in enumerate(zip(data["ids"], data["documents"])):
        print(f"{chunk_id}  ←  {content[:80].replace(chr(10), ' ')}")


# ====== 测试集 ======
# 构造方法（参考，面试时讲这个流程比讲"我有 30 条数据"更有价值）：
#   ① 用户历史 query：从 /rag/chat 的实际日志中提取真实问题（最贴近生产）
#   ② 文档反向生成：把文档分段，让 LLM 为每段生成一个"用户可能问的问题"
#      （例：SLA 段落 → "CS-Pro 的可用性保证是多少？"）
#   ③ LLM 合成 + 人工校验：用 GPT-4 批量生成 50 条 → 人工筛掉不合理/重复的 → 保留 20-30 条
#
# 标注方法（两套方案，主方案优先）：
#   主方案 — chunk ID：上传后运行 show_chunks() 查出 chunk_id → 人工确认哪个 chunk
#            包含正确答案 → 填入 relevant_chunk_ids。评估时精确比对 chunk ID——
#            不受 PDF 换行截断、同义改写、关键词遗漏等干扰。
#   回退方案 — 关键词：当 chunk ID 未标注时（relevant_chunk_ids 为空），自动降级为
#              relevant_keywords 子串匹配（含空白归一）。

test_queries = [
    # === 1. 精准事实直查（3 条，test_doc.pdf） ===
    {"query": "云枢科技的 CTO 是谁？",
     "relevant_chunk_ids": ["3b9cd67e-0"],           # 概述段（含林远舟）
     "relevant_keywords": ["林远舟"]},
    {"query": "CS-Pro 一年多少钱？",
     "relevant_chunk_ids": ["3b9cd67e-1"],           # 价格表段
     "relevant_keywords": ["128,000"]},
    {"query": "星言 Pro 一年多少钱？",
     "relevant_chunk_ids": ["5874b2a9-1"],           # test_doc3 竞品对比表（¥96,000）
     "relevant_keywords": ["96,000"]},

    # === 2. 语义替换（2 条，同义改写应命中同段） ===
    {"query": "云枢科技的首席技术官是哪位？",
     "relevant_chunk_ids": ["3b9cd67e-0"],
     "relevant_keywords": ["林远舟"]},
    {"query": "CS-Pro 每年要花多少钱？",
     "relevant_chunk_ids": ["3b9cd67e-1"],
     "relevant_keywords": ["128,000"]},

    # === 3. 表格 / 易混实体（2 条，验证精确检索而非模糊匹配） ===
    {"query": "CS-Lite 和 CS-Pro 的存储容量有什么区别？",
     "relevant_chunk_ids": ["3b9cd67e-1"],           # 版本规格表（100GB / 1TB）
     "relevant_keywords": ["100GB", "1TB"]},
    {"query": "CS-Enterprise 一年多少钱？",
     "relevant_chunk_ids": ["fc1b576c-2"],           # test_doc2 CS-Enterprise 段（面议）
     "relevant_keywords": ["面议"]},

    # === 4. 否定 / 排除（3 条，不被语义相似误导） ===
    {"query": "SLA 补偿可以退现金吗？",
     "relevant_chunk_ids": ["3b9cd67e-2"],           # SLA 正文（不以现金）
     "relevant_keywords": ["不以现金"]},
    {"query": "CS-Pro 支持私有化部署吗？",
     "relevant_chunk_ids": ["3b9cd67e-4"],           # FAQ Q1（本段）：CS-Pro 核心服务运行在公有云（不支持私有化）
     "relevant_keywords": ["公有云"]},
    {"query": "迁移窗口内的停机算不算 SLA 违约？",
     "relevant_chunk_ids": ["fc1b576c-3"],           # 迁移窗口例外条款（每月第二个周日，不计入 SLA）
     "relevant_keywords": ["不计入"]},

    # === 5. 时间 / 版本限定（3 条，跨文档版本冲突：取最新别答旧值） ===
    {"query": "最新版 CS-Pro 的 SLA 可用性是多少？",
     "relevant_chunk_ids": ["fc1b576c-1"],           # v3.1 SLA 表（99.98%）
     "relevant_keywords": ["99.98%"]},
    {"query": "v3.0 时期 CS-Pro 的 SLA 是多少？",
     "relevant_chunk_ids": ["3b9cd67e-2"],           # test_doc SLA 正文（99.95%；fc1b576c-1 对比表也有此值）
     "relevant_keywords": ["99.95%"]},
    {"query": "单文档上传上限现在是多少？",
     "relevant_chunk_ids": ["3b9cd67e-4"],           # 上传限制段（50MB）
     "relevant_keywords": ["50MB"]},

    # === 6. 长尾 / 冷门细节（2 条，验证低频信息召回） ===
    {"query": "云枢科技成立于哪一年？",
     "relevant_chunk_ids": ["3b9cd67e-0"],           # 概述段（2019年3月）
     "relevant_keywords": ["2019"]},
    {"query": "星阑科技成立于哪一年？",
     "relevant_chunk_ids": ["5874b2a9-1"],           # 厂商详述 3.1（星阑科技成立于 2020 年）
     "relevant_keywords": ["2020"]},

    # === 7. 多跳 / 综合推理（2 条，信息需跨 chunk 拼接） ===
    {"query": "CS-Pro 一年的费用是含税价吗？",
     "relevant_chunk_ids": ["3b9cd67e-1"],           # 报价表（含税，税率 6%）
     "relevant_keywords": ["含税"]},
    {"query": "智言平台中存储容量比 CS-Pro 更大的版本有哪些？",
     "relevant_chunk_ids": ["3b9cd67e-1", "fc1b576c-2"],  # CS-Max ≥10TB（报价表）+ CS-Enterprise ≥100TB（v3.1）
     "relevant_keywords": ["CS-Max", "CS-Enterprise"]},

    # === 8. 多步骤 / 流程（2 条，test_doc2.pdf 迁移手册）★ G/H/I 补进 ===
    {"query": "把知识库从 v3.0 迁移到 v3.1 需要哪几步？",
     "relevant_chunk_ids": ["fc1b576c-2", "fc1b576c-3"],  # ①②备份/升级在 fc1b576c-2，③④校验/切换在 fc1b576c-3
     "relevant_keywords": ["备份", "升级", "校验", "切换"]},
    {"query": "迁移升级失败怎么处理？",
     "relevant_chunk_ids": ["fc1b576c-3"],           # 迁移失败处理段（回滚）
     "relevant_keywords": ["回滚"]},

    # === 9. 跨文档 / 全局聚合（4 条，必须检索多份 PDF 才答对）★ G/H/I 补进 ===
    {"query": "云枢科技一共有几个产品版本？",
     "relevant_chunk_ids": ["fc1b576c-2", "fc1b576c-3"],  # v3.1 第四档 CS-Enterprise + 废止条款「现为四个版本」
     "relevant_keywords": ["四个"]},
    {"query": "智言平台的竞品有哪些？",
     "relevant_chunk_ids": ["5874b2a9-0", "5874b2a9-1"],  # 市场概览 + 厂商详述（星言/聆语）
     "relevant_keywords": ["星言", "聆语"]},
    {"query": "智言 CS-Pro 和星言 Pro 哪个更便宜？",
     "relevant_chunk_ids": ["3b9cd67e-1", "5874b2a9-1"],  # 跨 test_doc/test_doc3 比价
     "relevant_keywords": ["星言"]},
    {"query": "智言 CS-Pro 和聆语企业版哪个 SLA 更高？",
     "relevant_chunk_ids": ["fc1b576c-1", "5874b2a9-1"],  # 跨 test_doc2/test_doc3 比 SLA
     "relevant_keywords": ["聆语"]},

    # === 10. 诚实性测试（3 条，知识库无答案 → 检索应无匹配） ===
    {"query": "云枢科技的 CEO 是谁？",
     "relevant_chunk_ids": [],
     "relevant_keywords": []},
    {"query": "云枢科技的创始人是谁？",
     "relevant_chunk_ids": [],
     "relevant_keywords": []},
    {"query": "智言平台怎么申请退款？",
     "relevant_chunk_ids": [],
     "relevant_keywords": []},
]
# ★ 26 条，与 test_cases.py（Level 2/3）对齐：10 组、23 条有效 + 3 条诚实性。
#   chunk ID 已按 show_chunks() 完整输出逐条核实：迁移步骤①②在 fc1b576c-2、③④在
#   fc1b576c-3；存储对比/版本聚合/跨文档比价等多段答案均标注了多个 relevant_chunk_ids。
#   keyword 仅作 chunk ID 未命中时的回退兜底，跨文档时会撞车，别依赖它。


# ====== 评估函数 ======
def _norm(s: str) -> str:
    """空白归一：PDF 换行把 '50\nMB' 拆成两行，归一后才能匹配关键词 '50MB'。"""
    return re.sub(r"\s+", "", s)


def _get_retrieved_ids(results) -> set[str]:
    """从检索结果中提取 chunk_id 集合。"""
    ids = set()
    for r in results:
        if isinstance(r, tuple):
            cid = (r[1] or {}).get("chunk_id", "")
        elif hasattr(r, "metadata"):
            cid = (r.metadata or {}).get("chunk_id", "")
        else:
            cid = ""
        if cid:
            ids.add(cid)
    return ids


def _get_retrieved_texts(results) -> list[str]:
    """从检索结果中提取文本列表。"""
    return [r[0] if isinstance(r, tuple) else r.page_content for r in results]


def evaluate_hit_rate(test_set, retriever_fn) -> float:
    """Hit Rate：正确事实出现在 Top-3 结果中的 query 占比。

    判定优先级：chunk ID 精确匹配 > 关键词子串匹配（含空白归一）。
    """
    hits = 0
    for item in test_set:
        results = retriever_fn(item["query"])
        target_ids = item.get("relevant_chunk_ids", [])
        keywords = item.get("relevant_keywords", [])

        # 主方案：chunk ID 精确匹配
        if target_ids:
            retrieved_ids = _get_retrieved_ids(results)
            if any(tid in retrieved_ids for tid in target_ids):
                hits += 1
                continue  # 命中，跳过回退方案
            # chunk ID 未命中——说明检索确实没召回正确 chunk
            texts = _get_retrieved_texts(results)
            print(f"  ✗ chunk ID 未命中: {item['query'][:40]}  |  "
                  f"target={target_ids}  retrieved={retrieved_ids}  "
                  f"top1 前60字: {_norm(texts[0])[:60] if texts else '-'}")
            continue

        # 回退方案：关键词子串匹配（chunk ID 未标注时启用）
        if not keywords:
            continue  # 诚实性测试，不计入分母
        texts = _get_retrieved_texts(results)
        combined = _norm(" ".join(texts))
        if any(_norm(kw) in combined for kw in keywords):
            hits += 1
    total = sum(1 for t in test_set
                if t.get("relevant_chunk_ids") or t.get("relevant_keywords"))
    return hits / total if total > 0 else 0.0


def evaluate_mrr(test_set, retriever_fn) -> float:
    """MRR（Mean Reciprocal Rank）：正确答案首次出现排名的倒数平均值。

    排名从 1 开始（排第 1 位 = 1/1 = 1.0；排第 10 位 = 1/10 = 0.1）。
    只取第一个匹配的结果的排名。判定优先级同 evaluate_hit_rate。
    """
    reciprocal_ranks = []
    for item in test_set:
        results = retriever_fn(item["query"])
        target_ids = item.get("relevant_chunk_ids", [])
        keywords = item.get("relevant_keywords", [])

        # 主方案：chunk ID
        if target_ids:
            for rank, r in enumerate(results, start=1):
                cid = ""
                if isinstance(r, tuple):
                    cid = (r[1] or {}).get("chunk_id", "")
                elif hasattr(r, "metadata"):
                    cid = (r.metadata or {}).get("chunk_id", "")
                if cid and cid in target_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
            continue

        # 回退方案：关键词
        if not keywords:
            continue
        texts = _get_retrieved_texts(results)
        for rank, text in enumerate(texts, start=1):
            if any(_norm(kw) in _norm(text) for kw in keywords):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ====== 运行评估 ======
if __name__ == "__main__":
    # show_chunks()
    print("=" * 60)
    print("RAG 评估 —— v1 纯向量 vs v3 混合+Rerank")
    print(f"测试集大小: {len(test_queries)} 条")
    print("=" * 60)

    for label, fn in [("v1 纯向量 search()", search), ("v3 混合+Rerank search_v3()", search_v3)]:
        hr = evaluate_hit_rate(test_queries, fn)
        mrr = evaluate_mrr(test_queries, fn)
        print(f"\n{label}:")
        print(f"  Hit Rate = {hr:.2%}")
        print(f"  MRR      = {mrr:.3f}")

    print("\n💡 面试话术：'我对比了 v1 纯向量和 v3 混合+Rerank 两个版本。"
          "v1 的 Hit Rate 是 X%，v3 提升到 Y%，且 MRR 从 A 提升到 B——"
          "说明混合检索不仅让更多正确答案被找到，还让它们排得更靠前。'")