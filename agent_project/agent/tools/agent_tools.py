import json
import sys,os
from collections import Counter
from datetime import datetime, timedelta
from langchain_chroma import Chroma
sys.path.insert(0, r'D:\Desktop\experiment\llm\AI\agent_project')
from langchain_core.tools import tool
from utils.config_handler import chroma_conf,rag_conf,agent_conf
from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path
from langchain_community.embeddings import DashScopeEmbeddings
#几个知识库的缓存，同一个库只创建一次，之后复用
_rag_cache = {}
def _get_rag(db_name: str) -> RagSummarizeService:
    if db_name not in _rag_cache:
        _rag_cache[db_name] = RagSummarizeService(db_name)
    return _rag_cache[db_name]

def make_rag_tool(db_name: str):
    @tool(description=f"从【{db_name}】知识库中检索参考资料。")
    def rag_summarize(query: str) -> str:
        """入参：query，检索用的自然语言查询。"""
        return _get_rag(db_name).rag_summarize(query)
    return rag_summarize

@tool(description ="列出当前所有可用的向量库及其概况。")
def list_knowledge_bases() -> str:
    embeddings = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
    lines = []
    for db_name, db_conf in chroma_conf["VECTOR_DBS"].items():
        try:
            chroma = Chroma(
                collection_name=db_conf["collection_name"],
                embedding_function=embeddings,
                persist_directory=get_abs_path(db_conf["persist_directory"]),
            )
            data = chroma.get()
            total = len(data["ids"])

            # 统计综述 / 研究论文数量
            review_cnt = 0
            research_cnt = 0
            for meta in data["metadatas"]:
                if meta.get("is_review") is True:
                    review_cnt += 1
                else:
                    research_cnt += 1

            lines.append(
                f"库名：{db_name} | 文本块：{total} | "
                f"综述块：{review_cnt} | 研究块：{research_cnt}"
            )
        except Exception as e:
            lines.append(f"库名：{db_name} | 读取失败：{e}")

    if not lines:
        return "当前没有任何可用的向量库。"
    return "\n".join(lines)

@tool(description ="查询某篇论文的元数据（论文名、文件名、页码、是否综述、所属库）。")
def get_paper_metadata(paper_name: str) -> str:
    embeddings = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
    keyword = paper_name.lower()
    results = []

    for db_name, db_conf in chroma_conf["VECTOR_DBS"].items():
        try:
            chroma = Chroma(
                collection_name=db_conf["collection_name"],
                embedding_function=embeddings,
                persist_directory=get_abs_path(db_conf["persist_directory"]),
            )
            data = chroma.get()

            # 用 dict 去重，避免同一论文多页重复显示
            seen = set()
            for meta in data["metadatas"]:
                pname = str(meta.get("paper_name", ""))
                fname = str(meta.get("file_name", ""))
                if keyword in pname.lower() or keyword in fname.lower():
                    key = (pname, fname, db_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        f"论文名：{pname} | 文件名：{fname} | "
                        f"是否综述：{'是' if meta.get('is_review') else '否'} | "
                        f"所属库：{db_name}"
                    )
        except Exception as e:
            results.append(f"库 {db_name} 读取失败：{e}")

    if not results:
        return f"未找到与「{paper_name}」相关的论文。"
    return "\n".join(results)

@tool(description ="联网检索最新信息。")
def web_search(query: str) -> str:
    """
    入参：
        query：检索关键词。

    返回：网页摘要与来源链接。
    注意：使用后必须在回答中标注"以下内容来自网络检索，非本地论文"。
    适用场景：仅当本地知识库无相关内容，且用户明确需要最新/外部信息时使用。
    """
    return (
        "web_search 工具尚未配置搜索后端。"
        "请在 tools/rag_tools.py 中接入具体的搜索 API 后使用。"
    )

@tool(description="无入参，无返回值，调用后触发中间件，自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

@tool(description=(
    "读取知识库问答的失败记录，用于分析优化方向。"
    "可按知识库名和最近天数过滤，返回失败问题列表及高频统计。"
))
def read_failed_queries(db_name: str = "", days: int = 7) -> str:
    """读取 failed_queries.jsonl，返回失败问题及高频统计。

    入参：
        db_name：可选，只统计指定知识库，如 paper1。不传则统计全部。
        days：可选，只看最近 N 天的记录，默认 7 天。
    """
    log_path = get_abs_path(agent_conf["failed_queries_path"])
    if not os.path.exists(log_path):
        return "暂无失败记录。"

    cutoff = datetime.now() - timedelta(days=days)
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 时间过滤
            try:
                t = datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            if t < cutoff:
                continue

            # 库过滤
            if db_name and item.get("db_name") != db_name:
                continue

            records.append(item)

    if not records:
        return f"最近 {days} 天内没有失败记录（库：{db_name or '全部'}）。"

    # 高频问题统计
    counter = Counter(r["query"] for r in records)
    top_queries = counter.most_common(10)

    # 按失败原因统计
    reason_counter = Counter(r.get("reason", "未知") for r in records)

    lines = [f"共 {len(records)} 条失败记录，最近 {days} 天。"]
    lines.append("\n【高频失败问题 Top 10】")
    for q, cnt in top_queries:
        lines.append(f"  {cnt} 次：{q}")

    lines.append("\n【失败原因分布】")
    for reason, cnt in reason_counter.most_common():
        lines.append(f"  {cnt} 次：{reason}")

    return "\n".join(lines)

if __name__ == "__main__":
    print(list_knowledge_bases())
