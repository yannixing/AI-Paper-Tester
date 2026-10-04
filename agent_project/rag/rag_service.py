"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
import json
import sys,os
from datetime import datetime
from http import HTTPStatus
from typing import Tuple, List

import dashscope
from langchain_classic.retrievers import ContextualCompressionRetriever
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils.prompt_loader import load_rag_prompts
from utils.path_tool import get_abs_path
from rag.vector_store import VectorStoreService
from model.factory import chat_model
from utils.config_handler import chroma_conf,rag_conf,agent_conf


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService(object):
    def __init__(self,db_name:str):
        self.db_name = db_name
        self.vector_store = VectorStoreService(db_name)
        #self.retriever=self.vector_store.get_retriever()
        # 双路召回
        self.base_retriever = self.vector_store.get_hybrid_retriever()
        # SDK 会自动读取环境变量 DASHSCOPE_API_KEY
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

        self.prompt_text=load_rag_prompts()
        self.prompt_template=PromptTemplate.from_template(self.prompt_text)
        self.model=chat_model
        self.chain=self._init_chain()

    def _init_chain(self):
        chain=self.prompt_template|print_prompt|self.model|StrOutputParser()
        return chain

    # def retriever_doc(self,query:str)->list[Document]:
    #     return self.retriever.invoke(query)

    def retriever_doc(self, query: str) -> list[Document]:
        # 1. 双路召回，拿到候选文档列表（LangChain Document）
        candidate_docs = self.base_retriever.invoke(query)
        if not candidate_docs:
            return []

        # 2. 提取纯文本，准备传给 Rerank API
        texts = [doc.page_content for doc in candidate_docs]

        # 3. 调用 DashScope SDK 的 TextReRank
        resp = dashscope.TextReRank.call(
            model=rag_conf["rerank_model_name"],
            query=query,
            documents=texts,
            top_n=chroma_conf["top_k_threshold"],
            return_documents=False  # 我们只需要索引和分数，原文自己保留
        )

        # 4. 检查 API 是否成功，失败时降级返回原始召回结果
        if resp.status_code != HTTPStatus.OK:
            print(f"Rerank API 调用失败: {resp.code} - {resp.message}")
            return candidate_docs[:chroma_conf["top_k_threshold"]]

        # 5. 根据返回的索引和分数，重排原始 Document
        reranked_docs = []
        for item in resp.output.results:
            original_doc = candidate_docs[item.index]
            # 可选：把分数塞进元数据，方便调试
            original_doc.metadata["relevance_score"] = item.relevance_score
            reranked_docs.append(original_doc)

        return reranked_docs

    def retriever_doc_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """双路召回 + Rerank，返回 (文档, 相关性分数) 列表"""
        # 1. 双路召回，拿到候选文档列表
        candidate_docs = self.base_retriever.invoke(query)
        if not candidate_docs:
            return []

        # 2. 提取纯文本
        texts = [doc.page_content for doc in candidate_docs]

        # 3. 调用 DashScope Rerank
        resp = dashscope.TextReRank.call(
            model=rag_conf["rerank_model_name"],
            query=query,
            documents=texts,
            top_n=chroma_conf["top_k_threshold"],
            return_documents=False
        )

        # 4. Rerank 失败时降级：给一个默认分数（1.0 表示"不判定为低置信"）
        if resp.status_code != HTTPStatus.OK:
            print(f"Rerank API 调用失败: {resp.code} - {resp.message}")
            return [
                (doc, 1.0)
                for doc in candidate_docs[:chroma_conf["top_k_threshold"]]
            ]

        # 5. 按返回的索引和分数，重排并带上分数
        reranked = []
        for item in resp.output.results:
            original_doc = candidate_docs[item.index]
            score = item.relevance_score
            original_doc.metadata["relevance_score"] = score  # 同时塞进元数据
            reranked.append((original_doc, score))

        return reranked

    def _log_failure(self, query: str, reason: str):
        """记录检索失败/低置信的问题"""
        log_path = get_abs_path(agent_conf["failed_queries_path"])
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "db_name": self.db_name,
            "query": query,
            "reason": reason,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def rag_summarize(self,query:str)->str:
        scored_docs =self.retriever_doc_with_score(query)
        # ===== 检测 1：检索为空 =====
        if not scored_docs :
            self._log_failure(query, "检索结果为空")
            return f"抱歉，知识库{self.db_name}中未找到与您问题相关的内容。"

        # ===== 检测 2：低置信（用向量分数判断）=====
        # 取最高 Rerank 分数作为置信度
        max_score = max(score for _, score in scored_docs)

        # 阈值根据你的 Rerank 模型和业务调整，DashScope Rerank 通常 0~1
        if max_score < rag_conf["confidence_threshold"]:
            self._log_failure(query, f"低置信: {max_score:.3f}")
            return f"抱歉，知识库{self.db_name}中没有足够相关的内容回答该问题。"

        context=""
        counter=0
        for doc, score in scored_docs:
            counter+=1
            context+=f"【参考资料{counter}】：参考资料 {doc.page_content}|参考元数据：{doc.metadata}\n"
        return self.chain.invoke(
            {
                "input":query,
                "context":context,
            }
        )

if __name__ == '__main__':
    rag=RagSummarizeService('paper1')
    print(rag.rag_summarize("模型并行的方式有哪些"))