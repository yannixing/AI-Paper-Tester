import sys,os

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loder,pdf_loder,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger
from model.factory import embed_model
_bm25_cache = {}#模块级变量，只要 Python 进程不重启，它就一直存在，跨 Agent、跨知识库切换、跨请求都共享
class VectorStoreService(object):
    def __init__(self,db_name: str):
        self.db_conf = chroma_conf["VECTOR_DBS"][db_name]
        self.db_name = db_name
        self.vector_store=Chroma(
            collection_name=self.db_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(self.db_conf["persist_directory"]),
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        #返回向量检索，方便加入chain
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_conf["top_k_threshold"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        计算文件的MD5做去重
        :return: None
        """
        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                #创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]),"w",encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():
                    line=line.strip()
                    if line==md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path:str):
            if read_path.endswith("txt"):
                return txt_loder(read_path)

            if read_path.endswith("pdf"):
                return pdf_loder(read_path)

            return []

        allowed_files_path:list[str]=listdir_with_allowed_type(self.db_conf["data_path"],
                                                     tuple(chroma_conf["allow_knowledge_file_type"]))

        has_new_document = False  # ← 标记本次是否有新文档入库
        for path in allowed_files_path:
            md5_hex=get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已加载在知识库{self.db_name}中，跳过")
                continue

            try:
                documents:list[Document]=get_file_documents(path)
                if not documents:
                    logger.warnning(f"[加载知识库]{path}内容没有有效文本内容，跳过")
                    continue
                split_document:list[Document] =self.spliter.split_documents(documents)
                if not split_document:
                    logger.warnning(f"[加载知识库]{path}内容分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                has_new_document = True  # ← 有新文档，置 True
                logger.info(f"[加载知识库]加载成功！{path}内容已加载在知识库{self.db_name}中")
            except Exception as e:
                #exc_info=True会记录详细的报错堆栈
                logger.error(f"[加载知识库]{path}内容加载失败！{str(e)}",exc_info=True)
                continue
        if has_new_document and self.db_name in _bm25_cache:
            del _bm25_cache[self.db_name]
            logger.info(f"[BM25]{self.db_name} 有新文档入库，缓存已清除，下次检索将重建")

    def _get_all_documents(self):
        """从 Chroma 取出所有文档，用于建 BM25 索引"""
        data = self.vector_store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]

    def _build_bm25_retriever(self):
        """构建 BM25 关键词检索器（中文用 jieba 分词）"""
        if self.db_name in _bm25_cache:
            return _bm25_cache[self.db_name]

        docs = self._get_all_documents()
        if not docs:
            return None

        def jieba_tokenize(text: str):
            return list(jieba.cut(text))

        bm25 = BM25Retriever.from_documents(#在内存里建索引
            docs,
            preprocess_func=jieba_tokenize,  # 中文分词
        )
        bm25.k = chroma_conf["top_k_threshold"]
        _bm25_cache[self.db_name] = bm25
        return bm25

    def get_hybrid_retriever(self):
        """向量 + BM25 双路召回，RRF 融合"""
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["top_k_threshold"]}
        )
        bm25_retriever = self._build_bm25_retriever()

        # BM25 索引为空时降级为纯向量检索
        if bm25_retriever is None:
            return vector_retriever

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],  # 可调，BM25 权重高则更偏关键词
        )#EnsembleRetriever 内部默认就是用 RRF 融合的。


if __name__ == "__main__":
    vs=VectorStoreService("paper1")
    #vs.load_document()
    retriever=vs.get_hybrid_retriever()
    res=retriever.invoke("模型并行是什么")
    for r in res:
        print(r.page_content)
        print("-"*20)
