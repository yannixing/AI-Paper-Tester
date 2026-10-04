#知识库更新服务
import os
import tempfile
from datetime import datetime
from langchain_community.embeddings import DashScopeEmbeddings
from AI.rag_project import config_data as config
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class KnowledgeBaseService(object):
    def __init__(self, db_name: str):
        # 根据库名取配置，找不到就报错，避免写错路径
        if db_name not in config.VECTOR_DBS:
            raise ValueError(f"未知的向量库：{db_name}")

        db_conf = config.VECTOR_DBS[db_name]
        self.db_name = db_name
        self.persist_directory = db_conf["persist_directory"]
        self.collection_name = db_conf["collection_name"]

        os.makedirs(self.persist_directory, exist_ok=True)

        self.chroma=Chroma(
            collection_name=self.collection_name,#数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=self.persist_directory#数据库的本地存储文件夹
        )
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, #分割后文本的最大长度
            chunk_overlap=config.chunk_overlap, #连续文本段之间的字符重叠数量
            separators=config.separators, #自然段落划分的符号
            length_function=len, #使用python自带的len函数作为长度依据

        ) #文本分割器的对象

    def upload_by_str(self,data:str,filename):
        if(len(data)>config.max_split_char_number):
            knowledge_chunks:list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]
        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operators":"xyn"
        }
        self.chroma.add_texts(
            knowledge_chunks,
            meta_data=[metadata for _ in knowledge_chunks]
        )
        return "【成功】内容已成功载入向量库"

    @staticmethod
    def _is_review_paper(filename: str) -> bool:
        """根据文件名判断是否是综述论文"""
        name_lower = filename.lower()
        return any(kw in name_lower for kw in config.REVIEW_KEYWORDS)

    # 按 PDF 入库，元数据要求
    def upload_by_pdf(self, file_bytes: bytes, filename: str):
        # PyPDFLoader 需要文件路径，先写临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            #加载：一页一个 Document。
            docs = PyPDFLoader(tmp_path).load()#PyPDFLoader 只接受文件路径，不接受字节流或字符串。

            # 补元数据：paper_name / file_name / page(从1开始)
            is_review = self._is_review_paper(filename)  # 判断是否综述
            paper_name = os.path.splitext(filename)[0]
            for doc in docs:
                doc.metadata["paper_name"] = paper_name
                doc.metadata["file_name"] = filename
                doc.metadata["source"] = filename  # 覆盖临时路径
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1  # 0→1
                doc.metadata["create_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                doc.metadata["operators"] = "xyn"
                doc.metadata["is_review"] = is_review  # 综述标记
                doc.metadata["db_name"] = self.db_name  # 记一下来自哪个库

            #分块，切分不会跨页——因为每页是独立的 Document
            #把每个原 Document 的 metadata 复制一份，附加到它切出来的每一个 chunk 上。
            chunks = self.splitter.split_documents(docs)
            self.chroma.add_documents(chunks)  # 增量追加，不覆盖已有库

            tag = "综述" if is_review else "研究"
            return (f"【成功】{filename}（{tag}论文）共 {len(docs)} 页，"
                    f"切分 {len(chunks)} 块，已载入【{self.db_name}】库")
        finally:
            os.remove(tmp_path)  # 清理临时文件