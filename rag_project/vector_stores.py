#在线阶段，获取retriever，向量存储服务
from langchain_chroma import Chroma

from AI.rag_project import config_data as config


class VectorStoreService(object):
    def __init__(self,embedding,db_name: str):
        db_conf = config.VECTOR_DBS[db_name]
        self.db_name = db_name

        self.embedding = embedding
        self.vector_store=Chroma(
            collection_name=db_conf["collection_name"],
            embedding_function=self.embedding,
            persist_directory=db_conf["persist_directory"],
        )

    def get_retriever(self):
        #返回向量检索，方便加入chain
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})

if __name__=='__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4"),"paper1").get_retriever()
    res=retriever.invoke("请问DNN模型并行推理有哪几种")
    print(res)