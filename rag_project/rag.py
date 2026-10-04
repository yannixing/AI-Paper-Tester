#在线阶段，rag核心服务
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from AI.rag_project.file_history_store import get_history
from AI.rag_project.vector_stores import VectorStoreService
from AI.rag_project import config_data as config


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self,db_name: str):
        self.db_name = db_name
        self.vector_service=VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
            db_name=db_name
        )

        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的已知参考资料为主，简洁和专业地回答用户问题。参考资料：{context}。"),
                ("system","并且我提供用户的对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}")
            ]
        )
        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain=self._get_chain()

    def _get_chain(self):
        retriever=self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无参考资料"
            formatted_str=""
            for doc in docs:
                formatted_str+=f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value:dict)->str:
            return value['input']

        def format_for_prompt_template(value):
            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["context"]=value["context"]
            new_value["history"]=value["input"]["history"]
            return new_value

        chain=(
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever)|retriever|format_document
            }|RunnableLambda(format_for_prompt_template)| self.prompt_template |print_prompt|self.chat_model|StrOutputParser()
        )
        #增强链
        conversation_chain=RunnableWithMessageHistory(
            chain,
            lambda sid: get_history(f"{sid}_{self.db_name}"),  # 自动拼库名
            input_messages_key="input",#从输入字典的哪个 key 里取"当前用户消息"
            history_messages_key="history",#从 get_history(session_id) 拿到历史消息后，塞进字典的 "history" 这个 key
        )
        return conversation_chain

if __name__ == '__main__':
    # 固定格式，添加Langchain的配置，为当前程序配置所属的session_id
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    #DNN模型并行有哪几种方式
    #文中的这种并行推理方式更详细的解释是什么
    #刚刚的那篇论文的摘要是什么
    #请帮我检索相似的论文
    res=RagService("paper1").chain.invoke({"input":"文中的这种并行推理方式更详细的解释是什么，用中文回答"},session_config)
    print(res)