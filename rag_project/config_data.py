#配置文件
#Chroma
# 多向量库配置：key 是界面上显示的名字，value 是库的参数
VECTOR_DBS = {
    "paper1": {
        "persist_directory": "./chroma_db/paper1",
        "collection_name": "paper1",
    },
    "paper2": {
        "persist_directory": "./chroma_db/paper2",
        "collection_name": "paper2",
    },
    "paper3": {
        "persist_directory": "./chroma_db/paper3",
        "collection_name": "paper3",
    },
}
# 综述论文的识别关键词（不区分大小写）
REVIEW_KEYWORDS = ["review", "综述", "survey", "overview"]

#spliter
chunk_size=1000
chunk_overlap=100
separators=["\n\n","\n",",",".","?","!","，","。","？","！"," ",""]
max_split_char_number=1000 #文本分割的阈值

#检索生成
similarity_threshold=1 #检索返回匹配的文档数量
embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#对话
session_config = {
    "configurable": {"session_id": "user_001"}
}