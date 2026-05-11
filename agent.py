from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
import os

# 从环境变量读取配置（优先），否则使用默认值
API_KEY = os.environ.get("MODELSCOPE_API_KEY")

DB_DIR = "./chroma_db"

# 1. 加载 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 2. 加载 Chroma 向量库
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 4}
)


def format_docs(docs):
    return "\n\n".join(
        f"【片段{i + 1}】\n"
        f"论文名称：{doc.metadata.get('paper_name', '未知论文')}\n"
        f"页码：第{doc.metadata.get('page', '未知页码')}页\n"
        f"内容：{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# 3. 定义工具：论文检索
@tool
def search_papers(query: str) -> str:#Agent 工具版 RAG：需要时才查向量库
    """
    当用户询问论文内容、研究现状、某种方法、某篇论文贡献、实验结果时，使用该工具检索本地论文库。
    输入应该是用户问题或适合检索的关键词。
    """
    docs = retriever.invoke(query)

    if not docs:
        return "没有检索到相关论文片段。"

    return format_docs(docs)


# 4. 加载大模型
llm = ChatOpenAI(
    model="Qwen/Qwen3.5-27B",
    api_key=API_KEY,
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.2
)

# 5. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[search_papers],
    system_prompt="""
    你是一个论文阅读助手，也可以回答AI基础概念问题。

    规则：
    1. 如果用户询问“边缘计算研究领域、本地论文库中的内容、某篇论文、研究现状、实验结果、论文贡献”，必须调用 search_papers。
    2. 如果用户只是询问通用概念，例如什么是RAG、什么是Agent、什么是Embedding，可以直接回答，不需要调用 search_papers。
    3. 如果调用 search_papers 后没有相关信息，请回答“根据当前论文内容无法确定”。
    4. 回答论文相关问题时，尽量指出依据来自哪篇论文、哪一页。
    5. 不要编造论文标题、作者、页码或实验结果。
    6. 用中文回答。
    """
)

# 6. 命令行对话
chat_history = []

MAX_HISTORY = 10

while True:
    question = input("\n请输入你的问题（输入 exit 退出）：")

    if question.lower() in ["exit", "quit"]:
        break

    chat_history.append({
        "role": "user",
        "content": question
    })

    current_messages = chat_history[-MAX_HISTORY:]

    try:
        result = agent.invoke({
            "messages": current_messages# 单个问题不会报错，多个问题可能报错
        })

        answer = result["messages"][-1].content

        print("\n回答：")
        print(answer)

        chat_history.append({
            "role": "assistant",
            "content": answer[:500]
        })

    except Exception as e:
        print("\n运行出错：")
        print(e)
