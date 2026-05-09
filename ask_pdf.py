from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "./chroma_db"

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        paper_name = doc.metadata.get("paper_name") or doc.metadata.get("source") or "未知论文"
        page = doc.metadata.get("page", "未知页码")

        formatted.append(
            f"【片段{i}】\n"
            f"论文名称：{paper_name}\n"
            f"页码：第{page}页\n"
            f"内容：{doc.page_content}"
        )

    return "\n\n".join(formatted)

def format_sources(docs):
    sources = []
    seen = set()

    for doc in docs:
        paper_name = doc.metadata.get("paper_name") or doc.metadata.get("source") or "未知论文"
        page = doc.metadata.get("page", "未知页码")
        key = (paper_name, page)

        if key not in seen:
            seen.add(key)
            sources.append(f"- {paper_name}，第{page}页")

    return "\n".join(sources)

# 1. 加载 Embedding
embeddings = HuggingFaceEmbeddings(#本地加载开源 embedding 模型的封装。
        model="BAAI/bge-small-zh-v1.5"
)

# 2. 加载 Chroma 向量库
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(#生成的检索器。
    search_kwargs={"k": 4}
)

# 3. 加载大模型
llm = ChatOpenAI(
    model="Qwen/Qwen3.5-27B",
    api_key="ms-b73f4199-18a1-4529-a680-f47c8acbcbab",
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.2
)

# 4. Prompt 模板
prompt = ChatPromptTemplate.from_template("""
你是一个论文阅读助手。请只根据给定论文片段回答问题。
如果论文片段中没有相关信息，请回答“根据当前论文内容无法确定”。

论文片段：
{context}

问题：
{question}

请用中文回答，并尽量指出依据来自哪篇论文/哪一页。
""")

# 5. RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,#用用户问题去检索论文，然后把检索结果格式化。
        "question": RunnablePassthrough()#用户输入的问题，直接作为 {question}。
    }#构造 Prompt 需要的两个变量：context和question
    | prompt
    | llm
)

while True:
    question = input("\n请输入你的问题（输入 exit 退出）：")
    if question.lower() in ["exit", "quit"]:#把字符串全部转成小写
        break

    # answer = rag_chain.invoke(question)
    # print("\n回答：")
    # print(answer.content)

    # 先手动检索，拿到文档
    docs = retriever.invoke(question)
    # 再把文档内容交给大模型
    context = format_docs(docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })
    answer = llm.invoke(final_prompt)

    print("\n回答：")
    print(answer.content)

    print("\n参考来源：")
    print(format_sources(docs))