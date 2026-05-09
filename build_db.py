from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil

PDF_DIR = "D:/Desktop/毕业论文/论文/研究现状1"
DB_DIR = "./chroma_db"


def build_vector_db():
    # 0. 删除旧向量库，确保是重新建库
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("已删除旧向量库")

    # 1. 读取文件夹下所有 PDF
    loader = PyPDFDirectoryLoader(PDF_DIR)
    docs = loader.load()

    print(f"读取到 {len(docs)} 页 PDF 文档")

    # 2. 给每一页文档添加论文名称
    for doc in docs:
        pdf_path = doc.metadata.get("source", "")
        pdf_filename = os.path.basename(pdf_path)
        paper_name = os.path.splitext(pdf_filename)[0]

        doc.metadata["paper_name"] = paper_name
        doc.metadata["file_name"] = pdf_filename

        # PyPDFDirectoryLoader 的 page 通常从 0 开始，这里改成从 1 开始，方便显示
        if "page" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page"] + 1

    # 3. 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    print(f"切分后得到 {len(chunks)} 个文本块")

    # 4. 生成 Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    # 5. 写入 Chroma
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("新向量库构建完成")

    # 6. 检查前几条数据
    data = db.get()

    print("\n===== 向量库检查 =====")
    print(f"共保存 {len(data['ids'])} 条文本块")

    for i in range(min(3, len(data["ids"]))):
        print(f"\n--- 第 {i + 1} 条 ---")
        print("论文名称:", data["metadatas"][i].get("paper_name"))
        print("文件名:", data["metadatas"][i].get("file_name"))
        print("页码:", data["metadatas"][i].get("page"))
        print("内容预览:", data["documents"][i][:150])


if __name__ == "__main__":
    build_vector_db()