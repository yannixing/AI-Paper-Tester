#知识库更新主程序，streamlit
#离线阶段，文件上传知识库的网页接口
import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from pypdf import PdfReader
from io import BytesIO
from knowledge_base import KnowledgeBaseService

st.title("知识库更新服务")

# 选择要写入的向量库
db_options = ["paper1", "paper2","paper3"]   # 与 config.VECTOR_DBS 的 key 对应
selected_db = st.selectbox("选择要写入的向量库", db_options)

#file_loader
uploader_files=st.file_uploader(
    "请上传PDF文件",
    type=['pdf'],
    accept_multiple_files=True,   # True 表示接受多个文件
)

# 按库名缓存 service，切换库时不会重复初始化
if "services" not in st.session_state:
    st.session_state["services"] = {}

if selected_db not in st.session_state["services"]:
    st.session_state["services"][selected_db] = KnowledgeBaseService(selected_db)

service = st.session_state["services"][selected_db]

# if uploader_file is not None:
#     #提取文件的信息
#     file_name=uploader_file.name
#     file_type=uploader_file.type
#     file_size=uploader_file.size/1024 #KB
#     st.subheader(f"文件名{file_name}")
#     st.write(f"格式：{file_name}|大小：{file_size:.2f}KB")
#
#     try:
#         result = st.session_state["service"].upload_by_pdf(
#             uploader_file.getvalue(), file_name
#         )
#         st.success(result)
#     except Exception as e:
#         st.error(f"载入失败：{e}")

# 注意：多文件时它是 list，空的时候是 []，不是 None
if uploader_files:
    st.subheader(f"已选择 {len(uploader_files)} 个文件 → 将写入【{selected_db}】库")

    # 显示文件清单
    for f in uploader_files:
        file_size = f.size / 1024  # KB
        st.write(f"📄 {f.name} | 格式：{f.type} | 大小：{file_size:.2f} KB")

    # 点击按钮才开始入库，避免刚选文件就自动跑
    if st.button("开始载入知识库"):
        success_count = 0
        fail_count = 0

        progress = st.progress(0)   # 进度条
        status = st.empty()         # 实时状态文本

        for i, f in enumerate(uploader_files):
            status.write(f"正在处理：{f.name} ...")
            try:
                result =service.upload_by_pdf(
                    f.getvalue(), f.name
                )
                st.success(f"✅ {result}")
                success_count += 1
            except Exception as e:
                st.error(f"❌ {f.name} 载入失败：{e}")
                fail_count += 1

            # 更新进度条
            progress.progress((i + 1) / len(uploader_files))

        status.write("处理完成")
        st.info(f"共 {len(uploader_files)} 个文件：成功 {success_count} 个，失败 {fail_count} 个，写入【{selected_db}】库")
