#在线阶段，启动对话web网页
import streamlit as st
from rag import RagService
import config_data as config

st.title("AI智能论文问答助手")
st.divider() #分隔符

# ========== 1. 选择知识库 ==========
db_options = list(config.VECTOR_DBS.keys())   # ["paper1", "paper2", ...]
selected_db = st.selectbox("选择知识库", db_options)

# ========== 2. 按库缓存 RagService ==========
if "rags" not in st.session_state:
    st.session_state["rags"] = {}

if selected_db not in st.session_state["rags"]:
    st.session_state["rags"][selected_db] = RagService(selected_db)

rag = st.session_state["rags"][selected_db]

# ========== 3. 每个库独立的消息列表 ========== 确保不同库的记录不会显示在一起
if "messages" not in st.session_state:
    st.session_state["messages"] = {}

if selected_db not in st.session_state["messages"]:
    st.session_state["messages"][selected_db] = [
        {"role": "assistant", "content": f"你好，我是【{selected_db}】知识库助手，有什么可以帮助你？"}
    ]

messages = st.session_state["messages"][selected_db]

# ========== 4. 渲染历史消息 ==========
for message in messages:
    st.chat_message(message["role"]).write(message["content"])

prompt=st.chat_input()#输入框

if prompt:
    st.chat_message("user").write(prompt)
    messages.append({"role":"user","content":prompt})

    ai_res_list=[]
    with st.spinner("AI思考中……"):
        #res=st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        res_stream = rag.chain.stream({"input": prompt}, config.session_config)
        #流对象，可迭代，yield表达式
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        #st.chat_message("assistant").write(res)
        messages.append({"role": "assistant", "content": "".join(ai_res_list)})


