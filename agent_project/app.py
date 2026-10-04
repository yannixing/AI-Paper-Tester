import time
from utils.config_handler import chroma_conf
import streamlit as st
from agent.react_agent import ReactAgent

st.title("AI智能论文问答助手Agent")
st.divider()

# ========== 1. 从配置文件读取知识库列表 ==========
db_options = list(chroma_conf["VECTOR_DBS"].keys())   # ["paper1", "paper2", "paper3"]
selected_db = st.selectbox("选择知识库", db_options)

# ========== 2. 按库缓存 Agent ==========
if "agents" not in st.session_state:
    st.session_state["agents"] = {}

if selected_db not in st.session_state["agents"]:
    st.session_state["agents"][selected_db] = ReactAgent(selected_db)

agent = st.session_state["agents"][selected_db]

# ========== 3. 消息按库隔离 ==========
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

prompt=st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)
    messages.append({"role":"user","content":prompt})

    response_messages=[]
    with st.spinner("智能体思考中……"):
        res_stream=agent.execute_stream(prompt, user_id="user_001")
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        full_response = "".join(response_messages)
        messages.append({"role": "assistant", "content": full_response})

