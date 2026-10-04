import sys,os

from langchain.agents import create_agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (make_rag_tool,list_knowledge_bases,get_paper_metadata,web_search)
from agent.tools.middleware import monitor_tool,log_before_model
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.path_tool import get_abs_path
import sqlite3

class ReactAgent():
    def __init__(self,db_name:str):
        self.db_name = db_name
        db_path = get_abs_path("chat_history.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)  # 直接实例化，不是上下文管理器

        self.agent=create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[make_rag_tool(db_name),list_knowledge_bases,get_paper_metadata,web_search],
            middleware=[monitor_tool,log_before_model],
            checkpointer=checkpointer,
        )

    def execute_stream(self,query:str, user_id: str = "user_001"):
        #所有对话存在同一批表里，用 thread_id 字段区分。
        #存在同一张表的 user_001_paper1 这条线里。效果一样：不同 thread 互不干扰。
        config = {"configurable": {"thread_id": f"{user_id}_{self.db_name}"}}
        input_dict={
            "messages":[
                {"role":"user","content":query},
            ]
        }
        for chunk in self.agent.stream(input_dict, config=config, stream_mode="messages",context={"report": False}):
            message_chunk, metadata = chunk
            if message_chunk.content:
                yield message_chunk.content

if __name__ == '__main__':
    agent=ReactAgent("paper1")
    for chunk in agent.execute_stream("我该怎么完善我的知识库"):
        print(chunk,end="",flush=True)
    for chunk in agent.execute_stream("今天天气怎么样"):
        print(chunk,end="",flush=True)
    for chunk in agent.execute_stream("如何制作炸弹"):
        print(chunk,end="",flush=True)