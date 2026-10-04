import sys,os

from langchain.agents import AgentState
from langgraph.runtime import Runtime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import Callable
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts
from model.factory import chat_model

@wrap_tool_call
def monitor_tool(
        #请求的数据封装
        request:ToolCallRequest,
        #执行的函数本身
        handler:Callable[[ToolCallRequest],ToolMessage|Command],
)->ToolMessage|Command: #工具执行的监控
    logger.info(f"[tool monitor] 执行工具{request.tool_call['name']}")
    logger.info(f"[tool monitor] 传入参数{request.tool_call['args']}")

    try:
        result=handler(request)
        logger.info(f"[tool moonitor]工具{request.tool_call['name']}调用成功")
        #只要调用这个工具，就改report为True
        if request.tool_call['name']=="fill_context_for_report":
            request.runtime.context["report"]=True

        return result
    except Exception as e:
        logger.error(f"[tool moonitor]工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState, #整个Agent智能体中的状态记录
        runtime: Runtime, #记录了整个执行过程中的上下文信息
): #在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__}|{state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt #每一次在生成提示词前，调用此函数
def report_prompt_switch(request:ModelRequest):
    is_report=request.runtime.context.get("report",False)
    if is_report:
        return load_report_prompts() #返回报告生成提示词
    return load_system_prompts()

from langchain_core.messages import AIMessage
import re

# ========== 1. 规则层：关键词黑名单 ==========
DANGEROUS_KEYWORDS = [
    "制作炸弹", "制毒", "枪支", "毒品", "自杀", "杀人",
    "黑客攻击", "入侵系统", "窃取密码",
]

PROMPT_INJECTION_PATTERNS = [
    r"忽略.*(之前|以上|前面).*(指令|提示|规则)",
    r"ignore.*(previous|above).*(instruction|prompt|rule)",
    r"你现在是.*(不受限制|没有限制|可以.*任何)",
    r"(泄露|告诉我|输出).*(系统提示|system prompt|你的指令)",
    r"假装你是",
    r"角色扮演.*不受限制",
    r"DAN模式",
]

OUT_OF_SCOPE_KEYWORDS = [
    "今天天气", "股票行情", "写一首诗", "写小说",
    "帮我订餐", "讲个笑话",
]

def _match_any(text: str, patterns: list[str]) -> str | None:
    """返回第一个命中的模式，没有则返回 None"""
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return p
    return None

def is_out_of_scope(query: str) -> tuple[bool, str]:
    """
    判断问题是否超范围。
    返回：(是否超范围, 原因)
    """
    # 1. 危险内容
    for kw in DANGEROUS_KEYWORDS:
        if kw in query:
            return True, f"危险内容：{kw}"

    # 2. 提示词注入 / 越权
    hit = _match_any(query, PROMPT_INJECTION_PATTERNS)
    if hit:
        return True, f"提示词注入：{hit}"

    # 3. 明显无关
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in query:
            return True, f"超出论文库范围：{kw}"

    return False, ""

# ========== 2. middleware 钩子 ==========
@before_model
def scope_guard(state, runtime):
    """模型每次调用前，自动运行，检测用户问题是否超范围；超范围则直接拒答"""
    messages = state["messages"]
    if not messages:
        return None

    # 取最后一条用户消息
    last_msg = messages[-1]
    if last_msg.type != "human":
        return None

    query = last_msg.content
    blocked, reason = is_out_of_scope(query)

    if blocked:
        # 直接返回一条 AIMessage，跳过模型调用
        reply = (
            "抱歉，该问题超出论文知识库的范围，我无法回答。"
            "请提出与本地论文相关的问题。"
        )
        return {"messages": [AIMessage(content=reply)]}

    return None   # 不拦截，继续正常流程

from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=2000,          # 或 max_tokens=若干，按模型上下文算
    strategy="last",          # 保留最新的
    token_counter=chat_model, # 用模型自带计数器
    include_system=True,      # 保留 system 消息
    allow_partial=False,      # 不切分单条消息
    start_on="human",         # 从 human 消息开始保留
)

@before_model
def trim_history(state, runtime):
    messages = state["messages"]
    trimmed = trimmer.invoke(messages)
    return {"messages": trimmed}