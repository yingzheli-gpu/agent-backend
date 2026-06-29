from typing import Any

from langchain_core.messages import SystemMessage
from ...tcm_states import TCMAgentState
from ...tcm_prompts import TCM_WELLNESS_SYSTEM_PROMPT, TCM_ERROR_RESPONSE


def _stream_chunk_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


async def respond_to_general_query(state: TCMAgentState) -> dict:
    """处理一般性回答（闲聊/通用对话）"""
    from ...tcm_builder import get_llm
    messages = state.messages
    user_profile = state.user_profile or {}

    # 使用 state.llm_config 获取 LLM
    llm = get_llm(llm_config=state.llm_config)

    # 构建系统提示
    system_prompt = """"
    你是一个乐于助人的聊天助手，你的名字叫做小义，你擅长帮助用户解决各种各样的问题。
    请你使用欢快的语气和通俗易懂的方式回答用户的问题。

    下面是交互的上下文，请根据上下文内容生成回答。
    """
    try:
        input_messages = [
            SystemMessage(content=system_prompt),
            *messages[-5:],
        ]
        parts: list[str] = []
        async for chunk in llm.astream(input_messages):
            piece = _stream_chunk_content_to_str(getattr(chunk, "content", None))
            if piece:
                parts.append(piece)
        full = "".join(parts)

        return {
            "answer": full,
            "steps": ["一般性回答生成完成"],
        }
    except Exception as e:
        return {
            "answer": TCM_ERROR_RESPONSE,
            "error": str(e),
        }
