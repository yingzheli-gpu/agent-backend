"""
DeepSearch Agent 桩实现

注意：原始实现依赖 deepagents 库，此为桩实现。
"""

from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel

# 桩类替代 deepagents 导入
class StubDeepAgent:
    """桩代理类"""
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def invoke(self, *args, **kwargs):
        return {"result": "stub - not implemented"}

def create_deep_search_agent(**kwargs) -> StubDeepAgent:
    """创建深度搜索代理 - 桩实现"""
    return StubDeepAgent(**kwargs)


class FilesystemMiddleware:
    """文件系统中间件 - 桩实现"""
    def __init__(self, **kwargs):
        pass


class SubAgentMiddleware:
    """子代理中间件 - 桩实现"""
    def __init__(self, **kwargs):
        pass


class CompositeBackend:
    """复合后端 - 桩实现"""
    def __init__(self, **kwargs):
        pass


class StateBackend:
    """状态后端 - 桩实现"""
    def __init__(self, **kwargs):
        pass


class StoreBackend:
    """存储后端 - 桩实现"""
    def __init__(self, **kwargs):
        pass


# LangChain 中间件桩实现
class TodoListMiddleware:
    def __init__(self, **kwargs):
        pass

class SummarizationMiddleware:
    def __init__(self, **kwargs):
        pass

class ModelCallLimitMiddleware:
    def __init__(self, **kwargs):
        pass

class ToolCallLimitMiddleware:
    def __init__(self, **kwargs):
        pass

class ModelFallbackMiddleware:
    def __init__(self, **kwargs):
        pass
