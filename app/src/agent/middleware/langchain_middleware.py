"""
LangChain 内置中间件适配层（TCM 桩实现）

桩类继承 langchain.agents.middleware.types.AgentMiddleware，保证：
- 若被误入 create_agent(middleware=[...])，具备 name、abefore_model 等与工厂一致的契约；
- LangChain 的 wrap_tool_call(request, handler) 与 TCM MiddlewareChain 旧式三参数包装分离，
  后者仅通过 tcm_wrap_tool_call 走 LangChainMiddlewareWrapper。

更新日期：2026-03-24
"""

from typing import Optional, List, Any, Callable, Dict
from dataclasses import dataclass, field
import logging

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)


# ============================================================
# 桩实现类（替代 LangChain 官方中间件）
# ============================================================

class StubMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    """桩中间件基类：同时满足 TCM MiddlewareChain 与 LangChain create_agent 的契约。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.config = kwargs
        self.tools: list[Any] = []

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return None

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(request)

    def tcm_wrap_tool_call(
        self, tool_call: Callable[..., Any], tool_name: str, state: Dict[str, Any]
    ) -> Callable[..., Any]:
        """供 TCM MiddlewareChain 使用的旧式 (tool_call, tool_name, state) 包装。"""
        return tool_call


class ModelCallLimitMiddleware(StubMiddleware):
    """模型调用限制中间件 - 桩实现"""
    pass


class ToolCallLimitMiddleware(StubMiddleware):
    """工具调用限制中间件 - 桩实现"""
    pass


class ModelFallbackMiddleware(StubMiddleware):
    """模型降级中间件 - 桩实现"""
    pass


class ToolRetryMiddleware(StubMiddleware):
    """工具重试中间件 - 桩实现"""
    pass


class ModelRetryMiddleware(StubMiddleware):
    """模型重试中间件 - 桩实现"""
    pass


class SummarizationMiddleware(StubMiddleware):
    """摘要中间件 - 桩实现"""
    pass


# ============================================================
# 包装器类（统一接口）
# ============================================================

class LangChainMiddlewareWrapper:
    """LangChain 中间件轻量包装器"""
    
    def __init__(self, wrapped_middleware, priority: int, name: str, enabled: bool = True):
        self.wrapped = wrapped_middleware
        self.priority = priority
        self.name = name
        self.enabled = enabled
    
    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    # MiddlewareChain 会调用这些方法；桩类 StubMiddleware 无实现，必须在此显式兜底
    def before_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        fn = getattr(self.wrapped, "before_model", None)
        if callable(fn):
            return fn(state, runtime)
        return None

    def after_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        fn = getattr(self.wrapped, "after_model", None)
        if callable(fn):
            return fn(state, runtime)
        return None

    def wrap_tool_call(
        self, tool_call: Callable, tool_name: str, state: Dict[str, Any]
    ) -> Callable:
        fn = getattr(self.wrapped, "tcm_wrap_tool_call", None)
        if callable(fn):
            return fn(tool_call, tool_name, state)
        return tool_call


# ============================================================
# 配置类
# ============================================================

@dataclass
class TCMLangChainMiddlewareConfig:
    """LangChain 中间件统一配置"""

    model_call_thread_limit: int = 50
    model_call_run_limit: int = 10
    model_call_exit_behavior: str = "end"

    tool_call_limits: dict = field(default_factory=lambda: {
        "web_search": 5,
        "kg_syndrome_search": 10,
    })
    tool_call_exit_behavior: str = "continue"

    enable_model_fallback: bool = True
    fallback_models: List[str] = field(default_factory=lambda: [
        "deepseek:deepseek-chat",
        "openai:gpt-4o-mini",
    ])

    enable_tool_retry: bool = True
    tool_retry_max_retries: int = 3
    tool_retry_backoff_factor: float = 2.0

    enable_model_retry: bool = True
    model_retry_max_retries: int = 2
    model_retry_backoff_factor: float = 1.5

    enable_summarization: bool = False  # 默认禁用
    summarization_model: str = "deepseek:deepseek-chat"
    summarization_trigger_tokens: int = 6000
    summarization_trigger_messages: int = 20
    summarization_keep_messages: int = 10

    summarization_prompt: str = """请对以下中医对话进行摘要，保留关键医学信息。"""


# ============================================================
# 工厂函数
# ============================================================

def get_model_call_limit_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> LangChainMiddlewareWrapper:
    config = config or TCMLangChainMiddlewareConfig()

    wrapped = ModelCallLimitMiddleware(
        thread_limit=config.model_call_thread_limit,
        run_limit=config.model_call_run_limit,
        exit_behavior=config.model_call_exit_behavior,
    )

    logger.info(f"ModelCallLimitMiddleware (stub) 已创建")
    
    return LangChainMiddlewareWrapper(
        wrapped,
        priority=10,
        name="ModelCallLimitMiddleware"
    )


def get_tool_call_limit_middlewares(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> List[LangChainMiddlewareWrapper]:
    config = config or TCMLangChainMiddlewareConfig()

    middlewares = []
    for idx, (tool_name, limit) in enumerate(config.tool_call_limits.items()):
        wrapped = ToolCallLimitMiddleware(
            tool_name=tool_name,
            thread_limit=limit,
            exit_behavior=config.tool_call_exit_behavior,
        )
        
        middleware = LangChainMiddlewareWrapper(
            wrapped,
            priority=11 + idx,
            name=f"ToolCallLimitMiddleware_{tool_name}"
        )
        middlewares.append(middleware)

    logger.info(f"创建了 {len(middlewares)} 个工具调用限制中间件 (stub)")
    return middlewares


def get_tool_retry_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> Optional[LangChainMiddlewareWrapper]:
    config = config or TCMLangChainMiddlewareConfig()

    if not config.enable_tool_retry:
        return None

    wrapped = ToolRetryMiddleware(
        max_retries=config.tool_retry_max_retries,
        backoff_factor=config.tool_retry_backoff_factor,
    )

    logger.info(f"ToolRetryMiddleware (stub) 已创建")
    
    return LangChainMiddlewareWrapper(
        wrapped,
        priority=3,
        name="ToolRetryMiddleware"
    )


def get_model_retry_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> Optional[LangChainMiddlewareWrapper]:
    config = config or TCMLangChainMiddlewareConfig()

    if not config.enable_model_retry:
        return None

    wrapped = ModelRetryMiddleware(
        max_retries=config.model_retry_max_retries,
        backoff_factor=config.model_retry_backoff_factor,
    )

    logger.info(f"ModelRetryMiddleware (stub) 已创建")
    
    return LangChainMiddlewareWrapper(
        wrapped,
        priority=2,
        name="ModelRetryMiddleware"
    )


def get_summarization_middleware(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> Optional[LangChainMiddlewareWrapper]:
    """摘要中间件 - 当前禁用"""
    logger.info("SummarizationMiddleware 当前禁用")
    return None


def get_all_langchain_middlewares(
    config: Optional[TCMLangChainMiddlewareConfig] = None
) -> List[Any]:
    config = config or TCMLangChainMiddlewareConfig()

    middlewares = []

    model_retry = get_model_retry_middleware(config)
    if model_retry:
        middlewares.append(model_retry)

    tool_retry = get_tool_retry_middleware(config)
    if tool_retry:
        middlewares.append(tool_retry)

    model_limit = get_model_call_limit_middleware(config)
    middlewares.append(model_limit)

    tool_limits = get_tool_call_limit_middlewares(config)
    middlewares.extend(tool_limits)

    logger.info(f"共创建 {len(middlewares)} 个 LangChain 中间件 (stub)")
    return middlewares


__all__ = [
    "TCMLangChainMiddlewareConfig",
    "get_model_call_limit_middleware",
    "get_tool_call_limit_middlewares",
    "get_tool_retry_middleware",
    "get_model_retry_middleware",
    "get_summarization_middleware",
    "get_all_langchain_middlewares",
]
