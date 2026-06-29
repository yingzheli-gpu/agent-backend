"""
上下文工程模块

提供智能上下文管理功能：
- MessagePriority: 消息优先级分配
- CompressionStrategy: 压缩策略选择
- TCMSummarizer: TCM 专用摘要
- ToolTrimmer: 工具消息裁剪
- HierarchicalMemory: 分层记忆
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .message_priority import (
    MessagePriority,
    MessagePriorityAssigner,
    PrioritizedMessage,
)

# Stub implementations for compression_strategy (original is commented out)
class CompressionLevel(Enum):
    """压缩级别"""
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    AGGRESSIVE = 3

@dataclass
class CompressionStrategy:
    """压缩策略"""
    level: CompressionLevel
    target_tokens: int
    keep_system: bool = True
    keep_last_n: int = 3
    summarize_middle: bool = False
    drop_low_priority: bool = False
    trim_tool_results: bool = False


class CompressionStrategySelector:
    """
    压缩策略选择器（与 TCMContextManagerMiddleware 对齐）

    原 v1 实现见 context/v1/compression_strategy.py（已注释）；此处提供等价可运行版本。
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        warning_threshold: float = 0.7,
        light_threshold: float = 0.8,
        medium_threshold: float = 0.9,
        aggressive_threshold: float = 0.95,
    ):
        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self.light_threshold = light_threshold
        self.medium_threshold = medium_threshold
        self.aggressive_threshold = aggressive_threshold

    def select_strategy(
        self,
        current_tokens: int,
        message_count: int = 0,
    ) -> CompressionStrategy:
        if self.max_tokens <= 0:
            return CompressionStrategy(
                level=CompressionLevel.NONE,
                target_tokens=8000,
            )
        usage_ratio = current_tokens / self.max_tokens

        if usage_ratio >= self.aggressive_threshold:
            return CompressionStrategy(
                level=CompressionLevel.AGGRESSIVE,
                target_tokens=int(self.max_tokens * 0.6),
                keep_system=True,
                keep_last_n=2,
                summarize_middle=True,
                drop_low_priority=True,
                trim_tool_results=True,
            )
        if usage_ratio >= self.medium_threshold:
            return CompressionStrategy(
                level=CompressionLevel.MEDIUM,
                target_tokens=int(self.max_tokens * 0.7),
                keep_system=True,
                keep_last_n=3,
                summarize_middle=True,
                drop_low_priority=True,
                trim_tool_results=True,
            )
        if usage_ratio >= self.light_threshold:
            return CompressionStrategy(
                level=CompressionLevel.LIGHT,
                target_tokens=int(self.max_tokens * 0.75),
                keep_system=True,
                keep_last_n=5,
                summarize_middle=False,
                drop_low_priority=True,
                trim_tool_results=False,
            )
        return CompressionStrategy(
            level=CompressionLevel.NONE,
            target_tokens=self.max_tokens,
            keep_system=True,
            keep_last_n=10,
            summarize_middle=False,
            drop_low_priority=False,
            trim_tool_results=False,
        )

    def apply_strategy(
        self,
        prioritized_messages: List[PrioritizedMessage],
        strategy: CompressionStrategy,
        summarizer: Optional[Any] = None,
    ) -> List[Any]:
        if strategy.level == CompressionLevel.NONE:
            return [pm.message for pm in prioritized_messages]

        result: List[Any] = []
        total = len(prioritized_messages)
        system_messages: List[PrioritizedMessage] = []
        head_messages: List[PrioritizedMessage] = []
        middle_messages: List[PrioritizedMessage] = []
        tail_messages: List[PrioritizedMessage] = []

        for i, pm in enumerate(prioritized_messages):
            if pm.priority == MessagePriority.CRITICAL:
                system_messages.append(pm)
            elif i < strategy.keep_last_n:
                head_messages.append(pm)
            elif i >= total - strategy.keep_last_n:
                tail_messages.append(pm)
            else:
                middle_messages.append(pm)

        result.extend([pm.message for pm in system_messages])
        result.extend([pm.message for pm in head_messages])

        if strategy.summarize_middle and middle_messages:
            if strategy.drop_low_priority:
                filtered = [
                    pm for pm in middle_messages
                    if pm.priority.value >= MessagePriority.NORMAL.value
                ]
                result.extend([pm.message for pm in filtered])
            else:
                result.extend([pm.message for pm in middle_messages])
        elif not strategy.drop_low_priority:
            result.extend([pm.message for pm in middle_messages])
        else:
            filtered = [
                pm for pm in middle_messages
                if pm.priority.value >= MessagePriority.NORMAL.value
            ]
            result.extend([pm.message for pm in filtered])

        result.extend([pm.message for pm in tail_messages])
        return result

    def get_usage_status(self, current_tokens: int) -> Dict[str, Any]:
        usage_ratio = (
            current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
        )
        if usage_ratio >= self.aggressive_threshold:
            status = "critical"
        elif usage_ratio >= self.medium_threshold:
            status = "warning"
        elif usage_ratio >= self.light_threshold:
            status = "notice"
        elif usage_ratio >= self.warning_threshold:
            status = "approaching"
        else:
            status = "normal"

        return {
            "status": status,
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "usage_ratio": usage_ratio,
            "remaining_tokens": max(0, self.max_tokens - current_tokens),
        }

from .summarization import (
    TCMSummarizer,
    TCMSummaryResult,
    TCM_SUMMARY_TEMPLATE,
)
from .tool_trimmer import (
    ToolType,
    ToolRound,
    TrimmedToolRound,
    SmartToolTrimmer,
)

# Stub implementations for hierarchical_memory (original is in memory/v1)
class MemoryLevel(Enum):
    """记忆级别"""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"

@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    level: MemoryLevel
    timestamp: float
    metadata: Dict[str, Any] = None

@dataclass
class PatientProfile:
    """患者画像"""
    patient_id: str
    basic_info: Dict[str, Any]
    medical_history: List[MemoryEntry]

class HierarchicalMemory:
    """分层记忆管理（与 TCMContextManagerMiddleware 对齐：工作记忆 + 简单 episodic 占位）"""

    def __init__(
        self,
        working_capacity: int = 10,
        episodic_capacity: int = 50,
        patient_id: str = "",
    ):
        self.working_capacity = max(1, working_capacity)
        self.episodic_capacity = max(1, episodic_capacity)
        self.patient_id = patient_id
        self._working: List[Dict[str, Any]] = []
        self._episodic: List[MemoryEntry] = []
        self.memories: List[MemoryEntry] = self._episodic

    def add_to_working(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._working.append(
            {
                "content": content,
                "importance": importance,
                "metadata": metadata or {},
            }
        )
        if len(self._working) > self.working_capacity:
            self._working = self._working[-self.working_capacity :]

    def add_memory(self, content: str, level: MemoryLevel, metadata: Optional[Dict] = None):
        entry = MemoryEntry(
            content=content, level=level, timestamp=0.0, metadata=metadata
        )
        self._episodic.append(entry)
        if len(self._episodic) > self.episodic_capacity:
            self._episodic = self._episodic[-self.episodic_capacity :]
        self.memories = self._episodic

    def get_context_for_prompt(self, query: str, max_tokens: int) -> str:
        if not self._working:
            return ""
        parts = [str(x.get("content", "")) for x in self._working[-8:]]
        text = "\n".join(parts)
        approx_chars = max(1, max_tokens) * 2
        if len(text) > approx_chars:
            text = text[-approx_chars:]
        return text

    def get_stats(self) -> Dict[str, Any]:
        return {
            "working_size": len(self._working),
            "episodic_size": len(self._episodic),
        }

__all__ = [
    # Message Priority
    "MessagePriority",
    "MessagePriorityAssigner",
    "PrioritizedMessage",

    # Compression Strategy
    "CompressionLevel",
    "CompressionStrategy",
    "CompressionStrategySelector",

    # Summarization
    "TCMSummarizer",
    "TCMSummaryResult",
    "TCM_SUMMARY_TEMPLATE",

    # Tool Trimmer
    "ToolType",
    "ToolRound",
    "TrimmedToolRound",
    "SmartToolTrimmer",

    # Hierarchical Memory
    "MemoryLevel",
    "MemoryEntry",
    "PatientProfile",
    "HierarchicalMemory",
]
