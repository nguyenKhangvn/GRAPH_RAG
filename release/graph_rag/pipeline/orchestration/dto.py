from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.core.intents import IntentType
from graph_rag.pipeline.orchestration.query_frame_contract import QueryFrame
from graph_rag.pipeline.orchestration.query_plan import PipelineRuntime, QueryPlan


class FreezableDict(dict):
    """Dict that becomes read-only after freeze() is called.

    After freeze(), all write operations (set, update, pop, clear, etc.)
    raise RuntimeError. Use PipelineRuntime.metadata for mutable state
    after Step 1.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_frozen", False)

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    @property
    def is_frozen(self) -> bool:
        try:
            return object.__getattribute__(self, "_frozen")
        except AttributeError:
            return False

    def _check_writable(self, key: Any = None) -> None:
        if self.is_frozen:
            raise RuntimeError(
                f"FreezableDict is read-only after freeze(). "
                f"Attempted to modify key '{key}'. "
                f"Use PipelineRuntime.metadata for mutable state."
            )

    # --- Write operations (raise if frozen) ---

    def __setitem__(self, key: Any, value: Any) -> None:
        self._check_writable(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        self._check_writable(key)
        super().__delitem__(key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._check_writable("update")
        super().update(*args, **kwargs)

    def pop(self, key: Any, *args: Any) -> Any:
        self._check_writable(key)
        return super().pop(key, *args)

    def popitem(self) -> tuple[Any, Any]:
        self._check_writable("popitem")
        return super().popitem()

    def clear(self) -> None:
        self._check_writable("clear")
        super().clear()

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self._check_writable(key)
        return super().setdefault(key, default)




@dataclass
class PipelineRunState:
    """Active per-request state used by `PipelineApplicationService`.

    This is the production orchestration state.

    Note:
        - `query_plan`: Frozen business intent (immutable after Step 1)
        - `runtime`: Mutable execution state (updated in Step 2-5)
        - `query_frame`: Semantic contract (legacy, will be replaced by query_plan)
        - `metadata`: Runtime auxiliary data (debug, scores, raw output, etc.)
    """

    user_query: str
    history: List[Dict[str, Any]] = field(default_factory=list)

    # NEW: Mutable runtime state (updated in Step 2-5)
    runtime: PipelineRuntime = field(default_factory=PipelineRuntime)

    # Semantic contract - ý nghĩa ổn định cho retrieval/generation
    query_frame: QueryFrame = field(default_factory=QueryFrame)

    # Runtime auxiliary - debug, scores, temporary data
    # Giữ lại cho backward compatibility, sẽ migrate dần
    metadata: Dict[str, Any] = field(default_factory=FreezableDict)

    search_query: str = ""
    primary_intent: IntentType = IntentType.DISCOVERY
    entities: List[Dict[str, Any]] = field(default_factory=list)
    location_context: Dict[str, Any] = field(default_factory=dict)
    location: str = ""
    region_focus: str = "all"
    has_explicit_location: bool = False
    grounded_nodes: List[Any] = field(default_factory=list)
    all_seeds: List[Any] = field(default_factory=list)
    raw_context: List[str] = field(default_factory=list)
    clean_context: str = ""
    answer: str = ""
    query_state: Any = None

    # NEW: Frozen business intent (immutable after Step 1)
    query_plan: QueryPlan = field(default_factory=QueryPlan)

    # NEW: Resolved follow-up context from ConversationStateResolver
    resolved_query_frame: Optional[Any] = None


    def __post_init__(self) -> None:
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "primary_intent" and not isinstance(value, IntentType):
            value = IntentType.from_value(value)
        super().__setattr__(name, value)



