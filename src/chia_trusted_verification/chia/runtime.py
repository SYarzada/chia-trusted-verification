from __future__ import annotations

from typing import Any, Callable


try:
    from chia.base.ChiaFunction import ChiaFunction, get
except ImportError as exc:
    CHIA_AVAILABLE = False
    CHIA_IMPORT_ERROR: Exception | None = exc

    def ChiaFunction(*decorator_args: Any, **decorator_kwargs: Any):
        del decorator_args, decorator_kwargs

        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            return function

        return decorate

    def get(value: Any) -> Any:
        return value
else:
    CHIA_AVAILABLE = True
    CHIA_IMPORT_ERROR = None


try:
    from chia.base.tools.ChiaTool import ChiaTool
except ImportError as exc:
    CHIA_TOOL_AVAILABLE = False
    CHIA_TOOL_IMPORT_ERROR: Exception | None = exc

    class ChiaTool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("CHIA ChiaTool support is not installed")
else:
    CHIA_TOOL_AVAILABLE = True
    CHIA_TOOL_IMPORT_ERROR = None


def require_chia() -> None:
    if not CHIA_AVAILABLE:
        raise RuntimeError(
            "CHIA is required for a search run. Install the chia package and "
            "submit this loop to a CHIA cluster."
        ) from CHIA_IMPORT_ERROR


def require_chia_tool() -> None:
    require_chia()
    if not CHIA_TOOL_AVAILABLE:
        raise RuntimeError("The installed CHIA version does not provide ChiaTool") from CHIA_TOOL_IMPORT_ERROR
