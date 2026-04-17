from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable


LOGGER = logging.getLogger("sam3_finetune_lora")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def plt_settings(
    rcparams: dict[str, Any] | None = None, backend: str = "Agg"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to temporarily set matplotlib rc parameters and backend."""
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import matplotlib.pyplot as plt

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()

            if switch:
                plt.close("all")
                plt.switch_backend(backend)

            try:
                with plt.rc_context(rcparams):
                    return func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)

        return wrapper

    return decorator


from .plotting import plot_results

__all__ = ["LOGGER", "plot_results", "plt_settings"]
