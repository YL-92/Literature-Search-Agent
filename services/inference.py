"""Shared boundary for optional external-model inference."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from config import Settings

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


def attempt_inference(
    operation: Callable[[], T], settings: Settings, label: str
) -> T | None:
    """Retry an optional service call, returning None for explicit degradation.

    Broad exception handling is intentional only at this external boundary.
    Do not log exception messages: provider responses can contain credentials.
    KeyboardInterrupt and SystemExit are not caught.
    """
    for attempt in range(settings.inference_attempts):
        try:
            return operation()
        except Exception as exc:
            LOGGER.warning(
                "%s unavailable (%s), attempt %d/%d",
                label, type(exc).__name__, attempt + 1, settings.inference_attempts,
            )
            if attempt + 1 < settings.inference_attempts:
                time.sleep(settings.inference_backoff_seconds * 2**attempt)
    return None
