"""The operation registry glue for the coverage ratchet."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def operation(operation_id: str) -> Callable[[F], F]:
    """Tag a resource method with the v1 ``operationId`` it covers.

    tests/test_coverage_ratchet.py walks these tags against the committed
    OpenAPI snapshot, so every tagged v1 route is either implemented or
    explicitly planned — never silently missing. Method NAMES stay free to be
    ergonomic; the decorator is the source of truth for coverage.
    """

    def decorate(fn: F) -> F:
        fn._operation_id = operation_id  # type: ignore[attr-defined]
        return fn

    return decorate
