from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from makermodslab_sdk._transport import Transport


class Resource:
    """Base for the namespace clients (one per API tag); holds the shared Transport."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport


class SdkModel(BaseModel):
    """Base for all SDK response models: never reject server additions.

    ``extra="allow"`` everywhere — an older SDK against a newer server must
    keep working, and the extra keys stay readable on the object.
    """

    model_config = ConfigDict(extra="allow")


class RecordList(SdkModel):
    """A response envelope that iterates as its RECORDS, not as its fields.

    A pydantic model's default ``__iter__`` yields ``(field_name, value)``
    pairs, so ``for job in client.jobs.list()`` silently hands back tuples and
    fails later with ``'tuple' object has no attribute 'id'``. The envelope is
    still the right shape when the server sends one (it may carry trust or
    paging fields beside the records), so subclasses name their record field
    and get list-like iteration, ``len()`` and indexing over it.

    The envelope's own fields stay reachable by name, and ``model_dump()`` is
    unaffected — pydantic serializes through its own machinery, not this.
    """

    RECORDS_FIELD: ClassVar[str]

    def _records(self) -> list[Any]:
        return getattr(self, self.RECORDS_FIELD)

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        return iter(self._records())

    def __len__(self) -> int:
        return len(self._records())

    def __getitem__(self, index: int) -> Any:
        return self._records()[index]
