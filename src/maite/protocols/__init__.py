from typing import Protocol, Hashable, runtime_checkable


# define minimal DatumMetadata protocol class
@runtime_checkable
class DatumMetadata(Protocol):
    @property
    def uuid(self) -> Hashable:
        ...