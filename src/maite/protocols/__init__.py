from typing import Hashable, Protocol, runtime_checkable


# define minimal DatumMetadata protocol class
@runtime_checkable
class DatumMetadata(Protocol):
    @property
    def uuid(self) -> Hashable:
        ...
