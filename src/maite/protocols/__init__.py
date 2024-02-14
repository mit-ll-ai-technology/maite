from typing import Protocol, Hashable


# define minimal DatumMetadata protocol class
class DatumMetadata(Protocol):
    @property
    def uuid(self) -> Hashable:
        ...