from typing import Any


def is_typed_dict(obj: Any) -> bool:
    if not isinstance(obj, type):
        return False

    return all(
        hasattr(obj, attr)
        for attr in ("__required_keys__", "__optional_keys__", "__optional_keys__")
    )
