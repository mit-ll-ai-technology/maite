try:
    import augly  # noqa: F401
    import augly.image  # noqa: F401
except ImportError:
    raise ImportError("Augly is not installed for `jatic_toolbox.interop.augly`")
