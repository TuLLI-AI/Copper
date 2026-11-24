from abc import ABC

__all__ = ["UtilBase"]


class UtilBase(ABC):
    """Common abstract base class for cot_prog utility classes.

    Subclasses can override ``from_config`` to customise construction. The
    default implementation tries the most permissive option in order:

      1. ``cls(config, **kwargs)``  if a non-None ``config`` is passed
      2. ``cls(**kwargs)``           if any kwargs are passed
      3. ``cls()``                  no-argument construction

    This lets parameterless registry entries (e.g. the ``Prob``
    semiring solver) work without each one having to provide a stub
    ``from_config`` classmethod.
    """

    @classmethod
    def from_config(cls, config=None, **kwargs):
        # Try positional config first (matches the original docstring
        # convention `from_config(cls, config) -> cls(config)`).
        if config is not None:
            try:
                return cls(config, **kwargs) if kwargs else cls(config)
            except TypeError:
                pass
        # Fall back to kwargs-only init.
        if kwargs:
            try:
                return cls(**kwargs)
            except TypeError:
                pass
        # Last resort: no-arg construction (most semiring solvers).
        return cls()
