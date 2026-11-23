"""Access to third-party packages supplied by optional extras.

Each call site names the extra that provides the package, so a missing
dependency produces an actionable ``install chessnets[<extra>]`` error at the
point of use. ChessNets' own modules are imported normally.

Use :func:`optional_module` for a lazy module-level handle (the import happens
on first attribute access) and :func:`import_optional` when the package is
needed immediately, for example to define classes at import time.
"""

from __future__ import annotations

import importlib
import types
import typing as tp


def import_optional(module_name: str, extra: str | None) -> types.ModuleType:
    """Import ``module_name`` or explain how to install it.

    ``extra`` names the ChessNets extra that provides the package. Pass ``None``
    for rarely needed packages that no extra provides; the error then asks for
    the package itself.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only translate the error for the requested package itself; a missing
        # transitive dependency of an installed package should surface as-is.
        if exc.name is not None and (
            module_name == exc.name or module_name.startswith(f"{exc.name}.")
        ):
            requirement = (
                f"chessnets[{extra}]" if extra else module_name.split(".", 1)[0]
            )
            raise ImportError(
                f"{module_name} is required for this feature; install {requirement}."
            ) from exc
        raise


class OptionalModule:
    """Module proxy that imports its target on first attribute access."""

    def __init__(self, module_name: str, extra: str | None):
        self.module_name = module_name
        self.extra = extra
        self._module: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._module is None:
            self._module = import_optional(self.module_name, self.extra)
        return self._module

    def __getattr__(self, name: str) -> tp.Any:
        return getattr(self._load(), name)


def optional_module(module_name: str, extra: str | None) -> OptionalModule:
    return OptionalModule(module_name, extra)


__all__ = ["OptionalModule", "import_optional", "optional_module"]
