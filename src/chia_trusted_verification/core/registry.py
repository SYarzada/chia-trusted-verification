from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Callable

from .backend import TrustedVerificationBackend


BackendFactory = Callable[[], TrustedVerificationBackend]
_BUILTINS = {
    "teaal_evolve": "chia_trusted_verification.backends.teaal_evolve.backend:TeaalEvolveBackend",
    "rtl_formal": "chia_trusted_verification.backends.rtl_formal.backend:RtlFormalBackend",
}
_REGISTERED: dict[str, BackendFactory] = {}


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("backend name must be non-empty")
    if key in _REGISTERED and not replace:
        raise ValueError(f"backend {key!r} is already registered")
    _REGISTERED[key] = factory


def _load_symbol(spec: str) -> BackendFactory:
    module_name, symbol_name = spec.split(":", 1)
    symbol = getattr(import_module(module_name), symbol_name)
    return symbol


def _external_factories() -> dict[str, BackendFactory]:
    discovered: dict[str, BackendFactory] = {}
    try:
        selected = entry_points(group="chia_trusted_verification.backends")
    except TypeError:
        selected = entry_points().get("chia_trusted_verification.backends", [])
    for item in selected:
        discovered[str(item.name)] = item.load()
    return discovered


def backend_names() -> tuple[str, ...]:
    names = set(_BUILTINS) | set(_REGISTERED) | set(_external_factories())
    return tuple(sorted(names))


def get_backend(name: str) -> TrustedVerificationBackend:
    key = str(name)
    if key in _REGISTERED:
        backend = _REGISTERED[key]()
    elif key in _BUILTINS:
        backend = _load_symbol(_BUILTINS[key])()
    else:
        external = _external_factories()
        if key not in external:
            raise ValueError(
                f"unknown trusted verification backend {key!r}; available: {backend_names()}"
            )
        backend = external[key]()
    if not isinstance(backend, TrustedVerificationBackend):
        raise TypeError(f"backend factory for {key!r} did not return TrustedVerificationBackend")
    if backend.name != key:
        raise ValueError(
            f"backend registry name {key!r} does not match backend.name {backend.name!r}"
        )
    return backend
