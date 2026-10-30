

from __future__ import annotations

from typing import Any, Optional, Tuple


def component_name(raw_component: Any) -> Optional[str]:
    if isinstance(raw_component, str):
        return raw_component
    if isinstance(raw_component, dict):
        name = raw_component.get("name")
        if name is not None:
            return str(name)
    return None


def component_names(raw_components: Any) -> Tuple[str, ...]:
    if not isinstance(raw_components, list):
        return tuple()
    names = []
    for item in raw_components:
        name = component_name(item)
        if name is not None:
            names.append(name)
    return tuple(names)


def set_component_name(binding_entry: dict[str, Any], value: str) -> None:
    component = binding_entry.get("component")
    if isinstance(component, dict):
        component["name"] = value
    else:
        binding_entry["component"] = value
