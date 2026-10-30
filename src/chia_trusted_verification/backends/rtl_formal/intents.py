from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class FormalIntentSpec:
    intent_id: str
    category: str
    title: str
    requirement: str
    target_module: str
    source_file: str | None = None
    source_anchor: str | None = None
    signal_hints: tuple[str, ...] = ()
    required_signals: tuple[str, ...] = ()
    required_property_kinds: tuple[str, ...] = ("assert", "cover")
    allow_agent_assumptions: bool = False
    proof_depth: int = 20
    rationale: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalIntentSpec":
        required_property_kinds = tuple(
            str(item).strip().lower() for item in value.get("required_property_kinds", ["assert", "cover"])
        )
        unknown = sorted(set(required_property_kinds) - {"assert", "cover"})
        if unknown:
            raise ValueError(f"unsupported formal property kind(s): {', '.join(unknown)}")
        return cls(
            intent_id=str(value["intent_id"]),
            category=str(value.get("category", "uncategorized")),
            title=str(value.get("title", value["intent_id"])),
            requirement=str(value.get("requirement", "")),
            target_module=str(value.get("target_module", "")),
            source_file=(None if value.get("source_file") is None else str(value["source_file"])),
            source_anchor=(None if value.get("source_anchor") is None else str(value["source_anchor"])),
            signal_hints=tuple(str(item) for item in value.get("signal_hints", [])),
            required_signals=tuple(str(item) for item in value.get("required_signals", [])),
            required_property_kinds=required_property_kinds,
            allow_agent_assumptions=bool(value.get("allow_agent_assumptions", False)),
            proof_depth=int(value.get("proof_depth", 20)),
            rationale=str(value.get("rationale", "")),
            notes=str(value.get("notes", "")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "category": self.category,
            "title": self.title,
            "requirement": self.requirement,
            "target_module": self.target_module,
            "source_file": self.source_file,
            "source_anchor": self.source_anchor,
            "signal_hints": list(self.signal_hints),
            "required_signals": list(self.required_signals),
            "required_property_kinds": list(self.required_property_kinds),
            "allow_agent_assumptions": self.allow_agent_assumptions,
            "proof_depth": self.proof_depth,
            "rationale": self.rationale,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


def load_intent_catalog(path: str | Path) -> tuple[dict[str, Any], dict[str, FormalIntentSpec]]:
    catalog_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("RTL-formal intent catalog must be a YAML mapping")
    rows = raw.get("intents", [])
    if not isinstance(rows, list):
        raise TypeError("RTL-formal intent catalog field 'intents' must be a list")
    specs: dict[str, FormalIntentSpec] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each RTL-formal intent must be a mapping")
        spec = FormalIntentSpec.from_dict(row)
        if spec.intent_id in specs:
            raise ValueError(f"duplicate RTL-formal intent_id {spec.intent_id!r}")
        if not spec.target_module:
            raise ValueError(f"intent {spec.intent_id!r} has no target_module")
        if not spec.requirement:
            raise ValueError(f"intent {spec.intent_id!r} has no requirement")
        specs[spec.intent_id] = spec
    metadata = {str(k): v for k, v in raw.items() if k != "intents"}
    return metadata, specs


def get_intent_spec(path: str | Path, intent_id: str) -> FormalIntentSpec:
    _, specs = load_intent_catalog(path)
    try:
        return specs[str(intent_id)]
    except KeyError as exc:
        preview = ", ".join(sorted(specs)[:12])
        suffix = "..." if len(specs) > 12 else ""
        raise ValueError(
            f"unknown RTL-formal intent {intent_id!r}; available: {preview}{suffix}"
        ) from exc
