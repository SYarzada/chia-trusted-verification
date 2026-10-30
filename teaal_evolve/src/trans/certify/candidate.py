

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ir import (
    ElaboratedArchitecture,
    ElaboratedBinding,
    ElaboratedEinsum,
    ElaboratedFormat,
    ElaboratedMapping,
)
from src.parse import (
    ArchitectureParser,
    BaseParser,
    BindingParser,
    EinsumParser,
    FormatParser,
    MappingParser,
    OptionsParser,
)


@dataclass
class ParserBundle:
    einsum: EinsumParser
    mapping: MappingParser
    format: FormatParser
    binding: BindingParser
    architecture: ArchitectureParser
    options: OptionsParser


@dataclass
class CandidateArtifacts:
    parsers: ParserBundle
    elaborated_einsum: Optional[ElaboratedEinsum] = None
    elaborated_mapping: Optional[ElaboratedMapping] = None
    elaborated_format: Optional[ElaboratedFormat] = None
    elaborated_architecture: Optional[ElaboratedArchitecture] = None
    elaborated_binding: Optional[ElaboratedBinding] = None


@dataclass
class Candidate:
    

    spec_dict: Dict[str, Any]
    label: str = "candidate"
    source_files: List[str] = field(default_factory=list)
    source_strings: List[str] = field(default_factory=list)
    knobs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        files: List[str],
        strings: Optional[List[str]] = None,
        label: str = "candidate",
        knobs: Optional[Dict[str, Any]] = None,
    ) -> "Candidate":
        merged = "\n".join(strings or []) + "\n"
        for file_name in files:
            with open(file_name, "r", encoding="utf-8") as handle:
                merged += handle.read() + "\n"

        spec_dict = BaseParser.parse_str(merged)

        if label == "candidate" and files:
            label = Path(files[0]).stem

        return cls(
            spec_dict=spec_dict,
            label=label,
            source_files=list(files),
            source_strings=list(strings or []),
            knobs=dict(knobs or {}),
        )

    @classmethod
    def from_dict(
        cls,
        spec_dict: Dict[str, Any],
        label: str = "candidate",
        knobs: Optional[Dict[str, Any]] = None,
    ) -> "Candidate":
        return cls(
            spec_dict=copy.deepcopy(spec_dict),
            label=label,
            knobs=dict(knobs or {}),
        )

    def clone(self, label: Optional[str] = None) -> "Candidate":
        return Candidate.from_dict(
            spec_dict=self.spec_dict,
            label=label or self.label,
            knobs=self.knobs,
        )

    def build_parsers(self) -> ParserBundle:
        spec = copy.deepcopy(self.spec_dict)
        return ParserBundle(
            einsum=EinsumParser.from_dict(spec),
            mapping=MappingParser.from_dict(spec),
            format=FormatParser.from_dict(spec),
            binding=BindingParser.from_dict(spec),
            architecture=ArchitectureParser.from_dict(spec),
            options=OptionsParser.from_dict(spec),
        )

    def empty_artifacts(self) -> CandidateArtifacts:
        return CandidateArtifacts(parsers=self.build_parsers())