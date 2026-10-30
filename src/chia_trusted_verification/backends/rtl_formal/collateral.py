from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .intents import FormalIntentSpec


BEGIN_MARKER = "// CHIA-BEGIN-AGENT-COLLATERAL"
END_MARKER = "// CHIA-END-AGENT-COLLATERAL"


@dataclass(frozen=True)
class CollateralAnalysis:
    source: str
    assert_count: int
    assume_count: int
    cover_count: int
    referenced_required_signals: tuple[str, ...]
    missing_required_signals: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "assert_count": self.assert_count,
            "assume_count": self.assume_count,
            "cover_count": self.cover_count,
            "referenced_required_signals": list(self.referenced_required_signals),
            "missing_required_signals": list(self.missing_required_signals),
        }


class CollateralValidationError(ValueError):
    pass


def extract_collateral(candidate_source: str) -> str:
    text = str(candidate_source).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            lines = lines[1:-1]
            text = "\n".join(lines).strip()
    if BEGIN_MARKER in text or END_MARKER in text:
        if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
            raise CollateralValidationError(
                "candidate must contain exactly one CHIA collateral marker pair"
            )
        begin = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
        end = text.index(END_MARKER)
        if end <= begin:
            raise CollateralValidationError("CHIA collateral markers are out of order")
        text = text[begin:end].strip()
    if not text:
        raise CollateralValidationError("agent collateral is empty")
    return text + "\n"


def analyze_collateral(
    candidate_source: str,
    spec: FormalIntentSpec,
    *,
    strict_required_signals: bool,
) -> CollateralAnalysis:
    source = extract_collateral(candidate_source)
    scrubbed = _strip_comments_and_strings(source)
    lowered = scrubbed.lower()

    forbidden = {
        "module": r"\bmodule\b",
        "endmodule": r"\bendmodule\b",
        "bind": r"\bbind\b",
        "preprocessor directive": r"`[A-Za-z_]",
        "initial": r"\binitial\b",
        "continuous assignment": r"\bassign\b",
        "force/release": r"\b(?:force|release)\b",
        "restrict statement": r"\brestrict\b",
        "system command": r"\$system\b",
        "simulation termination": r"\$(?:finish|fatal|error|stop)\b",
        "file read": r"\$readmem[bh]\b",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, scrubbed, flags=re.IGNORECASE):
            raise CollateralValidationError(
                f"agent collateral may not contain {label}; it may only add formal properties"
            )

    assignment = re.search(
        r"(?m)(?:^|\bbegin\b|;)\s*[A-Za-z_$][A-Za-z0-9_$]*(?:\s*\[[^\]\n]+\])?\s*(?:<=|=(?!=))",
        scrubbed,
    )
    if assignment is not None:
        raise CollateralValidationError(
            "agent collateral may not assign or drive RTL signals; it may only observe them in formal properties"
        )

    assert_count = len(re.findall(r"\bassert\s*(?:property\s*)?\(", lowered))
    assume_count = len(re.findall(r"(?:\bassume|\$assume)\s*(?:property\s*)?\(", lowered))
    cover_count = len(re.findall(r"\bcover\s*(?:property\s*)?\(", lowered))

    if "assert" in spec.required_property_kinds and assert_count < 1:
        raise CollateralValidationError("candidate must contain at least one assertion")
    if "cover" in spec.required_property_kinds and cover_count < 1:
        raise CollateralValidationError(
            "candidate must contain at least one cover property for non-vacuity/reachability"
        )
    if assume_count and not spec.allow_agent_assumptions:
        raise CollateralValidationError(
            "agent-generated assumptions are disabled for this intent. Assumptions belong to the trusted harness"
        )

    referenced = tuple(signal for signal in spec.required_signals if _contains_identifier(scrubbed, signal))
    missing = tuple(signal for signal in spec.required_signals if signal not in referenced)
    if strict_required_signals and missing:
        raise CollateralValidationError(
            "candidate does not reference required signal(s): " + ", ".join(missing)
        )

    return CollateralAnalysis(
        source=source,
        assert_count=assert_count,
        assume_count=assume_count,
        cover_count=cover_count,
        referenced_required_signals=referenced,
        missing_required_signals=missing,
    )


def render_seed_candidate(spec: FormalIntentSpec) -> str:
    hints = ", ".join(spec.signal_hints[:8]) or "inspect the configured target RTL"
    return (
        f"{BEGIN_MARKER}\n"
        f"// Intent: {spec.intent_id}\n"
        f"// Requirement: {spec.requirement}\n"
        f"// Signal hints: {hints}\n"
        "// Replace this deliberately failing seed with a meaningful assertion and cover.\n"
        "always @(posedge clock) begin\n"
        "  if (!reset) begin\n"
        "    assert(1'b0);\n"
        "    cover(1'b1);\n"
        "  end\n"
        "end\n"
        f"{END_MARKER}\n"
    )


def _contains_identifier(text: str, name: str) -> bool:
    if not name:
        return False
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name):
        return re.search(rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])", text) is not None
    return name in text


def _strip_comments_and_strings(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return text
