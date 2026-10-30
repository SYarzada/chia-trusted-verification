

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .candidate import Candidate
from .knobs import KNOB_SPACE_RESTRICTED, extract_knob_map


@dataclass(frozen=True)
class ChangedKnob:
    

    knob_id: str
    before_value: str
    after_value: str
    detail: str


def diff_candidate_knobs(
    before: Candidate,
    after: Candidate,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> List[ChangedKnob]:
    

    before_map = extract_knob_map(before, knob_space=knob_space)
    after_map = extract_knob_map(after, knob_space=knob_space)

    all_ids = sorted(set(before_map.keys()) | set(after_map.keys()))
    changes: List[ChangedKnob] = []

    for knob_id in all_ids:
        before_knob = before_map.get(knob_id)
        after_knob = after_map.get(knob_id)

        before_value = before_knob.current_value if before_knob is not None else "<missing>"
        after_value = after_knob.current_value if after_knob is not None else "<missing>"

        if before_value != after_value:
            detail = (
                before_knob.detail
                if before_knob is not None
                else after_knob.detail
                if after_knob is not None
                else "Knob changed"
            )

            changes.append(
                ChangedKnob(
                    knob_id=knob_id,
                    before_value=before_value,
                    after_value=after_value,
                    detail=detail,
                )
            )

    return changes