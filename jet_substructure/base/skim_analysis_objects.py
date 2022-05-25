""" Analysis objects for analyzed skimmed trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

from typing import Dict, Sequence

import attr
from mammoth.framework.analysis.objects import ScaleFactor  # noqa: F401


def cross_check_task_branch_name_shim(grooming_method: str, input_branches: Sequence[str]) -> Dict[str, str]:
    """Map existing cross check task branch names to standardized names.

    Args:
        grooming_method: Grooming method stored in the cross check task.
        input_branches: Names of existing branches in the cross check task.
    Returns:
        Mapping from standardized branch names to existing branch names in the cross check task.
    """
    # Validation
    input_branches = list(input_branches)

    renames = {}
    # First, some specifics:
    for subjet_name in ["leading", "subleading"]:
        renames[
            f"{grooming_method}_det_level_{subjet_name}_subjet_momentum_fraction_in_hybrid_jet"
        ] = f"{grooming_method}_hybrid_det_level_matching_{subjet_name}_pt_fraction_in_hybrid_jet"

    for branch_name in input_branches:
        new_branch_name = branch_name
        # data -> hybrid
        # matched -> true
        # det_level -> det_level
        for old, new in [("data", "hybrid"), ("matched", "true"), ("det_level", "det_level")]:
            new_branch_name = new_branch_name.replace(old, new)

        if new_branch_name != branch_name and "subjet_momentum_fraction" not in new_branch_name:
            renames[new_branch_name] = branch_name

    return renames


@attr.s
class ResponseType:
    measured_like: str = attr.ib()
    generator_like: str = attr.ib()

    def __str__(self) -> str:
        return f"{self.measured_like}_{self.generator_like}"
