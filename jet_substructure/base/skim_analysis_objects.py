""" Analysis objects for analyzed skimmed trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Type

import attr
from pachyderm import binned_data


@attr.s
class ScaleFactor:
    # float cast to ensure that we get a standard float instead of an np.float
    cross_section: float = attr.ib(converter=float)
    n_trials: int = attr.ib(converter=int)
    n_entries: int = attr.ib(converter=int)
    n_accepted_events: int = attr.ib(converter=int)

    def value(self) -> float:
        """Value of the scale factor.

        NOTE:
            Leticia's integral method (copied below) is the same as above if we didn't scale by n_entries.
            However, I've historically scaled by n_entries, and will continue to do so here.
            `scaleFactor = hcross->Integral(ptHardBin, ptHardBin) / htrials->Integral(ptHardBin, ptHardBin);`.

        NOTE:
            It is the user's responsibility to utilize the number of accepted events to normalize the
            scale factors.

        Args:
            None.
        Returns:
            Scale factor calculated based on the extracted values.
        """
        return self.cross_section * self.n_entries / self.n_trials

    @classmethod
    def from_hists(
        cls: Type["ScaleFactor"], n_accepted_events: int, n_entries: int, cross_section: Any, n_trials: Any
    ) -> "ScaleFactor":
        # Validation
        # (and for convenience)
        h_cross_section = binned_data.BinnedData.from_existing_data(cross_section)
        h_n_trials = binned_data.BinnedData.from_existing_data(n_trials)

        # Find the first non-zero values bin.
        # argmax will return the index of the first instance of True.
        # NOTE: This isn't the true value of the pt hard bin because of indexing from 0.
        pt_hard_bin = (h_cross_section.values != 0).argmax(axis=0)

        return cls(
            cross_section=h_cross_section.values[pt_hard_bin],
            n_trials=h_n_trials.values[pt_hard_bin],
            n_entries=n_entries,
            n_accepted_events=n_accepted_events,
        )


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
