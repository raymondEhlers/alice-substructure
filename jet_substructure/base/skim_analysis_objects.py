""" Analysis objects for analyzed skimmed trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

from typing import Any, Type

import attr
from pachyderm import binned_data


@attr.s
class ScaleFactor:
    # float cast to ensure that we get a standard float instead of an np.float
    cross_section: float = attr.ib(converter=float)
    n_trials: float = attr.ib(converter=float)
    n_entries: float = attr.ib(converter=float)

    def value(self) -> float:
        """Value of the scale factor.

        NOTE:
            Leticia's integral method (copied below) is the same as above if we didn't scale by n_entries.
            However, I've historically scaled by n_entries, and will continue to do so here.
            `scaleFactor = hcross->Integral(ptHardBin, ptHardBin) / htrials->Integral(ptHardBin, ptHardBin);`.

        Args:
            None.
        Returns:
            Scale factor calculated based on the extracted values.
        """
        return self.cross_section * self.n_entries / self.n_trials

    @classmethod
    def from_hists(cls: Type["ScaleFactor"], n_entries: int, cross_section: Any, n_trials: Any) -> "ScaleFactor":
        # Validation
        # (and for convenience)
        h_cross_section = binned_data.BinnedData.from_existing_data(cross_section)
        h_n_trials = binned_data.BinnedData.from_existing_data(n_trials)

        # Find the first non-zero values bin.
        # argmax will return the index of the first instance of True.
        pt_hard_bin = (h_cross_section.values != 0).argmax(axis=0)

        return cls(
            cross_section=h_cross_section.values[pt_hard_bin],
            n_trials=h_n_trials.values[pt_hard_bin],
            n_entries=n_entries,
        )


@attr.s
class ResponseType:
    measured_like: str = attr.ib()
    generator_like: str = attr.ib()

    def __str__(self) -> str:
        return f"{self.measured_like}_{self.generator_like}"
