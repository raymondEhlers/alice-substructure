"""Unfolding configuration

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import attrs
import boost_histogram as bh
import numpy as np
import uproot
from pachyderm import binned_data

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def efficiency_substructure_variable(
    hists: Mapping[str, binned_data.BinnedData], true_jet_pt_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Efficiency for the substructure variable.

    Note:
        Since we need a set of hists, we just pass all of them.

    Args:
        hists: Input histograms.
        true_jet_pt_range: True jet pt range over which we will integrate.
    Returns:
        Efficiency hist for the substructure variable.
    """
    # Assign them for convenience
    try:
        bh_cut_efficiency = hists["true"].to_boost_histogram()
        bh_full_efficiency = hists["truef"].to_boost_histogram()

        # Select true pt range.
        selection = slice(bh.loc(true_jet_pt_range.min), bh.loc(true_jet_pt_range.max), bh.sum)
        cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[:, selection])
        full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[:, selection])

        return cut / full

    except KeyError:
        logger.warning(
            'Hist "true" was not found. Instead, trying to extract the efficiency directly from the projection.'
        )
        # This hist already has the efficiency applied, so we can return it directly!
        return binned_data.BinnedData.from_existing_data(
            hists[f"correff{int(true_jet_pt_range.min)}-{int(true_jet_pt_range.max)}"]
        )


def _project_substructure_variable(
    input_hist: binned_data.BinnedData, jet_pt_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Project the hist to the substructure variable.

    Args:
        input_hist: Hist to be projected.
        jet_pt_range: True jet pt range over which we will integrate.
    Returns:
        The input hist projected onto the substructure variable axis.
    """
    # For convenience
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(jet_pt_range.min), bh.loc(jet_pt_range.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[:, selection])


def efficiency_pt(
    hists: Mapping[str, binned_data.BinnedData], true_substructure_variable_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Efficiency for the jet pt.

    Note:
        Since we need a set of hists, we just pass all of them.

    Args:
        hists: Input histograms.
        true_substructure_variable_range: True substructure variable range over which we will integrate.
    Returns:
        Efficiency hist for jet pt.
    """
    # For convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    selection = slice(
        bh.loc(true_substructure_variable_range.min), bh.loc(true_substructure_variable_range.max), bh.sum
    )
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[selection, :])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[selection, :])

    return cut / full


def _project_jet_pt(
    input_hist: binned_data.BinnedData, substructure_variable_bin: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Project the hist to the jet pt.

    Args:
        input_hist: Hist to be projected.
        substructure_variable_range: True substructure variable range over which we will integrate.
    Returns:
        The input hist projected onto the the jet pt axis.
    """
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(substructure_variable_bin.min), bh.loc(substructure_variable_bin.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[selection, :])


def _normalize_unfolded(hist: binned_data.BinnedData, efficiency: binned_data.BinnedData) -> binned_data.BinnedData:
    """Normalized unfolded hist.

    This involves applying the efficiency and then normalizing by the integral and the bin width.

    Args:
        hist: Histogram to be normalized.
        efficiency: Efficiency histogram with the same binning as the input hist.
    Returns:
        The normalized histogram.
    """
    # Apply the efficiency.
    hist /= efficiency
    # Then normalize by the integral (sum) and bin width.
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def _normalize_refolded(hist: binned_data.BinnedData) -> binned_data.BinnedData:
    """Normalize refolded hist.

    This involves normalizing by the integral and the bin width.

    Args:
        hist: Histogram to be normalized.
    Returns:
        The normalized histogram.
    """
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def _smeared(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    smeared_range_to_integrate_over: helpers.RangeSelector,
) -> binned_data.BinnedData:
    """Helper function to get a smeared hist along a desired axis.

    Args:
        hists: Input hists.
        hist_name: Name of the smeared histogram to retrieve.
        projection_func: Function to project the histogram along the desired axis.
        smeared_range_to_integrate_over: Smeared range over which we will integrate.
    Returns:
        The desired smeared histogram.
    """
    hist = projection_func(hists[hist_name], smeared_range_to_integrate_over)
    return _normalize_refolded(hist=hist)


def _unfolded(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_range_to_integrate_over: helpers.RangeSelector,
) -> binned_data.BinnedData:
    """Helper function to get an unfolded hist along a desired axis.

    Args:
        hists: Input hists.
        hist_name: Name of the unfolded histogram to retrieve.
        projection_func: Function to project the histogram along the desired axis.
        true_range_to_integrate_over: True range over which we will integrate.
    Returns:
        The desired unfolded histogram.
    """
    # efficiency = efficiency_func(hists, true_bin)
    ## For convenience in normalizing.
    # _normalize_hist = functools.partial(_normalize_unfolded, efficiency=efficiency)
    hist = projection_func(hists[hist_name], true_range_to_integrate_over)
    # hist = _normalize_hist(hist)
    efficiency = efficiency_func(hists, true_range_to_integrate_over)
    return _normalize_unfolded(hist=hist, efficiency=efficiency)


@attrs.define
class UnfoldingOutput:
    substructure_variable: str = attrs.field()
    grooming_method: str = attrs.field()
    smeared_var_range: helpers.RangeSelector = attrs.field()
    smeared_untagged_var: helpers.RangeSelector = attrs.field()
    smeared_jet_pt_range: helpers.JetPtRange = attrs.field()
    collision_system: str = attrs.field()
    base_dir: Path = attrs.field(converter=Path)
    _input_dir_tag: str = attrs.field(default="")
    _output_dir_tag: str | None = attrs.field(default=None)
    pure_matches: bool = attrs.field(default=False)
    suffix: str = attrs.field(default="")
    label: str = attrs.field(default="")
    double_counting_cut: str = attrs.field(default="")
    n_iter_compare: int = attrs.field(default=4)
    raw_hist_name: str = attrs.field(default="raw")
    smeared_hist_name: str = attrs.field(default="smeared")
    true_hist_name: str = attrs.field(default="true")
    _max_n_iter: int | None = attrs.field(default=None)
    hists: MutableMapping[str, binned_data.BinnedData] = attrs.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        # Possibility to to separate input and output dir tags...
        # By default, they can agree, but I need to let myself change it
        if self._output_dir_tag is None:
            self._output_dir_tag = self._input_dir_tag

        self.base_dir = self.base_dir / self.collision_system / "unfolding"

        # Initialize the file if the histograms aren't specified.
        if not self.hists:
            #logger.info(f"{self.input_filename=}")
            f = uproot.open(self.input_filename)
            for k in f.keys(cycle=False):
                self.hists[k] = binned_data.BinnedData.from_existing_data(f[k])

    @property
    def identifier(self) -> str:
        name = f"{self.substructure_variable}_grooming_method_{self.grooming_method}"
        name += f"_smeared_{self.smeared_var_range}"
        name += f"_untagged_{self.smeared_untagged_var}"
        name += f"_smeared_{self.smeared_jet_pt_range}"
        if self.double_counting_cut and self.double_counting_cut != "disabled":
            name += f"__DCC_{self.double_counting_cut}_"
        if self.suffix:
            name += f"_{self.suffix}"
        if self.pure_matches:
            name += "_pure_matches"
        if self.label:
            name += f"_{self.label}"
        return name

    @property
    def max_n_iter(self) -> int:
        if hasattr(self, "_max_n_iter") and self._max_n_iter is not None:
            return self._max_n_iter
        else:
            n = 1
            for hist_name in self.hists:
                # We could equally use the unfolded.
                if "bayesian_folded_iter_" in hist_name:
                    # We add a +1 so we can use it easily with range(...).
                    n = max(n, int(hist_name.split("_")[-1]) + 1)
            self._max_n_iter: int = n
        return self._max_n_iter

    def n_iter_range_to_plot(self) -> Iterable[int]:
        """Generate the n_iter range to plot.

        This lets us cut down on the iterations to plot when it would be too much to view comfortably.
        First, we return the first n, and then take every second from there.
        """
        change_to_sparse = 8
        if self.max_n_iter > change_to_sparse:
            return itertools.chain(range(1, change_to_sparse + 1), range(change_to_sparse + 1, self.max_n_iter, 2))
        return range(1, self.max_n_iter)

    @property
    def input_filename(self) -> Path:
        return self.base_dir / self._input_dir_tag / f"unfolding_{self.identifier}.root"

    @property
    def output_dir(self) -> Path:
        # Help out mypy. This must be true because of the post_int
        assert self._output_dir_tag is not None

        p = self.base_dir / self._output_dir_tag / self.substructure_variable / self.identifier
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir_png(self) -> Path:
        return self.output_dir / "png"

    @property
    def disabled_untagged_bin(self) -> bool:
        """If the untagged bin min and max are the same, the untagged bin was disabled."""
        return self.smeared_untagged_var.min == self.smeared_untagged_var.max

    def unfolded_substructure(self, n_iter: int, true_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Helper to retrieve the unfolded substructure directly """
        return self.true_substructure(
            hist_name=f"bayesian_unfolded_iter_{n_iter}",
            true_jet_pt_range=true_jet_pt_range,
        )

    def true_substructure(self, hist_name: str, true_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Retrieve a true level substructure hist. """
        return _unfolded(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_substructure_variable,
            efficiency_func=efficiency_substructure_variable,
            true_range_to_integrate_over=true_jet_pt_range,
        )

    def unfolded_jet_pt(
        self, n_iter: int, true_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        return self.true_jet_pt(
            hist_name=f"bayesian_unfolded_iter_{n_iter}",
            true_substructure_variable_range=true_substructure_variable_range,
        )

    def true_jet_pt(
        self, hist_name: str, true_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        return _unfolded(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_jet_pt,
            efficiency_func=efficiency_pt,
            true_range_to_integrate_over=true_substructure_variable_range,
        )

    def refolded_substructure(self, n_iter: int, smeared_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Helper to retrieve the refolded substructure directly. """
        return self.smeared_substructure(
            hist_name=f"bayesian_folded_iter_{n_iter}",
            smeared_jet_pt_range=smeared_jet_pt_range,
        )

    def smeared_substructure(self, hist_name: str, smeared_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Retrieve a smeared substructure hist. """
        return _smeared(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_substructure_variable,
            smeared_range_to_integrate_over=smeared_jet_pt_range,
        )

    def refolded_jet_pt(
        self, n_iter: int, smeared_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        """ Helper to retrieve the refolded jet pt directly. """
        return self.smeared_jet_pt(
            hist_name=f"bayesian_folded_iter_{n_iter}",
            smeared_substructure_variable_range=smeared_substructure_variable_range,
        )

    def smeared_jet_pt(
        self, hist_name: str, smeared_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        """ Retrieve a smeared jet pt hist. """
        return _smeared(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_jet_pt,
            smeared_range_to_integrate_over=smeared_substructure_variable_range,
        )


@attrs.define
class SingleResult:
    """ Container for a single unfolding result. """

    data: binned_data.BinnedData = attrs.field()
    n_iter: int = attrs.field()
    ranges: Sequence[helpers.RangeSelector] = attrs.field(factory=list)


@attrs.define
class ModelDependenceConfiguration:
    nominal: str
    variations: list[str]
    approach_to_combining: str = attrs.field(default="max")
    legacy_production: bool = attrs.field(default=False)

    @property
    def all_models(self) -> list[str]:
        return [self.nominal] + self.variations


@attrs.define
class NonClosureConfiguration:
    """Define how to handle the non-closure."""
    contributors: list[str]
    approach_to_combining: str = attrs.field(default="max")

@attrs.define
class BackgroundSubtractionConfiguration:
    """Define how to handle the background subtraction """
    contributors: list[str]

