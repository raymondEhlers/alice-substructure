""" 2D unfolding implemented via RooUnfold.

The RooUnfold response is created in cxx (because we can't use ROOT DataFrames directly due
to unresolved issues with template discovery), and then we utilize it for unfolding, closures,
etc, in python. We have to rely on ROOT very strongly here because RooUnfold...

Conventions:
- Separate file for each result (closure or otherwise).
- Same names for the hists in each file.
- Standard has no tag.
- Closures always start with "closure" in the tag.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

import attr
import numpy as np
import uproot
from pachyderm import binned_data

from jet_substructure.base import helpers, skim_analysis_objects


logger = logging.getLogger(__name__)

# Type helpers.
# We could do better, but I would prefer weak (ie. only delayed) dependence on ROOT.
RooUnfoldErrorTreatment = Any
RooUnfoldResponse = Any
TH2D = Any
TMatrixD = Any


def _np_array_converter(value: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    """Convert the given value to a numpy array.

    Normally, we would just use np.array directly as the converter function. However, mypy will complain if
    the converter is untyped. So we add (trivial) typing here.  See: https://github.com/python/mypy/issues/6172.

    Note:
        To change the dtype for the converter, one would need to use `partial`.

    Args:
        value: Value to be converted to a numpy array.
        dtype: Dtype to utilize. Default: np.float64.
    Returns:
        The converted numpy array.
    """
    return np.array(value, dtype=dtype)


@attr.s
class ParameterSettings:
    """Parameter settings

    Args:
        true_bins: True bins.
        smeared_bins: Smeared bins.
    """

    true_bins: np.ndarray = attr.ib(converter=_np_array_converter)
    smeared_bins: np.ndarray = attr.ib(converter=_np_array_converter)


@attr.s
class SubstructureVariableSettings(ParameterSettings):
    """Settings specific to the substructure variable.

    Supports z, Rg, and kt.

    Args:
        name: Name of the substructure variable.
        variable_name: Name of the substructure variable in the tree.
        smeared_range: Smeared binning min and max. Values vary due to the location of the untagged bin.
        untagged_bin: Untagged bin min and max.
    """

    name: str = attr.ib()
    variable_name: str = attr.ib()
    smeared_range: helpers.RangeSelector = attr.ib()
    untagged_bin: helpers.RangeSelector = attr.ib()

    @property
    def untagged_value(self) -> float:
        return (self.untagged_bin.max - self.untagged_bin.min) / 2 + self.untagged_bin.min

    @classmethod
    def from_binning(
        cls: Type["SubstructureVariableSettings"],
        true_bins: np.ndarray,
        smeared_bins: np.ndarray,
        name: str,
        variable_name: str,
        untagged_bin_below_range: bool = True,
    ) -> "SubstructureVariableSettings":
        # Determine the appropriate range class.
        # Either "Kt", "Rg", or "Zg"
        range_class_name = variable_name
        if variable_name != "kt" and "g" not in variable_name:
            range_class_name += "g"
        range_class_name = range_class_name.capitalize()
        range_class_name += "Range"
        range_class: Type[helpers.RangeSelector] = getattr(helpers, range_class_name)

        # Determine the binning
        if untagged_bin_below_range:
            smeared_range = range_class(min=smeared_bins[1], max=smeared_bins[-1])
            untagged_bin = range_class(min=smeared_bins[0], max=smeared_bins[1])
        else:
            smeared_range = range_class(min=smeared_bins[0], max=smeared_bins[-2])
            untagged_bin = range_class(min=smeared_bins[-2], max=smeared_bins[-1])

        return cls(
            true_bins=true_bins,
            smeared_bins=smeared_bins,
            name=name,
            variable_name=variable_name,
            smeared_range=smeared_range,
            untagged_bin=untagged_bin,
        )


@attr.s
class Settings:
    grooming_method: str = attr.ib()
    jet_pt: ParameterSettings = attr.ib()
    substructure_variable: SubstructureVariableSettings = attr.ib()
    suffix: str = attr.ib()
    output_dir: Path = attr.ib()
    label: str = attr.ib(default="")
    use_pure_matches: bool = attr.ib(default=False)
    filename_padding_factor: int = attr.ib(default=0)

    @property
    def output_tag(self) -> str:
        # Start with the basic information
        base_filename = f"unfolding_{self.substructure_variable.name}_grooming_method_{self.grooming_method}"
        # Then add the binning information.
        # First, the substructure edges.
        base_filename += f"_smeared_{self.substructure_variable.smeared_range}"
        # Then the untagged
        base_filename += f"_untagged_{self.substructure_variable.untagged_bin}"
        # Then the jet pt
        smeared_jet_pt = helpers.JetPtRange(min=self.jet_pt.smeared_bins[0], max=self.jet_pt.smeared_bins[-1])
        # base_filename += f"_smeared_{smeared_jet_pt.zero_padded_str(self.filename_padding_factor)}"
        base_filename += f"_smeared_{smeared_jet_pt}"
        base_filename += f"_{self.suffix}"
        # Additional options
        # Optional tag
        if self.label:
            base_filename += f"_{self.label}"
        # Put other possible options after the tag so we can sort by tag if it exists.
        if self.use_pure_matches:
            base_filename += "_pure_matches"
        return base_filename

    @property
    def output_filename(self) -> Path:
        return (self.output_dir / self.output_tag).with_suffix(".root")


@attr.s
class InputFileSettings:
    filenames: Sequence[Path] = attr.ib()
    tree_name: str = attr.ib()


@attr.s
class DataSettings(InputFileSettings):
    prefix: str = attr.ib()


@attr.s
class EmbeddedSettings(InputFileSettings):
    hybrid_prefix: str = attr.ib()
    true_prefix: str = attr.ib()
    det_level_prefix: str = attr.ib()


def _pass_filenames_to_ROOT(filenames: Sequence[Path]) -> List[str]:
    """ Helper to convert Path to str for ROOT. """
    return [str(f) for f in filenames]


def _array_to_ROOT(arr: np.ndarray, type_name: str = "double") -> Any:
    """Convert numpy array to std::vector via ROOT.

    Because it apparently can't handle conversions directly. Which is really dumb...

    In principle, we could convert the numpy dtype into the c++ type, but that's a lot of mapping
    to be done for a function that (hopefully) isn't used so often. So we let the user decide.

    Args:
        arr: Numpy array to be converted.
        type_name: c++ type name to be used for the vector. Default: "double".
    Returns:
        std::vector containing the numpy array values.
    """
    import ROOT

    vector = ROOT.std.vector(type_name)()
    for a in arr:
        vector.push_back(a)
    return vector


def correlation_hist_substructure_var(cov: TMatrixD, name: str, title: str, na: int, nb: int, kbin: int) -> TH2D:
    """Correlation histogram for the substructure variable.

    Varies from the pt by the indexing of the covariance matrix.

    Args:
        cov: Covariance matrix derived from the unfolding.
        name: Name of the covariance matrix.
        title: Title of the covariance matrix.
        na: Number of x bins.
        nb: Number of y bins.
        kbin: Bin in the selected dimension.
    Returns:
        The correlation histogram.
    """
    import ROOT

    h = ROOT.TH2D(name, title, nb, 0, nb, nb, 0, nb)

    for i in range(0, nb):
        for n in range(0, nb):
            index1 = kbin + na * i
            index2 = kbin + na * n
            Vv = cov(index1, index1) * cov(index2, index2)
            if Vv > 0.0:
                h.SetBinContent(i + 1, n + 1, cov(index1, index2) / np.sqrt(Vv))
    return h


def correlation_hist_pt(cov: TMatrixD, name: str, title: str, na: int, nb: int, kbin: int) -> TH2D:
    """Correlation histogram for the jet pt.

    Varies from the substructure variable by the indexing of the covariance matrix.

    Args:
        cov: Covariance matrix derived from the unfolding.
        name: Name of the covariance matrix.
        title: Title of the covariance matrix.
        na: Number of x bins.
        nb: Number of y bins.
        kbin: Bin in the selected dimension.
    Returns:
        The correlation histogram.
    """
    import ROOT

    h = ROOT.TH2D(name, title, na, 0, na, na, 0, na)

    for i in range(0, na):
        for n in range(0, na):
            index1 = i + na * kbin
            index2 = n + na * kbin
            Vv = cov(index1, index1) * cov(index2, index2)
            if Vv > 0.0:
                h.SetBinContent(i + 1, n + 1, cov(index1, index2) / np.sqrt(Vv))
    return h


def unfolding_2D(
    response: RooUnfoldResponse,
    input_spectra: TH2D,
    true_spectra: TH2D,
    error_treatment: Optional[RooUnfoldErrorTreatment] = None,
    tag: str = "",
    max_iter: int = 20,
    n_iter_for_covariance: int = 8,
) -> Dict[str, TH2D]:
    """Perform unfolding in 2D.

    Args:
        response: Response matrix.
        input_spectra: Input histogram.
        true_spectra: True histogram. Just used for binning with the covariance matrices.
        error_treatment: Error treatment to be used for unfolding.
        tag: Tag...
        max_iter: Maximum number of iterations for unfolding. Default: 20.
        n_iter_for_covariance: Number of iterations that should be used for calculating the covariance. Default: 8.
    Returns:
        Unfolded and folded hists per iter, as well as the covariance matrices. See the hist names in the code.
    """
    # Delayed import for convenience.
    import ROOT

    # Validation
    if error_treatment is None:
        error_treatment = ROOT.RooUnfold.ErrorTreatment.kCovariance

    # Setup
    logger.info("=======================================================")
    logger.info(f'Unfolding for tag "{tag}"')
    # Determine the tag. If we have a non-empty tag, we append it to all of the histograms.
    if tag != "":
        tag += "_"
    output_hists = {}

    for n_iter in range(1, max_iter):
        logger.debug(f"Iteration {n_iter}")

        # Setup the response for unfolding.
        unfold = ROOT.RooUnfoldBayes(response, input_spectra, n_iter)
        # And then unfold.
        h_unfold = unfold.Hreco(error_treatment)

        # Refold the truth (ie. fold back).
        h_fold = response.ApplyToTruth(h_unfold, "")

        # Clone unfolded and refolded hists to write to the output file.
        name = f"{tag}bayesian_unfolded_iter_{n_iter}"
        output_hists[name] = h_unfold.Clone(name)
        name = f"{tag}bayesian_folded_iter_{n_iter}"
        output_hists[name] = h_fold.Clone(name)

        # Retrieve the covariance matrix. Only for a selected iteration.
        if n_iter == n_iter_for_covariance:
            covariance_matrix = unfold.Ereco(ROOT.RooUnfold.kCovariance)
            # Substructure variable.
            for k in range(0, true_spectra.GetNbinsX()):
                h_corr = correlation_hist_substructure_var(
                    covariance_matrix,
                    f"{tag}corr{k}",
                    "Covariance matrix",
                    true_spectra.GetNbinsX(),
                    true_spectra.GetNbinsY(),
                    k,
                )
                name = f"{tag}pearsonmatrix_iter{n_iter}_bin_substructure_var{k}"
                cov_substructure_var = h_corr.Clone(name)
                cov_substructure_var.SetDrawOption("colz")
                # Save
                output_hists[name] = cov_substructure_var

            # Jet pt.
            for k in range(0, true_spectra.GetNbinsY()):
                h_corr = correlation_hist_pt(
                    covariance_matrix,
                    f"{tag}corr{k}pt",
                    "Covariance matrix",
                    true_spectra.GetNbinsX(),
                    true_spectra.GetNbinsY(),
                    k,
                )
                name = f"{tag}pearsonmatrix_iter{n_iter}_bin_pt{k}"
                cov_pt = h_corr.Clone(name)
                cov_pt.SetDrawOption("colz")
                # Save
                output_hists[name] = cov_pt

    logger.info("Finished unfolding!")
    logger.info("=======================================================")

    return output_hists


def _setup_unfolding() -> None:
    """ Setup RooUnfold and the additional unfolding code. """
    # Delayed import to avoid direct dependence.
    import ROOT

    # Nominally setup for MT. It's not really going to do us any good here, but it doesn't hurt anything.
    # NOTE: We do need to specify 1 to ensure that we don't use extra cores.
    ROOT.ROOT.EnableImplicitMT(1)
    # Load RooUnfold
    ROOT.gSystem.Load("libRooUnfold")
    # Load the unfolding utilities. We're careful to be (relatively) position independent.
    # This just assumes that this file is in the same directory as the unfolding.cxx file, which should
    # usually be a reasonable assumption.
    unfolding_cxx = Path(__file__).resolve().parent / "unfolding.cxx"
    # We only want to load it if it hasn't been already, so we use the `create_response_2D` function
    # as a proxy for this. Loading it twice appears to cause segfaults in some cases.
    if not hasattr(ROOT, "create_response_2D"):
        # ROOT.gInterpreter.ProcessLine(f"""#include "{str(unfolding_cxx)}" """)
        ROOT.gInterpreter.ProcessLine(f""".L {str(unfolding_cxx)} """)


def _write_hists(hists: Sequence[Dict[str, TH2D]], output_filename: Path, additional_tag: str = "") -> None:
    # Delayed import to avoid direct dependence.
    import ROOT

    # Setup
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    # Add an additional tag if requested
    if additional_tag:
        output_filename = (output_filename.parent / f"{output_filename.stem}_{additional_tag}").with_suffix(".root")

    logger.debug(f"Writing hists to {output_filename}")
    # Uproot3 also works, but it's far less space efficient. Since we already have to import ROOT,
    # we may as well just use it...
    # with uproot3.recreate(settings.output_filename.with_suffix(".uproot.root")) as f:
    #    for k, v in hists.items():
    #        f[k] = binned_data.BinnedData.from_existing_data(v).to_numpy()
    #    for k, v in output_hists.items():
    #        f[k] = binned_data.BinnedData.from_existing_data(v).to_numpy()
    f_out = ROOT.TFile(str(output_filename), "RECREATE")
    f_out.cd()
    for hists_dict in hists:
        for v in hists_dict.values():
            v.Write()

    f_out.Close()


def _default_hists(settings: Settings) -> Dict[str, TH2D]:
    import ROOT

    hists = {}
    # the raw correlation (ie. data)
    hists["h2_raw"] = ROOT.TH2D(
        "raw",
        "raw",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        len(settings.jet_pt.smeared_bins) - 1,
        settings.jet_pt.smeared_bins,
    )
    # detector measure level (ie. hybrid)
    hists["h2_smeared"] = ROOT.TH2D(
        "smeared",
        "smeared",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        len(settings.jet_pt.smeared_bins) - 1,
        settings.jet_pt.smeared_bins,
    )
    # detector measure level no cuts (ie. hybrid, but no cuts).
    hists["h2_smeared_no_cuts"] = ROOT.TH2D(
        "smearednocuts",
        "smearednocuts",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        # NOTE: We're actually going to fill hybrid jet pt. But we want a wider range, so we use true jet pt bins for convenience.
        len(settings.jet_pt.true_bins) - 1,
        settings.jet_pt.true_bins,
    )
    # true correlations with measured cuts
    hists["h2_true"] = ROOT.TH2D(
        "true",
        "true",
        len(settings.substructure_variable.true_bins) - 1,
        settings.substructure_variable.true_bins,
        len(settings.jet_pt.true_bins) - 1,
        settings.jet_pt.true_bins,
    )
    # full true correlation (without cuts)
    hists["h2_full_eff"] = ROOT.TH2D(
        "truef",
        "truef",
        len(settings.substructure_variable.true_bins) - 1,
        settings.substructure_variable.true_bins,
        len(settings.jet_pt.true_bins) - 1,
        settings.jet_pt.true_bins,
    )
    # Correlation between the splitting variables at true and hybrid (with cuts).
    hists["h2_substructure_variable"] = ROOT.TH2D(
        "h2_substructure_variable",
        "h2_substructure_variable",
        len(settings.substructure_variable.smeared_bins) - 1,
        settings.substructure_variable.smeared_bins,
        len(settings.substructure_variable.true_bins) - 1,
        settings.substructure_variable.true_bins,
    )

    # Sumw2 for all hists, store for passing them...
    for k, h in hists.items():
        h.Sumw2()

    return hists


def _hists_to_map_for_ROOT(hists: Dict[str, TH2D]) -> Any:
    # Delayed import to avoid direct dependence.
    import ROOT

    hists_map_for_root = ROOT.std.map("std::string", "TH2D *")()
    for k, h in hists.items():
        # Why not via __setitem__? Because that would be too easy...
        hists_map_for_root.insert((k, h))
        # hists_to_root[k] = ROOT.addressof(h, True)

    return hists_map_for_root


def _branch_name_shim_to_map_for_ROOT(branch_renames: Mapping[str, str]) -> Any:
    # Delayed import to avoid direct dependence.
    import ROOT

    map = ROOT.std.map("std::string", "std::string")()
    for k, h in branch_renames.items():
        # Why not via __setitem__? Because that would be too easy...
        map.insert((k, h))
        # map[k] = ROOT.addressof(h, True)

    return map


def get_reweighted_ratio(
    embedded_dataset_name: str, data_dataset_name: str, grooming_method: str, base_directory: Path = Path("output")
) -> TH2D:
    # Delayed import to avoid direct dependence.
    import ROOT

    # Retrieve embedded hist
    embedded_filename = (
        base_directory
        / "embedPythia"
        / "RDF"
        / f"{embedded_dataset_name}_{grooming_method}_prefixes_hybrid_true_det_level_closure.root"
    )
    f_embedded = ROOT.TFile(str(embedded_filename), "READ")
    h_embedded = f_embedded.Get(f"{grooming_method}_hybrid_kt_jet_pt")
    # Retrieve data hist
    data_filename = (
        base_directory / "PbPb" / "RDF" / f"{data_dataset_name}_{grooming_method}_prefixes_data_closure.root"
    )
    f_PbPb = ROOT.TFile(str(data_filename), "READ")
    h_PbPb = f_PbPb.Get(f"{grooming_method}_data_kt_jet_pt")

    # Calculate the ratio and cleanup
    h_ratio = h_embedded.Clone("h_ratio")
    h_ratio.Divide(h_PbPb)
    h_ratio.SetDirectory(0)

    # Cleanup
    f_embedded.Close()
    f_PbPb.Close()

    return h_ratio


def _get_reweighting_ratio(
    reweight_data_dataset_name: str, reweight_embedded_dataset_name: str, settings: Settings
) -> TH2D:
    # Validation
    if not reweight_data_dataset_name or not reweight_embedded_dataset_name:
        raise ValueError(
            f"Must pass data and embedded dataset names. Passed data: {reweight_data_dataset_name}, embedded: {reweight_embedded_dataset_name}"
        )

    h_reweighting_ratio = get_reweighted_ratio(
        data_dataset_name=reweight_data_dataset_name,
        embedded_dataset_name=reweight_embedded_dataset_name,
        grooming_method=settings.grooming_method,
    )

    # Validate the reweighting ratio
    # x axis should contain the smeared substructure variable
    # y axis contains the smeared jet pt.
    temp_hist = binned_data.BinnedData.from_existing_data(h_reweighting_ratio)
    np.testing.assert_allclose(temp_hist.axes[0].bin_edges, settings.substructure_variable.smeared_bins)
    np.testing.assert_allclose(temp_hist.axes[1].bin_edges, settings.jet_pt.smeared_bins)

    return h_reweighting_ratio


def _create_branch_rename_shim(
    embedded_filenames: Sequence[Path], embedded_tree_name: str, grooming_method: str
) -> Dict[str, str]:
    """Create cross check task branch rename shim.

    Used to standardize the output of the cross check task. However, ROOT makes
    this wayyyyyyy to difficult to do sustainably, so we eventually ended up
    just skimming the cross check task output.

    Args:
        embedded_filenames: Embedded filenames.
        embedded_tree_name: Embedded tree name.
        grooming_method: Name of the grooming method stored in the cross check task.

    Returns:
        Map from standardized branch names to existing branch names. To be used with
        `RDataFrame.Define(...)` or `TTree.Alias(...)`.
    """
    # Need to do a quick read of the branch names. The branch names shouldn't vary by file, so we can
    # use the first one.
    with uproot.open(embedded_filenames[0]) as f:
        input_branches = list(f[embedded_tree_name].keys())

    branch_renames = skim_analysis_objects.cross_check_task_branch_name_shim(
        grooming_method=grooming_method,
        input_branches=input_branches,
    )

    return branch_renames


def run_unfolding(
    settings: Settings,
    data_filenames: Sequence[Path],
    data_tree_name: str,
    embedded_filenames: Sequence[Path],
    embedded_tree_name: str,
    reweight_prior: bool = False,
    reweight_data_dataset_name: str = "",
    reweight_embedded_dataset_name: str = "",
) -> bool:
    # Delayed import to avoid direct dependence.
    import ROOT

    # Setup
    _setup_unfolding()

    # Define hists (and the map to pass them into ROOT for unfolding)
    hists = _default_hists(settings=settings)
    hists_map_for_root = _hists_to_map_for_ROOT(hists=hists)

    h_reweighting_response_ratio = ROOT.nullptr
    if reweight_prior:
        h_reweighting_response_ratio = _get_reweighting_ratio(
            reweight_data_dataset_name=reweight_data_dataset_name,
            reweight_embedded_dataset_name=reweight_embedded_dataset_name,
            settings=settings,
        )

    # Create the responses. We assume some conventions about column names.
    # They should generally be reasonable, but may require tweaks from time to time.
    responses = ROOT.create_response_2D(
        hists_map_for_root,
        settings.grooming_method,
        settings.substructure_variable.variable_name,
        _array_to_ROOT(settings.jet_pt.smeared_bins, "double"),
        _array_to_ROOT(settings.jet_pt.true_bins, "double"),
        _array_to_ROOT(settings.substructure_variable.smeared_bins, "double"),
        _array_to_ROOT(settings.substructure_variable.true_bins, "double"),
        settings.substructure_variable.untagged_value,
        settings.substructure_variable.smeared_range.min,
        settings.substructure_variable.smeared_range.max,
        _array_to_ROOT(_pass_filenames_to_ROOT(data_filenames), "std::string"),
        _array_to_ROOT(_pass_filenames_to_ROOT(embedded_filenames), "std::string"),
        settings.use_pure_matches,
        h_reweighting_response_ratio,
        data_tree_name,
        embedded_tree_name,
    )

    logger.debug(responses)

    # Perform the actual unfolding.
    # First, the standard unfolding.
    output_hists = unfolding_2D(
        response=responses.response, input_spectra=hists["h2_raw"], true_spectra=hists["h2_true"]
    )
    # Write the output before we move onto the next case.
    _write_hists([hists, output_hists], settings.output_filename)

    # Next, the trivial closure test where the input is the smeared hybrid spectra.
    output_hists = unfolding_2D(
        response=responses.response,
        input_spectra=hists["h2_smeared"],
        true_spectra=hists["h2_true"],
    )
    # Write the output before we move onto the next case.
    _write_hists(
        [hists, output_hists], settings.output_filename, additional_tag="closure_trivial_hybrid_smeared_as_input"
    )

    # For instance, closure test 5 should be trivial
    selected_iter_for_closure = 5
    output_hists = unfolding_2D(
        response=responses.response,
        input_spectra=output_hists[f"bayesian_folded_iter_{selected_iter_for_closure}"],
        # The true spectra doesn't matter here...
        true_spectra=hists["h2_true"],
    )
    # Write the output before we move onto the next case.
    _write_hists(
        [hists, output_hists], settings.output_filename, additional_tag=f"closure_5_iter_{selected_iter_for_closure}"
    )

    # Try out ROOT based unfolding. Based on my tests, this doesn't matter...
    # fOut = ROOT.TFile(str(settings.output_dir / "test_unfolding_cpp.root"), "RECREATE")
    # ROOT.Unfold2D(responses.response, hists["h2_true"], hists["h2_raw"], ROOT.RooUnfold.ErrorTreatment.kCovariance, fOut, "", 20)
    # for v in hists.values():
    #    v.Write()
    ##fOut.Write()
    # fOut.Close()

    return True


def run_unfolding_closure_reweighting(
    settings: Settings,
    embedded_filenames: Sequence[Path],
    embedded_tree_name: str,
    closure_variation: str,
    fraction_for_response: float = 0.75,
    reweight_data_dataset_name: str = "",
    reweight_embedded_dataset_name: str = "",
) -> bool:
    """Run unfolding closure with reweighting.

    Note:
        Must run separately with and without pure matches because the response is different.

    Args:
        settings: Unfolding settings.
        embedded_filenames: Filenames for embedded data.
        closure_variation: Name of the closure variation.
        fraction_for_response: Fraction of statistics for the response. Default: 0.75, as determined by
            comparing error bars in data and embedded.
    Returns:
        True if successful.
    """
    # Delayed import to avoid direct dependence.
    import ROOT

    # Setup
    _setup_unfolding()
    # Validate variations.
    _variations = {
        "split_MC": ROOT.ClosureVariation_t.splitMC,
        "reweight_pseudo_data": ROOT.ClosureVariation_t.reweightPseudoData,
        "reweight_response": ROOT.ClosureVariation_t.reweightResponse,
    }
    variation = _variations[closure_variation]

    # Define hists (and the map to pass them into ROOT for unfolding)
    hists = _default_hists(settings=settings)
    # Add pseudo-data and pseudo-true. They're equivalent to raw and true, so they can just be cloned.
    hists["h2_pseudo_data"] = hists["h2_raw"].Clone("h2_pseudo_data")
    hists["h2_pseudo_true"] = hists["h2_true"].Clone("h2_pseudo_true")

    hists_map_for_root = _hists_to_map_for_ROOT(hists=hists)

    # Load hists for reweighting and calculate ratio.
    h_reweighting_ratio = ROOT.nullptr
    if variation != ROOT.ClosureVariation_t.splitMC:
        h_reweighting_ratio = _get_reweighting_ratio(
            reweight_data_dataset_name=reweight_data_dataset_name,
            reweight_embedded_dataset_name=reweight_embedded_dataset_name,
            settings=settings,
        )

    # Create the responses. We assume some conventions about column names.
    # They should generally be reasonable, but may require tweaks from time to time.
    responses = ROOT.create_closure_response_2D(
        hists_map_for_root,
        settings.grooming_method,
        settings.substructure_variable.variable_name,
        _array_to_ROOT(settings.jet_pt.smeared_bins, "double"),
        _array_to_ROOT(settings.jet_pt.true_bins, "double"),
        _array_to_ROOT(settings.substructure_variable.smeared_bins, "double"),
        _array_to_ROOT(settings.substructure_variable.true_bins, "double"),
        settings.substructure_variable.untagged_value,
        settings.substructure_variable.smeared_range.min,
        settings.substructure_variable.smeared_range.max,
        _array_to_ROOT(_pass_filenames_to_ROOT(embedded_filenames), "std::string"),
        variation,
        fraction_for_response,
        settings.use_pure_matches,
        h_reweighting_ratio,
        embedded_tree_name,
    )

    # Perform the actual unfolding.
    # First, the standard split MC closure
    output_hists = {}
    output_hists.update(
        unfolding_2D(
            response=responses.response,
            input_spectra=hists["h2_pseudo_data"],
            true_spectra=hists["h2_pseudo_true"],
        )
    )

    # TODO: Do I want the trivial closure for the splitMC?

    # Store the output hists.
    _write_hists([hists, output_hists], settings.output_filename, additional_tag=f"closure_{closure_variation}")

    return True


def run_unfolding_tree(
    grooming_method: str,
    substructure_variable_name: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
    true_substructure_variable_bins: np.ndarray,
    true_jet_pt_bins: np.ndarray,
    # data_filenames: Sequence[Path],
    # embedded_filenames: Sequence[Path],
    # output_filename: Path,
) -> bool:
    ...

    # Delayed import to avoid direct dependence.
    import ROOT

    # Configuration (not totally clear if this actually does anything for this script...)
    ROOT.ROOT.EnableImplicitMT(1)

    data_chain = ROOT.TChain("tree")
    data_chain.Add("trains/PbPb/5863/skim/*.root")

    # the raw correlation (ie. data)
    h2_raw = ROOT.TH2D(
        "r",
        "raw",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(smeared_jet_pt_bins) - 1,
        smeared_jet_pt_bins,
    )
    ## detector measure level (ie. hybrid)
    # h2_smeared = ROOT.TH2D(
    #    "smeared",
    #    "smeared",
    #    len(smeared_substructure_variable_bins) - 1,
    #    smeared_substructure_variable_bins,
    #    len(smeared_jet_pt_bins) - 1,
    #    smeared_jet_pt_bins,
    # )
    ## detector measure level no cuts (ie. hybrid, but no cuts).
    ## NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the true_jet_pt_bins.
    # h2_smeared_no_cuts = ROOT.TH2D(
    #    "smearednocuts",
    #    "smearednocuts",
    #    len(smeared_substructure_variable_bins) - 1,
    #    smeared_substructure_variable_bins,
    #    len(true_jet_pt_bins) - 1,
    #    true_jet_pt_bins,
    # )
    ## true correlations with measured cuts
    # h2_true = ROOT.TH2D(
    #    "true",
    #    "true",
    #    len(true_substructure_variable_bins) - 1,
    #    true_substructure_variable_bins,
    #    len(true_jet_pt_bins) - 1,
    #    true_jet_pt_bins,
    # )
    ## full true correlation (without cuts)
    # h2_full_eff = ROOT.TH2D(
    #    "truef",
    #    "truef",
    #    len(true_substructure_variable_bins) - 1,
    #    true_substructure_variable_bins,
    #    len(true_jet_pt_bins) - 1,
    #    true_jet_pt_bins,
    # )
    ## Correlation between the splitting variables at true and hybrid (with cuts).
    # h2_substructure_variable = ROOT.TH2D(
    #    "h2SplittingVariable",
    #    "h2SplittingVariable",
    #    len(smeared_substructure_variable_bins) - 1,
    #    smeared_substructure_variable_bins,
    #    len(true_substructure_variable_bins) - 1,
    #    true_substructure_variable_bins,
    # )

    # Should determine the untagged bin value (?)
    # untagged_bin_value = [1, 2]

    data_prefix = "data"
    data_jet_pt_name = f"{data_prefix}_jet_pt"
    data_substructure_variable_name = f"{grooming_method}_{data_prefix}_{substructure_variable_name}"

    # TEMP for quick performance test
    smeared_untagged_bin_value = 2.5
    min_smeared_substructure_variable = 3
    max_smeared_substructure_variable = 15

    logger.info("Starting loop")
    # TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
    # TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());
    for jet in data_chain:
        # Since the names are dynamic, we need to retrieve them each event using getattr.
        data_jet_pt = getattr(jet, data_jet_pt_name)
        data_substructure_variable = getattr(jet, data_substructure_variable_name)
        # Jet pt cut.
        if data_jet_pt < smeared_jet_pt_bins[0] or data_jet_pt > smeared_jet_pt_bins[-1]:
            continue
        # Substructure variable cut.
        if data_substructure_variable < 0:
            # Assign to the untagged bin.
            data_substructure_variable = smeared_untagged_bin_value
        else:
            if (
                data_substructure_variable < min_smeared_substructure_variable
                or data_substructure_variable > max_smeared_substructure_variable
            ):
                continue
        h2_raw.Fill(data_substructure_variable, data_jet_pt)
    logger.info("Done with loop")

    # Embedding

    return True


def run_unfolding_rdf(
    grooming_method: str,
    substructure_variable_name: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
    true_substructure_variable_bins: np.ndarray,
    true_jet_pt_bins: np.ndarray,
    # data_filenames: Sequence[Path],
    # embedded_filenames: Sequence[Path],
    # output_filename: Path,
    jet_pt_prefix_first: bool = True,
) -> bool:
    # Validation
    # Parameters
    jet_pt_column_format = "jet_pt_{prefix}"
    if jet_pt_prefix_first:
        jet_pt_column_format = "{prefix}_jet_pt"

    # Delayed import to avoid direct dependence.
    import ROOT

    # Configuration (not totally clear if this actually does anything for this script...)
    ROOT.ROOT.EnableImplicitMT(1)

    data_chain_data = ROOT.TChain("tree")
    data_chain_data.Add("trains/PbPb/5863/skim/*.root")
    df_data = ROOT.RDataFrame(data_chain_data)

    # the raw correlation (ie. data)
    h2_raw_args = (
        "r",
        "raw",
        len(smeared_substructure_variable_bins) - 1,
        smeared_substructure_variable_bins,
        len(smeared_jet_pt_bins) - 1,
        smeared_jet_pt_bins,
    )
    # detector measure level (ie. hybrid)
    # h2_smeared_args = (
    #    "smeared",
    #    "smeared",
    #    len(smeared_substructure_variable_bins) - 1,
    #    smeared_substructure_variable_bins,
    #    len(smeared_jet_pt_bins) - 1,
    #    smeared_jet_pt_bins,
    # )
    ## detector measure level no cuts (ie. hybrid, but no cuts).
    ## NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the true_jet_pt_bins.
    # h2_smeared_no_cuts_args = (
    #    "smearednocuts",
    #    "smearednocuts",
    #    len(smeared_substructure_variable_bins) - 1,
    #    smeared_substructure_variable_bins,
    #    len(true_jet_pt_bins) - 1,
    #    true_jet_pt_bins,
    # )
    ## true correlations with measured cuts
    # h2_true_args = (
    #    "true",
    #    "true",
    #    len(true_substructure_variable_bins) - 1,
    #    true_substructure_variable_bins,
    #    len(true_jet_pt_bins) - 1,
    #    true_jet_pt_bins,
    # )
    ## full true correlation (without cuts)
    # h2_full_eff_args = (
    #    "truef",
    #    "truef",
    #    len(true_substructure_variable_bins) - 1,
    #    true_substructure_variable_bins,
    #    len(true_jet_pt_bins) - 1,
    #    true_jet_pt_bins,
    # )
    ## Correlation between the splitting variables at true and hybrid (with cuts).
    # h2_substructure_variable_args = (
    #    "h2SplittingVariable",
    #    "h2SplittingVariable",
    #    len(smeared_substructure_variable_bins) - 1,
    #    smeared_substructure_variable_bins,
    #    len(true_substructure_variable_bins) - 1,
    #    true_substructure_variable_bins,
    # )

    # Need to make arguments, cleanup, consolidate...
    n_cores = 1
    data_prefix = "data"
    data_jet_pt_name = f"{data_prefix}_jet_pt"
    data_substructure_variable_name = f"{grooming_method}_{data_prefix}_{substructure_variable_name}"

    # TEMP for quick performance test
    smeared_untagged_bin_value = 2.5
    min_smeared_substructure_variable = 3
    max_smeared_substructure_variable = 15

    # Define RooUnfold Objects
    # We have to do this in c++ for ROOT to be able to access them for the RDF.
    # Might also be necessary...
    ROOT.gSystem.Load("libRooUnfold")
    r = f"""
    #include <vector>
    #include <RooUnfoldBayes.h>
    #include <RooUnfoldResponse.h>

    // We can access these directly.
    std::vector<std::unique_ptr<RooUnfoldResponse>> responses;
    std::vector<std::unique_ptr<RooUnfoldResponse>> responses_no_trunc;

    // NOTE: You can't have a for loop directly in a Declare, apparently...
    // NOTE: You can't do this directly with RooUnfold objects without setting up the response because the
    //       copy constructor calls setup...
    //       RooUnfold sucks!
    void setupResponses() {{
        for (unsigned int i = 0; i < {n_cores}; i++) {{
            responses.emplace_back(std::unique_ptr<RooUnfoldResponse>());
            responses_no_trunc.emplace_back(std::unique_ptr<RooUnfoldResponse>());
        }}
    }}

    /*double getSubstructureVariable(double {data_substructure_variable_name}) {{
        return {data_substructure_variable_name} < 0 ? {smeared_untagged_bin_value} : {data_substructure_variable_name};
    }}*/
    double getSubstructureVariable(double var) {{
        return var < 0 ? {smeared_untagged_bin_value} : var;
    }}
    std::vector <TRandom3> randoms;
    void setupRandomNumbers() {{
        for (unsigned int i = 0; i < {n_cores}; i++) {{
            randoms.emplace_back(TRandom3(0));
        }}
    }}
    // Alternatively, we could use the special rdfentry_ to do something like this, but less well...
    bool randomSample(unsigned int slotNumber, double jetPt) {{
        return randoms[slotNumber].Rndm() > 0.9;
    }}
    //auto testRandomSample = [randoms](unsigned int slotNumber,

    void setup() {{
        setupResponses();
        setupRandomNumbers();
    }}
    std::vector<std::string> colName = {{ "{jet_pt_column_format.format(prefix='hybrid')}" }};
    """
    print(r)
    ROOT.gInterpreter.Declare(r)

    logger.info(ROOT.RooUnfold.kCovariance)
    logger.info(ROOT.setup())
    logger.info(ROOT.responses)
    logger.info(ROOT.randomSample(0, 12.0))

    logger.info("Starting data frame")
    # TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
    # TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());

    smeared_jet_pt_filter = (
        f"{data_jet_pt_name} >= {smeared_jet_pt_bins[0]} && {data_jet_pt_name} <= {smeared_jet_pt_bins[-1]}"
    )
    substructure_variable_value_filter = f"({data_substructure_variable_name} >= {min_smeared_substructure_variable} && {data_substructure_variable_name} <= {max_smeared_substructure_variable}) || ({data_substructure_variable_name} < 0)"
    df_data = df_data.Filter(f"({smeared_jet_pt_filter}) && ({substructure_variable_value_filter})")
    print(data_substructure_variable_name)
    df_data = df_data.Define(
        "data_substructure_variable",
        f"getSubstructureVariable({data_substructure_variable_name})",
        # f"[](double {data_substructure_variable_name}) {{ return {data_substructure_variable_name} < 0 ? {smeared_untagged_bin_value} : {data_substructure_variable_name} }}",
        # lambda substructure_variable: smeared_untagged_bin_value if substructure_variable < 0 else substructure_variable,
        # lambda leading_kt_data_kt: smeared_untagged_bin_value if leading_kt_data_kt < 0 else leading_kt_data_kt,
        # [data_substructure_variable_name]
    )

    # Data
    h2_raw = df_data.Histo2D(
        h2_raw_args,
        "data_substructure_variable",
        data_jet_pt_name,
    )

    logger.info("Starting calculation")
    logger.info(f"Entries: {h2_raw.GetEntries()}")

    # Starting embedding from here, but needs cleanup...
    data_chain_embedded = ROOT.TChain("tree")
    # Make these an argument...
    data_chain_embedded.Add("trains/embedPythia/5966/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5967/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5968/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5969/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5970/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5971/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5972/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5973/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5974/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5975/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5976/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5977/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5978/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5979/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5980/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5981/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5982/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5983/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5984/skim/*.root")
    data_chain_embedded.Add("trains/embedPythia/5985/skim/*.root")
    df_embedded = ROOT.RDataFrame(data_chain_embedded)

    # Make these arguments
    smeared_cut_prefix = "hybrid"

    # Implement all of the embedding cuts via filters.
    # NOTE: Many of these values are hard coded and should be refactored at some point.
    true_jet_pt_filter = f"{jet_pt_column_format.format(prefix='true')} <= 160"
    true_substructure_variable_filter = f"{grooming_method}_true_kt <= 100"
    # Double counting is already applied above.
    # Now, the hybrid cuts.
    # Hybrid jet pt
    hybrid_jet_pt_filter = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= 40 && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < 120"
    # Hybrid substructure variable
    df_embedded = df_embedded.Filter(
        "(" + ") && (".join([true_jet_pt_filter, true_substructure_variable_filter, hybrid_jet_pt_filter]) + ")"
    )
    # df_embedded.ForeachSlot(ROOT.randomSample)
    # df_embedded.ForeachSlot(ROOT.randomSample, [jet_pt_column_format.format(prefix=smeared_cut_prefix)])
    # df_embedded.ForeachSlot(lambda slot, val: (slot, val), "jet_pt_hybrid")
    # df_embedded.ForeachSlot("randomSample", ROOT.colName)
    # df_embedded.Foreach("randomSample")

    # TEST
    smearedJetPtBins = np.array([30, 40, 50, 60, 80, 100, 120], dtype=np.float64)
    trueJetPtBins = np.array([0, 30, 40, 60, 80, 100, 120, 160], dtype=np.float64)
    smearedSplittingVariableBins = np.array([1, 2, 3, 4, 5, 7, 10, 15], dtype=np.float64)
    # NOTE: (-0.05, 0) is the untagged bin.
    trueSplittingVariableBins = np.array([-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100], dtype=np.float64)

    h1_test = ROOT.TH1D("smeared_1", "smeared_1", len(smearedJetPtBins) - 1, smearedJetPtBins)
    h2_smeared = ROOT.TH2D(
        "smeared",
        "smeared",
        len(smearedSplittingVariableBins) - 1,
        smearedSplittingVariableBins,
        len(smearedJetPtBins) - 1,
        smearedJetPtBins,
    )
    h2_true = ROOT.TH2D(
        "true",
        "true",
        len(trueSplittingVariableBins) - 1,
        trueSplittingVariableBins,
        len(trueJetPtBins) - 1,
        trueJetPtBins,
    )
    response = ROOT.RooUnfoldResponse()
    response.Setup(h2_smeared, h2_true)
    # df_embedded.Fill(response, ["hybridSubstructureVariableValue", jet_pt_column_format.format(prefix=smeared_cut_prefix), "trueSubstructureVariable", jet_pt_column_format.format(prefix="true"), "scale_factor"])
    # df_embedded.Fill["double", "double", "double", "double"](response, ["hybridSubstructureVariableValue", jet_pt_column_format.format(prefix=smeared_cut_prefix), "trueSubstructureVariable", jet_pt_column_format.format(prefix="true")])
    df_embedded.Fill("double")(h1_test, ROOT.colName)

    logger.info("Done with loop")

    return True


if __name__ == "__main__":
    helpers.setup_logging()
    # run_unfolding_rdf(
    #    grooming_method="leading_kt",
    #    substructure_variable_name="kt",
    #    smeared_substructure_variable_bins=np.array(
    #        [1, 2, 3, 4, 5, 7, 10, 15],
    #        dtype=np.float64,
    #    ),
    #    smeared_jet_pt_bins=np.array(
    #        [30, 40, 50, 60, 80, 100, 120],
    #        dtype=np.float64,
    #    ),
    #    true_substructure_variable_bins=np.array(
    #        # NOTE: (-0.05, 0) is the untagged bin.
    #        [-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100],
    #        dtype=np.float64,
    #    ),
    #    true_jet_pt_bins=np.array(
    #        [0, 30, 40, 60, 80, 100, 120, 160],
    #        dtype=np.float64,
    #    ),
    # )

    grooming_method = "leading_kt"
    default_settings = Settings(
        grooming_method=grooming_method,
        jet_pt=ParameterSettings(
            true_bins=np.array([0, 30, 40, 60, 80, 100, 120, 160], dtype=np.float64),
            smeared_bins=np.array([30, 40, 50, 60, 80, 100, 120], dtype=np.float64),
        ),
        substructure_variable=SubstructureVariableSettings.from_binning(
            true_bins=np.array(
                # NOTE: (-0.05, 0) is the untagged bin.
                [-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100],
                dtype=np.float64,
            ),
            smeared_bins=np.array([1, 2, 3, 4, 5, 7, 10, 15], dtype=np.float64),
            name="kt",
            variable_name="kt",
            untagged_bin_below_range=True,
        ),
        suffix="central",
        output_dir=Path("output/PbPb/unfolding/test_2"),
        use_pure_matches=False,
    )

    logger.info("Running...")
    run_unfolding(
        settings=default_settings,
        # NOTE: TChain can only handle one "*" in the filename.
        data_filenames=[Path("trains/PbPb/6672/skim/*.root")],
        embedded_filenames=[
            Path(f"trains/embedPythia/{train_number}/skim/*.root") for train_number in list(range(6650, 6669)) + [6671]
        ],
        data_tree_name="tree",
        embedded_tree_name="tree",
    )

    # run_unfolding_closure_reweighting(
    #    settings=setup("dynamical_kt"),
    #    # NOTE: TChain can only handle one "*" in the filename.
    #    embedded_filenames=[
    #        Path(f"trains/embedPythia/{train_number}/skim/*.root") for train_number in range(5966, 5986)
    #    ],
    #    # closure_variation="reweight_pseudo_data",
    #    closure_variation="split_MC",
    # )
