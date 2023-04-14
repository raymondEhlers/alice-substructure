"""Model calculations for comparisons

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, Sequence

import attrs
import numpy as np
import numpy.typing as npt
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import unfolding_base
from jet_substructure.base import helpers

logger = logging.getLogger(__name__)


@attrs.define
class ModelCalculation:
    name: str
    label: str
    normalized: bool
    metadata: dict[str, Any] = attrs.field(factory=dict)
    pp: dict[str, binned_data.BinnedData] = attrs.field(factory=dict)
    semi_central: dict[str, binned_data.BinnedData] = attrs.field(factory=dict)
    semi_central_ratio: dict[str, binned_data.BinnedData] = attrs.field(factory=dict)
    central: dict[str, binned_data.BinnedData] = attrs.field(factory=dict)
    central_ratio: dict[str, binned_data.BinnedData] = attrs.field(factory=dict)


class Model(Protocol):
    base_dir: Path
    label: str
    needs_normalization: bool
    metadata: dict[str, Any]

    def load_predictions(self, grooming_methods: list[str] | None = None) -> dict[str, dict[str, binned_data.BinnedData]]:
        ...


@attrs.define
class Jetscape:
    base_dir: Path
    label: str = attrs.field(default="JETSCAPEv3.5 AA22")
    needs_normalization: bool = attrs.field(default=False)
    metadata: dict[str, Any] = attrs.field(factory=dict)

    def load_predictions(self, grooming_methods: list[str] | None = None) -> dict[str, dict[str, binned_data.BinnedData]]:
        if grooming_methods is None:
            grooming_methods = [
                "dynamical_core",
                "dynamical_kt",
                "dynamical_time",
                "soft_drop_z_cut_02",
                "dynamical_core_z_cut_02",
                "dynamical_kt_z_cut_02",
                "dynamical_time_z_cut_02",
                "soft_drop_z_cut_04",
            ]
        grooming_methods_to_parameters = {
            # Here, (DyG a, z cut)
            "dynamical_core": (0.5, 0.0),
            "dynamical_kt": (1.0, 0.0),
            "dynamical_time": (2.0, 0.0),
            "dynamical_core_z_cut_02": (0.5, 0.2),
            "dynamical_kt_z_cut_02": (1.0, 0.2),
            "dynamical_time_z_cut_02": (2.0, 0.2),
            # Here, (beta, z_cut)
            "soft_drop_z_cut_02": (0.0, 0.2),
            "soft_drop_z_cut_04": (0.0, 0.4),
        }
        _centrality_bins = {
            "0-5": "pbpb0-5",
            "5-10": "pbpb5-10",
            "30-40": "pbpb30-40",
            "40-50": "pbpb40-50",
            "pp": "pp",
        }
        # First, try to grab the DyG predictions
        values: dict[str, dict[str, binned_data.BinnedData]] = {}
        for grooming_method in grooming_methods:
            values[grooming_method] = {}

            _grooming_param, _z_cut = grooming_methods_to_parameters[grooming_method]
            _grooming_label = "DynamicalGroom" if "dynamical" in grooming_method else "SoftDropGroom"
            _grooming_parameter_label = "aDyn" if "dynamical" in grooming_method else "beta"
            input_dir = self.base_dir / "combined"
            for _cent_bin, _cent_bin_label in _centrality_bins.items():
                filename = f"{_cent_bin_label}_{_grooming_label}_ktG_jetr0.2_ptj60-80_rapj0.0-0.7_pt0.0-2510.0_rap0.0-1.1_{_grooming_parameter_label}{_grooming_param:.02f}_zCut{_z_cut:.02f}.txt"
                logger.info(f"Loading {grooming_method}, {_cent_bin} from {filename}")
                loaded_data = np.loadtxt(input_dir / _cent_bin_label / filename)
                # Construct binned_data from input
                logger.info(f"{loaded_data[:, 1]}")
                bin_edges = np.concatenate([loaded_data[:, 1], loaded_data[-1:, 2]])
                logger.info(f"{bin_edges=}")
                data = binned_data.BinnedData(
                    axes=bin_edges,
                    values=loaded_data[:, 3],
                    variances=(loaded_data[:, 4] ** 2),
                )
                values[grooming_method][_cent_bin] = data

        # Merge relevant hists
        return_values: dict[str, dict[str, binned_data.BinnedData]] = {}
        # Handle by hand
        # First, just handle pp
        #return_values["pp"] = {
        #    _method: values[_method]["pp"]
        #    for _method in grooming_methods
        #}
        # Next, handle the PbPb
        ...

        # Alternatively, Handle automatically for all
        for collision_system, contributors in {
            "pp": ["pp"],
            "semi_central": ["30-40", "40-50"],
            "central": ["0-5", "5-10"],
        }.items():
            # NOTE: len(contributors) assumes that there's the same number of events in each cent bin.
            #       This should be approximately true, although it would be nicer if we could do this precisely.
            return_values[collision_system] = {
                _method: sum([  # type: ignore[misc]
                    values[_method][contributor] for contributor in contributors
                ]) / len(contributors)
                for _method in grooming_methods
            }

            for _method in grooming_methods:
                _data = return_values[collision_system][_method]
                if self.needs_normalization:
                    _data /= np.sum(_data.values)
                # normalize by bin widths
                # NOTE: I will need to check if this is necessary to compare to the spectra,
                #       but in any case, it's not critical for the ratio since it will cancel!
                _data /= _data.axes[0].bin_widths
                return_values[collision_system][_method] = _data

        return return_values

@attrs.define
class HybridModel:
    base_dir: Path
    label: str = attrs.field()
    needs_normalization: bool = attrs.field(default=False)
    metadata: dict[str, Any] = attrs.field(factory=dict)

    def load_predictions(self, grooming_methods: list[str] | None = None) -> dict[str, dict[str, binned_data.BinnedData]]:
        # NOTE: For now, only returns the ratio
        if grooming_methods is None:
            grooming_methods = [
                "dynamical_core",
                "dynamical_kt",
                "dynamical_time",
                "soft_drop_z_cut_02",
                "dynamical_core_z_cut_02",
                "dynamical_kt_z_cut_02",
                "dynamical_time_z_cut_02",
                "soft_drop_z_cut_04",
            ]

        grooming_methods_to_parameters = {
            # Here, (DyG a, z cut)
            "dynamical_core": (0.5, 0.0),
            "dynamical_kt": (1.0, 0.0),
            "dynamical_time": (2.0, 0.0),
            "dynamical_core_z_cut_02": (0.5, 0.2),
            "dynamical_kt_z_cut_02": (1.0, 0.2),
            "dynamical_time_z_cut_02": (2.0, 0.2),
            # Here, (beta, z_cut)
            "soft_drop_z_cut_02": (0.0, 0.2),
            "soft_drop_z_cut_04": (0.0, 0.4),
        }
        _soft_drop_prediction_indices = {
            "soft_drop_z_cut_00": 0,
            "soft_drop_z_cut_02": 1,
            "soft_drop_z_cut_04": 1,
        }
        _dyg_prediction_indices = {
            "dynamical_core": 0,
            "dynamical_core_z_cut_02": 0,
            "dynamical_kt": 1,
            "dynamical_kt_z_cut_02": 1,
            "dynamical_time": 2,
            "dynamical_time_z_cut_02": 2,
        }
        _pt_bin_index = {
            helpers.JetPtRange(60, 80): 2,
        }
        ...

        _include_elastic_map = {
            True: "Elastic",
            False: "NoElastic",
        }
        _include_wake_map = {
            True: "Wake_1",
            False: "Wake_0",
        }
        _centrality_map = {
            "semi_central": "3050",
            "central": "010",
        }

        # Input options
        include_elastic = self.metadata["include_elastic"]
        include_wake = self.metadata["include_wake"]
        bin_edges = self.metadata["bin_edges"]

        # Bin edges
        _axis = binned_data.Axis(bin_edges=bin_edges)
        _bin_centers = _axis.bin_centers

        values: dict[str, dict[str, binned_data.BinnedData]] = {}
        for centrality_bin in ["central", "semi_central"]:
            values[centrality_bin] = {}
            for grooming_method in grooming_methods:
                logger.info(f"Processing {centrality_bin}, {grooming_method}")
                _centrality_label = _centrality_map[centrality_bin]
                _elastic_label = _include_elastic_map[include_elastic]
                _wake_label = _include_wake_map[include_wake]

                _grooming_param, _z_cut = grooming_methods_to_parameters[grooming_method]
                _grooming_method_label = "DyG" if "dynamical" in grooming_method else "SD"
                _z_label_for_dyg = "_wzcut_" + ("1" if _z_cut > 0 else "0") if "dynamical" in grooming_method else ""
                if "dynamical" in grooming_method:
                    _second_grooming_label = "a_" + str(_dyg_prediction_indices[grooming_method])
                else:
                    _second_grooming_label = "Gro_" + str(_soft_drop_prediction_indices[grooming_method])

                filename = f"HYBRID_Hadrons_{_elastic_label}_5020_{_centrality_label}_{_wake_label}_JetR_2_kT_{_grooming_method_label}{_z_label_for_dyg}_JetBin_2_{_second_grooming_label}.dat"

                input_data = np.loadtxt(self.base_dir / filename)

                # Validate bin_edges with bin_centers
                bin_centers = input_data[:, 0]
                if not np.allclose(bin_centers, _bin_centers):
                    _msg = f"bin edges don't match centers: In file: {bin_centers}. passed: {_bin_centers}"
                    raise ValueError(_msg)

                ratio_upper_band = input_data[:, 5]
                ratio_lower_band = input_data[:, 6]

                # Convert from bands to a symmetric uncertainty to simplify plotting
                central_values = (ratio_upper_band + ratio_lower_band) / 2
                # Make it symmetric by construction.
                error_squared = (central_values - ratio_lower_band) ** 2
                #logger.info(f"{ratio_upper_band=}")
                #logger.info(f"{ratio_lower_band=}")
                #logger.info(f"{central_values=}")
                #logger.info(f"{error_squared=}")
                #logger.info(f"{np.sqrt(error_squared)=}")
                values[centrality_bin][grooming_method] = binned_data.BinnedData(
                    axes=[bin_edges],
                    values=central_values,
                    variances=error_squared,
                )

        return values


def _load_hybrid_model(
    base_dir: Path,
    dyg_filename: str,
    sd_filename: str,
    bin_edges: dict[str, dict[str, npt.NDArray[np.float64]]],
    jet_R: str,
    jet_pt_bin: helpers.JetPtRange,
    quantity_to_retrieve: str = "ratio",
) -> dict[str, dict[str, binned_data.BinnedData]]:
    output: dict[str, dict[str, binned_data.BinnedData]] = {}

    # Encode the file specification below
    # First, the options in the filename
    _wake_label = {
        True: "WantWake",
        # NOTE: I don't know it would look like without the wake. But since I don't have that file at
        #       the moment, it's not really an issue.
    }
    #_moliere_label = {
    #    True: "WithElastic",
    #    False: "NoElastic",
    #}
    # Next, the encoding of the data
    _kt_bin_center_index = 0
    # First value is the upper bound, second is the lower bound.
    _n_quantities = 6
    _quantity_to_relative_index = {
        "PbPb": (0, 1),
        "pp": (2, 3),
        "ratio": (4, 5),
    }
    _soft_drop_prediction_indices = {
        "soft_drop_z_cut_00": 0,
        "soft_drop_z_cut_02": 1,
    }
    _dyg_prediction_indices = {
        "dynamical_core": 0,
        "dynamical_kt": 1,
        "dynamical_time": 2,
    }
    _pt_bin_index = {
        helpers.JetPtRange(40, 60): 0,
        helpers.JetPtRange(60, 80): 1,
        helpers.JetPtRange(80, 100): 2,
        helpers.JetPtRange(100, 150): 3,
    }

    # Here's what we should grab, manually calculated:
    """
    SD zcut=0     1:18:19 w filledcu. (From file ending with SD.dat)
    SD zcut=0.2   1:24:25 w filledcu. (From file ending with SD.dat)
    DyG a=0.5     1:24:25 w filledcu. (From file ending with DyG.dat)
    DyG a=1       1:30:31 w filledcu. (From file ending with DyG.dat)
    DyG a=2       1:36:37 w filledcu. (From file ending with DyG.dat)
    """

    output[jet_R] = {}
    for grooming_method in [
            "dynamical_core",
            "dynamical_kt",
            "dynamical_time",
            "soft_drop_z_cut_02",
        ]:

        # This isn't especially efficient, but good enough for now
        filename = base_dir
        filename = base_dir / sd_filename if "soft_drop" in grooming_method else base_dir / dyg_filename
        data = np.loadtxt(filename)

        _axis = binned_data.Axis(bin_edges=bin_edges[jet_R][grooming_method])
        _bin_centers = _axis.bin_centers
        _values = []
        _variances = []
        _prediction_indices = _dyg_prediction_indices if "dynamical" in grooming_method else _soft_drop_prediction_indices
        for i_kt, row in enumerate(data):
            _bin_center = row[_kt_bin_center_index]
            # Assuming these all match up, we can just use the provided bin centers
            if _bin_center != _bin_centers[i_kt]:
                _msg = f"Mismatch between file bin center {_bin_center} and provided bin center: {_bin_centers[i_kt]}"
                raise ValueError(_msg)

            _pt_bin_offset = _pt_bin_index[jet_pt_bin] * 6
            _offset_for_bin_center = 1
            _offset_to_desired_quantities = (
                _offset_for_bin_center
                + (_pt_bin_index[jet_pt_bin] * len(_prediction_indices) * _n_quantities)
                + (_prediction_indices[grooming_method] * _n_quantities)
            )
            _relative_indices = _quantity_to_relative_index[quantity_to_retrieve]
            # (lower, upper)
            # Convert from bands to a symmetric uncertainty to simplify plotting
            band_indices = (
                _offset_to_desired_quantities + _relative_indices[1],
                _offset_to_desired_quantities + _relative_indices[0]
            )
            logger.info(
                f"grooming_method: {grooming_method}: (lower_band_index, upper_band_index): {band_indices}"
            )
            band = (row[band_indices[0]], row[band_indices[1]])
            central_value = sum(band)/len(band)
            # Make it symmetric by construction.
            error_squared = (central_value - band[0]) ** 2

            _values.append(central_value)
            _variances.append(error_squared)

        output[jet_R][grooming_method] = binned_data.BinnedData(
            axes=_axis,
            values=_values,
            variances=_variances,
        )

    return output


def load_hybrid_model_QM22(
    base_dir: Path,
    bin_edges: dict[str, dict[str, npt.NDArray[np.float64]]],
    jet_R: str,
    jet_pt_bin: helpers.JetPtRange,
    quantity_to_retrieve: str = "ratio",
) -> dict[str, dict[str, dict[str, binned_data.BinnedData]]]:
    _moliere_label = {
        True: "WithElastic",
        False: "NoElastic",
    }
    _moliere_output_label = {
        True: "hybrid_moliere",
        False: "hybrid_without_moliere",
    }

    output = {}
    for include_moliere in [False, True]:
        dyg_filename_template: str = "3050_kT_{moliere_label}_WantWake_1_JetR_2_kT_DyG.dat"
        sd_filename_template: str = "3050_kT_{moliere_label}_WantWake_1_JetR_2_kT_SD.dat"

        output[_moliere_output_label[include_moliere]] = _load_hybrid_model(
            base_dir=base_dir,
            dyg_filename=dyg_filename_template.format(moliere_label=_moliere_label[include_moliere]),
            sd_filename=sd_filename_template.format(moliere_label=_moliere_label[include_moliere]),
            bin_edges=bin_edges,
            jet_R=jet_R,
            jet_pt_bin=jet_pt_bin,
            quantity_to_retrieve=quantity_to_retrieve,
        )

    return output


def _load_analytical_calculations(filename: Path, bin_edges: npt.NDArray[np.float64]) -> binned_data.BinnedData:
    """Load analytical calculations for a given jet R, as determined by the filename."""
    # May not be terribly efficient, but it works automatically and it's a small amount of data, so it's good enough.
    arr = np.loadtxt(filename)
    central_values = arr[:, 0]
    lower_bounds = arr[:, 1]
    upper_bounds = arr[:, 2]

    h = binned_data.BinnedData(
        axes=bin_edges,
        values=central_values,
        variances=np.zeros(len(central_values)),
    )
    h.metadata["y_systematic"] = {
        # The asymmetric errors are expected to be differences
        "quadrature": unfolding_base.AsymmetricErrors(
            low=central_values-lower_bounds,
            high=upper_bounds-central_values,
        )
    }
    return h


def load_analytical_calculations(
    path_to_calculations: Path, bin_edges: dict[str, npt.NDArray[np.float64]]
) -> dict[str, dict[str, binned_data.BinnedData]]:
    """Load analytical calculations for a collection of jet R, as determined by the bin edges dict."""
    _grooming_methods_to_files = {
        "dynamical_kt": "1",
        "dynamical_time": "2",
    }
    output: dict[str, dict[str, binned_data.BinnedData]] = {}

    for jet_R_str, edges in bin_edges.items():
        output[jet_R_str] = {}
        for grooming_method, label in _grooming_methods_to_files.items():
            output[jet_R_str][grooming_method] = _load_analytical_calculations(
                filename=path_to_calculations / jet_R_str / f"ktg_a{label}.dat", bin_edges=edges
            )
    return output


def load_jetscape_data_jetscape_analysis(filename: Path) -> dict[str, dict[str, binned_data.BinnedData]]:
    """Load jetscape predictions for all jet R.

    Include DyG core, kt, and time, as well as SD z > 0.2.
    """
    _dyg_values = {
        "005": "dynamical_core",
        "010": "dynamical_kt",
        "020": "dynamical_time",
    }
    output: dict[str, dict[str, binned_data.BinnedData]] = {}
    with uproot.open(filename) as f:
        for jet_R in ["02", "04", "05"]:
            output[f"R{jet_R}"] = {}
            for val, grooming_method in _dyg_values.items():
                output[f"R{jet_R}"][grooming_method] = binned_data.BinnedData.from_existing_data(
                    f[f"h_chjet_ktg_dyg_a_{val}_alice_R{jet_R}_pt0.0Scaled"]
                )
            # Soft Drop
            output[f"R{jet_R}"]["soft_drop_z_cut_02"] = binned_data.BinnedData.from_existing_data(
                f[f"h_chjet_ktg_soft_drop_z_cut_02_alice_R{jet_R}_pt0.0Scaled"]
            )

    return output


def calculate_jetscape_ratio(
    pp: dict[str, dict[str, binned_data.BinnedData]], PbPb: dict[str, dict[str, binned_data.BinnedData]]
) -> dict[str, dict[str, binned_data.BinnedData]]:
    """ Calculate jetscape predictions from the pp and PbPb kt spectra. """
    output: dict[str, dict[str, binned_data.BinnedData]] = {}
    for jet_R, pp_R in pp.items():
        output[jet_R] = {}
        # Retrieve by hand just in case they're not in the same order...
        PbPb_R = PbPb[jet_R]
        for grooming_method, pp_hist in pp_R.items():
            # Retrieve by hand just in case they're not in the same order...
            PbPb_hist = PbPb_R[grooming_method]
            ratio = PbPb_hist / pp_hist
            # Then normalize
            ratio /= np.sum(ratio.values)
            ratio /= ratio.axes[0].bin_widths
            output[jet_R][grooming_method] = ratio

    return output


def load_sherpa_predictions(
    filename: Path, jet_R_values: float | Sequence[float]
) -> dict[str, dict[str, binned_data.BinnedData]]:
    """Load sherpa predictions for a given jet R.

    Include DyG core, kt, and time, as well as SD z > 0.2.
    """
    if isinstance(jet_R_values, float):
        jet_R_values = [jet_R_values]
    _name_map = {
        "k0": "dynamical_core",
        "k1": "dynamical_kt",
        "k2": "dynamical_time",
        "ksd": "soft_drop_z_cut_02",
    }
    output: dict[str, dict[str, binned_data.BinnedData]] = {}
    with uproot.open(filename) as f:
        for jet_R in jet_R_values:
            jet_R_str = f"R{round(jet_R * 10):02}"
            output[jet_R_str] = {}
            for tag, grooming_method in _name_map.items():
                output[jet_R_str][grooming_method] = binned_data.BinnedData.from_existing_data(f[f"histo{tag}"])

    return output
