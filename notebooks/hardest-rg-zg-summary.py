# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Substructure w/ ROOT 6.28.04, conda
#     language: python
#     name: substructure_c_28_04
# ---

# # Rg and zg summary and performance plots

# +
from __future__ import annotations

# Setup
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pachyderm.plot as pb
from mammoth import helpers as mammoth_helpers

from jet_substructure.analysis import (
    model_calculations,
    plot_paper,
    plot_from_skim,
    plot_unfolding,
    unfolding_analysis,
    plot_style
)
from jet_substructure.base import helpers, notebook_utils as nb_utils

# %load_ext autoreload
# %autoreload 2

# #%matplotlib inline
# #%config InlineBackend.figure_formats = ["png", "pdf"]
# Don't show mpl images inline. We'll handle displaying them separately.
plt.ioff()
# Ensure the axes are legible on a dark background
mpl.rcParams['figure.facecolor'] = 'w'

mammoth_helpers.setup_logging(level=logging.DEBUG)
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
logging.getLogger("boost_histogram").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# General settings
embed_images = False
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)


# -

# ## pp
#
# ### Load data

# +
def grooming_name(grooming_method: str, prefixes: list[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "pp"
prefix = "data"
#jet_pt_bin = helpers.JetPtRange(min=40, max=120)
jet_pt_bin = helpers.JetPtRange(min=20, max=85)
text_label = "Iterative splittings"
text_label += "\n" + f"${jet_pt_bin.display_str(label='meas.')}$"
tag = jet_pt_bin.histogram_str(prefix)
system_label = plot_style.label_to_display_string["collision_system"][f"{collision_system}_5TeV"]

# Setup
#_all_available_methods = list(nb_utils.all_grooming_methods)
_grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
]
_method_groups = [
    _grooming_methods,
    # Older
    ##nb_utils.leading_kt_grooming_methods,
    #nb_utils.dynamical_grooming_methods,
    #nb_utils.soft_drop_grooming_methods,
    #nb_utils.dynamical_grooming_with_z_cut,
    #nb_utils.z_cut_02_grooming_methods,
    ##_all_available_methods,
]
# We want to skip the leading_kt methods, but not redefine all_grooming_methods in nb_utils,
# so we combine all of the above
_all_available_methods = [_method for _group in _method_groups for _method in _group]
# Add the add the all methods to the method groups
_method_groups.append(_all_available_methods)

# Output
#_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "2023-paper" /"semi_central_R02" / "pass3" / prefix
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "2023-paper" /"pp_R02" / "pass1" / prefix
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_data_pp = {}
_successfully_loaded_methods = []
#dataset_name = "LHC18qr_semi_central_R02_6765"
#dataset_name = "LHC18qr_semi_central_R02_0067"
dataset_name = "LHC17pq_pp_R02_0060"
#for grooming_method in nb_utils.all_grooming_methods:
for grooming_method in _grooming_methods:
    try:
        hists_data_pp.update(nb_utils.load_histograms(
            filename = f"{dataset_name}_{grooming_name(grooming_method, [prefix])}.root", collision_system=collision_system,
            tag = "RDF", base_path = Path("output")
        ))
        _successfully_loaded_methods.append(grooming_method)
    except FileNotFoundError:
        logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_data_pp = {k: v.to_boost_histogram() for k, v in hists_data_pp.items()}
# -

logger.info(f"\nSuccessfully loaded grooming methods: {_successfully_loaded_methods}")

from importlib import reload
reload(plot_from_skim)

# ### Stats

# Setup
plot_png = False
for smeared_min_kt in [None, 1.0, 1.5]:
    _filenames = plot_from_skim.plot_Rg_vs_jet_pt_stats(
        hists=hists_data_pp,
        grooming_methods=_all_available_methods,
        prefix=prefix,
        jet_pt_bin=jet_pt_bin,
        rdf_plots=True,
        output_dir=_output_dir,
        plot_png=plot_png,
        system_label=system_label,
        smeared_min_kt=smeared_min_kt,
    )

    _filenames = plot_from_skim.plot_zg_vs_jet_pt_stats(
        hists=hists_data_pp,
        grooming_methods=_all_available_methods,
        prefix=prefix,
        jet_pt_bin=jet_pt_bin,
        rdf_plots=True,
        output_dir=_output_dir,
        plot_png=plot_png,
        system_label=system_label,
        smeared_min_kt=smeared_min_kt,
    )


# ## Central PbPb

# ### Load data

# +
def grooming_name(grooming_method: str, prefixes: list[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "PbPb"
prefix = "data"
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
text_label += "\n" + f"${jet_pt_bin.display_str(label='meas.')}$"
tag = jet_pt_bin.histogram_str(prefix)
system_label = plot_style.label_to_display_string["collision_system"][f"{collision_system}_5TeV"]

# Setup
#_all_available_methods = list(nb_utils.all_grooming_methods)
_grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
]
_method_groups = [
    _grooming_methods,
    # Older
    ##nb_utils.leading_kt_grooming_methods,
    #nb_utils.dynamical_grooming_methods,
    #nb_utils.soft_drop_grooming_methods,
    #nb_utils.dynamical_grooming_with_z_cut,
    #nb_utils.z_cut_02_grooming_methods,
    ##_all_available_methods,
]
# We want to skip the leading_kt methods, but not redefine all_grooming_methods in nb_utils,
# so we combine all of the above
_all_available_methods = [_method for _group in _method_groups for _method in _group]
# Add the add the all methods to the method groups
_method_groups.append(_all_available_methods)

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "2023-paper" /"central_R02" / "pass3" / prefix
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_data_central = {}
_successfully_loaded_methods = []
#dataset_name = "LHC18qr_semi_central_R02_6765"
#dataset_name = "LHC18qr_semi_central_R02_0067"
dataset_name = "LHC18qr_central_R02_0070"
#for grooming_method in nb_utils.all_grooming_methods:
for grooming_method in _grooming_methods:
    try:
        hists_data_central.update(nb_utils.load_histograms(
            filename = f"{dataset_name}_{grooming_name(grooming_method, [prefix])}.root", collision_system=collision_system,
            tag = "RDF", base_path = Path("output")
        ))
        _successfully_loaded_methods.append(grooming_method)
    except FileNotFoundError:
        logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_data_central = {k: v.to_boost_histogram() for k, v in hists_data_central.items()}
# -

logger.info(f"\nSuccessfully loaded grooming methods: {_successfully_loaded_methods}")

from importlib import reload
reload(plot_from_skim)

# ### Stats

# +
# Setup
plot_png = False
smeared_min_kt_central = 1.5

_filenames = plot_from_skim.plot_Rg_vs_jet_pt_stats(
    hists=hists_data_central,
    grooming_methods=_all_available_methods,
    prefix=prefix,
    jet_pt_bin=jet_pt_bin,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=plot_png,
    system_label=system_label,
    smeared_min_kt=smeared_min_kt_central,
)

_filenames = plot_from_skim.plot_zg_vs_jet_pt_stats(
    hists=hists_data_central,
    grooming_methods=_all_available_methods,
    prefix=prefix,
    jet_pt_bin=jet_pt_bin,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=plot_png,
    system_label=system_label,
    smeared_min_kt=smeared_min_kt_central,
)
# -



# ## Semi-central PbPb

# ### Load data

# +
def grooming_name(grooming_method: str, prefixes: list[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "PbPb"
prefix = "data"
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
text_label += "\n" + f"${jet_pt_bin.display_str(label='meas.')}$"
tag = jet_pt_bin.histogram_str(prefix)
system_label = plot_style.label_to_display_string["collision_system"][f"{collision_system}_5TeV"]

# Setup
#_all_available_methods = list(nb_utils.all_grooming_methods)
_grooming_methods = [
    "dynamical_core",
    "dynamical_kt",
    "dynamical_time",
]
_method_groups = [
    _grooming_methods,
    # Older
    ##nb_utils.leading_kt_grooming_methods,
    #nb_utils.dynamical_grooming_methods,
    #nb_utils.soft_drop_grooming_methods,
    #nb_utils.dynamical_grooming_with_z_cut,
    #nb_utils.z_cut_02_grooming_methods,
    ##_all_available_methods,
]
# We want to skip the leading_kt methods, but not redefine all_grooming_methods in nb_utils,
# so we combine all of the above
_all_available_methods = [_method for _group in _method_groups for _method in _group]
# Add the add the all methods to the method groups
_method_groups.append(_all_available_methods)

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "2023-paper" /"semi_central_R02" / "pass3" / prefix
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_data_semi_central = {}
_successfully_loaded_methods = []
#dataset_name = "LHC18qr_semi_central_R02_6765"
dataset_name = "LHC18qr_semi_central_R02_0067"
#for grooming_method in nb_utils.all_grooming_methods:
for grooming_method in _grooming_methods:
    try:
        hists_data_semi_central.update(nb_utils.load_histograms(
            filename = f"{dataset_name}_{grooming_name(grooming_method, [prefix])}.root", collision_system=collision_system,
            tag = "RDF", base_path = Path("output")
        ))
        _successfully_loaded_methods.append(grooming_method)
    except FileNotFoundError:
        logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_data_semi_central = {k: v.to_boost_histogram() for k, v in hists_data_semi_central.items()}
# -

logger.info(f"\nSuccessfully loaded grooming methods: {_successfully_loaded_methods}")

from importlib import reload
reload(plot_from_skim)

# ### Stats

# +
# Setup
plot_png = False
smeared_min_kt_semi_central = 1.0

_filenames = plot_from_skim.plot_Rg_vs_jet_pt_stats(
    hists=hists_data_semi_central,
    grooming_methods=_all_available_methods,
    prefix=prefix,
    jet_pt_bin=jet_pt_bin,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=plot_png,
    system_label=system_label,
    smeared_min_kt=smeared_min_kt_semi_central,
)

_filenames = plot_from_skim.plot_zg_vs_jet_pt_stats(
    hists=hists_data_semi_central,
    grooming_methods=_all_available_methods,
    prefix=prefix,
    jet_pt_bin=jet_pt_bin,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=plot_png,
    system_label=system_label,
    smeared_min_kt=smeared_min_kt_semi_central,
)
# -


