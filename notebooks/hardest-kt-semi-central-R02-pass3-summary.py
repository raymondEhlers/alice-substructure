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
#     display_name: substructure_c_24_06
#     language: python
#     name: python3
# ---

# # Hardest $k_{\text{T}}$ Summary
#
# Summarize plots for semi-central PbPb R = 0.2, pass 3
#
# My overall summary based on everything below:
#
# - Distributions generally look okay to me.
# - Looking at the stats by bins, we may only be able to get to 7 GeV (central goes to 8). There are some counts in the 7-8 bin, but only 1-5 depending on the jet pt.
# - I'm a bit worried that we're missing the first splitting with some consistency based on the $n_{\text{split}}$ distribution, perhaps because it's going out of cone.
#     - Probably should check the responses next.

# Table of Contents:
#
# 1. [Embedded Substructure Variables](#Embedded-substructure-variables)
#     1. [$k_{\text{T}}$](#Embedded-$k_{\text{T}}$)
#     2. [$\Delta R$](#Embedded-$\Delta-R$)
#     3. [$z$](#Embedded-$z$)
#     4. [Number of splittings to selected splitting](#Embedded-$n_{\text{split}}$)
#     5. [Number of groomed splittings to selected splitting](#Embedded-$n_{\text{split,groomed}}$)
#     6. [Number of splittings which passed grooming](#Embedded-nsplit,groomedn_{\text{split,groomed}})
# 2. [Data Substructure Variables](#Pb-Pb-Data)
#     1. [$k_{\text{T}}$](#Data-$k_{\text{T}}$)
#     2. [$\Delta R$](#Data-$\Delta-R$)
#     3. [$z$](#Data-$z$)
#     4. [Number of splittings to selected splitting](#Data-$n_{\text{split}}$)
#     5. [Number of groomed splittings to selected splitting](#Data-$n_{\text{split,groomed}}$)
#     6. [Number of splittings which passed grooming](#Data-$n_{\text{passed-grooming}}$)
# 3. [Data-Embedded Comparison](#Data-vs-Embedded-Comparison)
#     1. [$k_{\text{T}}$](#Data-vs-Embedded-$k_{\text{T}}$)
#     2. [$\Delta R$](#Data-vs-Embedded-ΔR\Delta-R)
#     3. [$z$](#Data-vs-Embedded-$z$)
#     4. [Number of splittings to selected splitting](#Data-vs-Embedded-$n_{\text{split}}$)
#     5. [Number of groomed splittings to selected splitting](#Data-vs-Embedded-$n_{\text{split,groomed}}$)
#     6. [Number of splittings which passed grooming](#Data-vs-Embedded-$n_{\text{passed-grooming}}$)
# 4. [Hybrid vs True vs Det Level Substructure Variables](#Hybrid-vs-True-vs-Det-Level-Substructure-Variables)
#     1. [Fixed True Jet Pt](#Fixed-True-Jet-Pt)
#     2. [Fixed Hybrid Jet Pt](#Fixed-Hybrid-Jet-Pt)

# As a first step, we need to setup packages, as well as the data that we're going to use. First, the packages

# + tags=["remove_cell"]
# Setup
import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
from pachyderm import binned_data
import uproot

import jet_substructure.analysis.plot_base as pb
from jet_substructure.base import helpers, notebook_utils as nb_utils
from jet_substructure.analysis import new_plot_comparison, plot_from_skim

# %load_ext autoreload
# %autoreload 2

# #%matplotlib inline
# #%config InlineBackend.figure_formats = ["png", "pdf"]
# Don't show mpl images inline. We'll handle displaying them separately.
plt.ioff()
# Ensure the axes are legible on a dark background
matplotlib.rcParams['figure.facecolor'] = 'w'

helpers.setup_logging()
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
logging.getLogger("boost_histogram").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# General settings
embed_images = False
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
# -

# Next, code that's likely to be shared and refactored.



# ## Grooming methods comparison

# Setup for embedding

# + tags=["remove_cell"]
# First, any required imports for embedding
from jet_substructure.base import skim_analysis_objects

# Next, any helper functions
def grooming_name(grooming_method: str, prefixes: Sequence[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "embedPythia"
prefix = "hybrid"
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
text_label += "\n" + f"${jet_pt_bin.display_str(label=prefix)}$"
tag = jet_pt_bin.histogram_str(prefix)
# Group some of the grooming methods together for clarity
#_all_available_methods = []
#_all_available_methods.extend(nb_utils.leading_kt_grooming_methods[:2])
#_all_available_methods.extend(nb_utils.dynamical_grooming_methods[1:])
#_all_available_methods.extend(nb_utils.soft_drop_grooming_methods[:1])
#_method_groups = [
#    nb_utils.leading_kt_grooming_methods[:2],
#    nb_utils.dynamical_grooming_methods[1:],
#    nb_utils.soft_drop_grooming_methods[:1],
#    _all_available_methods
#]
_all_available_methods = list(nb_utils.all_grooming_methods)
_method_groups = [
    nb_utils.leading_kt_grooming_methods,
    nb_utils.dynamical_grooming_methods,
    nb_utils.soft_drop_grooming_methods,
    _all_available_methods,
]

# Helpers for plotting responses
_matching_name_to_axis_value: Dict[str, int] = {
    "all": 0,
    "pure": 1,
    "leading_untagged_subleading_correct": 2,
    "leading_correct_subleading_untagged": 3,
    "leading_correct_subleading_mistag": 4,
    "leading_mistag_subleading_correct": 5,
    "leading_untagged_subleading_mistag": 6,
    "leading_mistag_subleading_untagged": 7,
    "swap": 8,
    "both_untagged": 9,
}
_response_types = [
    skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="det_level"),
    skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="true"),
    skim_analysis_objects.ResponseType(measured_like="det_level", generator_like="true"),
]

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "semi_central_R02" / "pass3" / prefix
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_embedding = {}
_successfully_loaded_methods = []
dataset_name = "LHC20g4_embedded_into_LHC18qr_semi_central_R02_6798_6817"
for grooming_method in nb_utils.all_grooming_methods:
    try:
        hists_embedding.update(nb_utils.load_histograms(
            filename = f"{dataset_name}_{grooming_name(grooming_method, prefixes=['hybrid', 'true', 'det_level'])}.root", collision_system=collision_system,
            tag = "RDF", base_path = Path("output")
        ))
        _successfully_loaded_methods.append(grooming_method)
    except FileNotFoundError:
        logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_embedding = {k: v.to_boost_histogram() for k, v in hists_embedding.items()}
# -

logger.info(f"\nSuccessfully loaded grooming methods: {_successfully_loaded_methods}")

# ## Embedded substructure variables

# ### Embedded $k_{\text{T}}$

# + tags=["remove_cell"]
#fig, axes = plt.subplots(figsize=(24, 6), nrows=2, ncols=3, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, sharey=True,)
#fig, axes = plt.subplots(figsize=(8 * 3, 6), nrows=1, ncols=3, sharey=True,)
#for methods, ax in [(leading_kt_grooming_methods[:2], axes[0]),
#                    (dynamical_grooming_methods[1:], axes[1]),
#                    (soft_drop_grooming_methods[:1], axes[2])]:
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_embedding,
        grooming_methods=methods,
        attr_name="kt",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"kt_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig(
                        "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                    ),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="lower left", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        #fig=fig, ax=ax,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# For embedded $k_{\text{T}}$, we see:
#
# - Everything converge at high $k_{\text{T}}$.
# - We see some variations between the different methods at low $k_{\text{T}}$. 
# - $z > 0.2$ only varies from that without a z cut for $k_{\text{T}} < 2$
# - Approximately the same range as central R = 0.2

# ### Embedded $\Delta R$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_embedding,
        grooming_methods=methods,
        attr_name="delta_R",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"delta_R_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$\Delta R$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", log=False),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="upper left", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations for $\Delta R$:
#
# - Same kind of trends as for semi-central R = 0.4, but peaked towards smaller relative deltaR.
#     - ie. the semi-central had a much larger peak at the right side of the plot. More like DyG time, but most grooming methods were like that.
# - Peaks at smaller deltaR may indicate less background.

# ### Embedded $z$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_embedding,
        grooming_methods=methods,
        attr_name="z",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=True,
        plot_config=pb.PlotConfig(
            name=f"z_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$z$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", log=False, range=(-0.2, None)),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="lower right", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations for $z$:
#
# - Seems quite similar to semi-central R = 0.4, with perhaps slightly more untagged of the z cuts.
# - Overall, seems fine.
#

# ### Embedded $n_{\text{split}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_embedding,
        grooming_methods=methods,
        attr_name="n_to_split",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"number_to_split_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$n_{\text{split}}$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{split}}$"),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="center right", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Compared to semi-central R=0.4, the DyG seem a bit shifted from peaks at the first splitting towards more peaks at the second splitting.
#     - This seems a bit worrisome to me that this may be indicative of background. Seems to be strongest from lowest $a$.

# ### Embedded $n_{\text{split,groomed}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_embedding,
        grooming_methods=methods,
        attr_name="n_groomed_to_split",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"number_groomed_to_split_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$n_{\text{split,groomed}}$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{groomed,split}}$")
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="center right", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images);

# Observations:
#
# - Same trends for DyG as above.
# - $z > 0.2$ means that it's almost always the first splitting (or always, for SD - see below).
# - As a sanity check, SD looks right: only entries at 0 or 1, as it can't possibly have more.

# ### Embedded $n_{\text{passed grooming}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_embedding,
        grooming_methods=methods,
        attr_name="n_passed_grooming",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"number_passed_grooming_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$n_{\text{passed grooming}}$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{passed grooming}}$"),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="center right", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - This is equivalent to $n_{\text{SD}}$.
# - All of the grooming methods of the same $z$ cut sit on top of each other, as expected.

_filenames = plot_from_skim.plot_kt_vs_jet_pt(
    hists=hists_embedding,
    grooming_methods=nb_utils.all_grooming_methods,
    prefix=prefix,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=True,
)

nb_utils.display_images([
    _filenames[3*i:3*(i+1)] for i in range(0, int(len(_filenames)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)


# ## Pb-Pb Data
#
# Now, shift focus to the Pb-Pb data distributions
#
# First, a bit of setup
#
#

# + tags=["remove_cell"]
def grooming_name(grooming_method: str, prefixes: Sequence[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "PbPb"
prefix = "data"
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
text_label += "\n" + f"${jet_pt_bin.display_str(label='meas.')}$"
tag = jet_pt_bin.histogram_str(prefix)
system_label = pb.label_to_display_string["collision_system"][f"{collision_system}_5TeV"]

# Setup
#_all_available_methods = list(nb_utils.all_grooming_methods)
_method_groups = [
    #nb_utils.leading_kt_grooming_methods,
    nb_utils.dynamical_grooming_methods,
    nb_utils.soft_drop_grooming_methods,
    nb_utils.dynamical_grooming_with_z_cut,
    nb_utils.z_cut_02_grooming_methods,
    #_all_available_methods,
]
# We want to skip the leading_kt methods, but not redefine all_grooming_methods in nb_utils,
# so we combine all of the above
_all_available_methods = [_method for _group in _method_groups for _method in _group]
# Add the add the all methods to the method groups
_method_groups.append(_all_available_methods)

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "2023-02-HP" /"semi_central_R02" / "pass3" / prefix
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_data = {}
_successfully_loaded_methods = []
#dataset_name = "LHC18qr_semi_central_R02_6765"
dataset_name = "LHC18qr_semi_central_R02_0067"
for grooming_method in nb_utils.all_grooming_methods:
    try:
        hists_data.update(nb_utils.load_histograms(
            filename = f"{dataset_name}_{grooming_name(grooming_method, [prefix])}.root", collision_system=collision_system,
            tag = "RDF", base_path = Path("output")
        ))
        _successfully_loaded_methods.append(grooming_method)
    except FileNotFoundError:
        logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_data = {k: v.to_boost_histogram() for k, v in hists_data.items()}
# -

logger.info(f"\nSuccessfully loaded grooming methods: {_successfully_loaded_methods}")

# ### Data $k_{\text{T}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_data,
        grooming_methods=methods,
        attr_name="kt",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"kt_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(-2, 12)),
                    pb.AxisConfig(
                        "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                    ),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="lower left", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - $k_{\text{T}}$ stats look like it might make it up to maybe 8 GeV. This is a bit lower than central R=0.2 (~10)
# - Grooming methods are again sorted based on $z$ cut. $z > 0.4$ is really distorted, but $z > 0.2$ is more manageable.

# ### Data $\Delta R$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_data,
        grooming_methods=methods,
        attr_name="delta_R",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"delta_R_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$\Delta R$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", log=False),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="upper left", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images);

# Observations:
#
# - Looks quite similar to central R = 0.2

# ### Data $z$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_data,
        grooming_methods=methods,
        attr_name="z",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=True,
        plot_config=pb.PlotConfig(
            name=f"z_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$z$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$", log=False, range=(-0.2, None)),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="upper center", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Looks quite similar to central R=0.2 and semi-central R=0.4.

# ### Data $n_{\text{split}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_data,
        grooming_methods=methods,
        attr_name="n_to_split",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"number_to_split_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$n_{\text{split}}$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{split}}$"),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="center right", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Same shift for DyG towards the second splitting as shown for the embedded. Again, I find this a bit worrisome that we're losing the true hard splitting out of the cone. However, I'm not sure how much we should read into this.

# ### Data $n_{\text{split,groomed}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_data,
        grooming_methods=methods,
        attr_name="n_groomed_to_split",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"number_groomed_to_split_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$n_{\text{groomed,split}}$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{groomed,split}}$")
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="center right", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Same deal as with the embedding, and with $n_\text{split}$

# ### Data $n_{\text{passed groomed}}$

# + tags=["remove_cell"]
_filenames = []
for methods in _method_groups:
    _filenames.append(new_plot_comparison.plot_compare_grooming_methods_for_attribute(
        hists=hists_data,
        grooming_methods=methods,
        attr_name="n_passed_grooming",
        prefix=prefix,
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"number_passed_grooming_grooming_methods_{'_'.join(methods)}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$n_{\text{passed grooming}}$"),
                    pb.AxisConfig("y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{passed grooming}}$"),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text_label),
                legend=pb.LegendConfig(location="upper center", font_size=14),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.12)),
        ),
        output_dir=_output_dir,
        plot_png=True,
    ))
# -

nb_utils.display_images([
    # First, the summary
    _filenames[-1],
    # And then the breakdown
    _filenames[:-1],
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Same trends as embedding. Seems fine

# ### Statistics

# +
# Setup
plot_png = False

_filenames = plot_from_skim.plot_kt_vs_jet_pt_stats(
    hists=hists_data,
    grooming_methods=_all_available_methods,
    prefix=prefix,
    jet_pt_bin=jet_pt_bin,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=plot_png,
    system_label=system_label,
)
# -

nb_utils.display_images([
    _filenames[3*i:3*(i+1)] for i in range(0, int(len(_filenames)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# May only be able to go so 7.

# ### Lund Plane

from importlib import reload
reload(plot_from_skim)

_filenames = plot_from_skim.lund_plane(
    hists=hists_data,
    grooming_methods=nb_utils.all_grooming_methods,
    prefix=prefix,
    rdf_plots=True,
    output_dir=_output_dir,
    plot_png=True,
)

# # Data vs Embedded Comparison
#
# In order to run this section, it's required to run the setup for the embedded and data above.

# ### Data vs Embedded

# + tags=["remove_output"]
# Settings (merge with below)
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
# Group some of the grooming methods together for clarity
_all_available_methods = list(nb_utils.all_grooming_methods)
_method_groups = [
    nb_utils.leading_kt_grooming_methods,
    nb_utils.dynamical_grooming_methods,
    nb_utils.soft_drop_grooming_methods,
    _all_available_methods,
]

# Output
_output_dir = output_dir / "comparison" / "RDF" / "jupyter" / "semi_central_R02" / "pass3"
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

plot_from_skim.compare_grooming_methods_for_substructure_data_embed_prod(
    hists = [
        plot_from_skim.PlotHists(hists=hists_data, prefix="data", identifier="PbPb", display_label="Pb--Pb",),
        plot_from_skim.PlotHists(hists=hists_embedding, prefix="hybrid", identifier="hybrid", display_label="Hybrid",),
    ],
    grooming_methods=_all_available_methods,
    output_dir=_output_dir,
    rdf_plots=True,
    plot_png=True,
)


# -

def comparison_filename(grooming_method: str, variable: str, jet_pt_bin: helpers.JetPtRange, ) -> str:
    return f"{variable}_grooming_methods_{jet_pt_bin}_{grooming_method}_PbPb_hybrid_iterative_splittings"


# ### Data vs Embedded $k_{\text{T}}$

_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable="kt", jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, round(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# (Sorry, I need to fix the colors - they're automatically assigned, but got scrambled after HP)
#
# - Definitely some shape differences.
# - Embedded reproduces Pb-Pb fairly well for all grooming methods for most of the kt range.
# - Embedded usually under-predicts untagged.
# - At high $k_{\text{T}} > 8$, embedded is **above** Pb-Pb, but uncertainties are large for that bin.

# ### Data vs Embedded $\Delta R$

_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable="delta_R", jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Same sort of trends as semi-central R = 0.4
# - For all grooming methods, we have the same behavior:
#     - Small angles, PbPb is enhanced.
#     - Mid angles, PbPb is suppressed.
#     - Large angles, PbPb is mostly in agreement, but trending towards enhanced.
# - Exactly where the enhancement or suppression occur depends on the grooming method.
#     - ie. The details change a bit, even if the general story is the same.

# ### Data vs Embedded $z$

_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable="z", jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Agree within 10% for the most part.
# - Some definite differences at low $z$, as well as in the untagged.

# ### Data vs Embedded $n_{\text{split}}$

_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable="number_to_split", jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - For all grooming methods, PbPb has more untagged (usually with large uncertainties)
# - Ratio close to unity everywhere else.
# - For those without a cut, there generally seems to be a PbPb enhancement at high n.

# ### Data vs Embedded $n_{\text{split,groomed}}$

_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable="number_groomed_to_split", jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# Observations:
#
# - Same trends as for $n_{\text{split}}$.
# - Not much info for the $z > 0.2$ methods.

# ### Data vs Embedded $n_{\text{passed grooming}}$

_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable="number_passed_grooming", jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)


# Observations:
#
# - For the no $z$ cut methods: PbPb has enhanced untagged, and agreement around 4-6, but then enhancement for high $n$.
# - For $z > 0.2$ and $z > 0.4$ methods: Seems pretty consistent.
# - I'm not sure if we read too much into this.
#
# **Overall Summary**
#
# Embedded generally reproduces PbPb, although less well than for semi-central R = 0.4. In the details, there are some variations. As usual, the greatest difference in behavior between grooming methods is between no $z$ cut and $z > 0.2$ methods.

# # Hybrid vs True vs Det Level Substructure Variables

# + tags=["remove_cell"]
def grooming_name(grooming_method: str, prefixes: Sequence[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "embedPythia"
prefixes = ["hybrid", "true", "det_level"]
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
text_label += "\n" + f"${jet_pt_bin.display_str(label='')}$"
# Group some of the grooming methods together for clarity
_all_available_methods = list(nb_utils.all_grooming_methods)
_method_groups = [
    nb_utils.leading_kt_grooming_methods,
    nb_utils.dynamical_grooming_methods,
    nb_utils.soft_drop_grooming_methods,
    _all_available_methods,
]

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "semi_central_R02" / "pass3"
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"
_fig_output_dir.mkdir(parents=True, exist_ok=True)

# Load embedded hists for comparison
hists_embed = {}
dataset_name = "LHC20g4_embedded_into_LHC18qr_semi_central_R02_6798_6817"
_successfully_loaded_methods = []
for grooming_method in nb_utils.all_grooming_methods:
    try:
        hists_embed.update(nb_utils.load_histograms(
            filename = f"{dataset_name}_{grooming_name(grooming_method, prefixes)}.root", collision_system=collision_system,
            tag = "RDF", base_path = Path("output")
        ))
        _successfully_loaded_methods.append(grooming_method)
    except FileNotFoundError:
        logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")

# Convert to boost_histogram because it simplifies projections later.
hists_embed = {k: v.to_boost_histogram() for k, v in hists_embed.items()}
# -

# ## Fixed Hybrid Jet Pt

# + tags=["remove_cell"]
hists = [
    plot_from_skim.PlotHists(hists=hists_embed, prefix="true", identifier="true", display_label="True",),
    plot_from_skim.PlotHists(hists=hists_embed, prefix="det_level", identifier="det_level", display_label="Det. Level",),
    plot_from_skim.PlotHists(hists=hists_embed, prefix="hybrid", identifier="hybrid", display_label="Hybrid",),
]

# Settings
plot_png = True
jet_pt_selection_prefix = "hybrid"

# Settings to be refactored.
jet_pt_bin = helpers.RangeSelector(min=40, max=120)
text = "Iterative splittings"
text += "\n" + fr"${jet_pt_bin.display_str(label=jet_pt_selection_prefix)}\:\text{{GeV}}/c$"
ratio_label = f"Others/{hists[0].display_label}"
tag = jet_pt_bin.histogram_str(jet_pt_selection_prefix)

for grooming_method in _all_available_methods:
    new_plot_comparison.plot_compare_grooming_methods_for_prefix(
        hists=hists,
        grooming_methods=[grooming_method],
        attr_name="kt",
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"kt_embed_comparison_{jet_pt_selection_prefix}_jet_pt_selection",
            panels=[
                # Main axis.
                pb.Panel(
                    axes=pb.AxisConfig(
                        "y",
                        label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                        log=True,
                    ),
                    text=pb.TextConfig(x=0.96, y=0.96, text=text),
                    legend=pb.LegendConfig(location="lower left"),
                ),
                # Ratio.
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig("y", label=ratio_label, range=(-0.2, 4)),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12}),
        ),
        output_dir=_output_dir,
        plot_png=plot_png,
    )
    new_plot_comparison.plot_compare_grooming_methods_for_prefix(
        hists=hists,
        grooming_methods=[grooming_method],
        attr_name="delta_R",
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"delta_R_embed_comparison_{jet_pt_selection_prefix}_jet_pt_selection",
            panels=[
                # Main axis.
                # NOTE: This intentionally cuts off the normalization bin
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", range=(0, 0.41)),
                        pb.AxisConfig(
                            "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", range=(-0.4, 19.1)
                        ),
                    ],
                    text=pb.TextConfig(x=0.04, y=0.96, text=text),
                    legend=pb.LegendConfig(location="upper left", anchor=(0.02, 0.79)),
                ),
                # Ratio.
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$\Delta R$"),
                        pb.AxisConfig("y", label=ratio_label, range=(0, 2)),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12}),
        ),
        output_dir=_output_dir,
        plot_png=plot_png,
    )


# -

def comparison_filename(grooming_method: str, variable: str, jet_pt_selection_prefix: str, jet_pt_bin: helpers.JetPtRange, ) -> str:
    # kt_embed_comparison_hybrid_jet_pt_selection_jetPt_40_120_dynamical_kt_true_det_level_hybrid_iterative_splittings.pdf
    return f"{variable}_embed_comparison_{jet_pt_selection_prefix}_jet_pt_selection_{jet_pt_bin}_{grooming_method}_true_det_level_hybrid_iterative_splittings"


# ### Embedded Hybrid vs True vs Det level $k_{\text{T}}$ - Fixed hybrid jet pt

substructure_variable = "kt"
jet_pt_selection_prefix = "hybrid"
_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable=substructure_variable, jet_pt_selection_prefix=jet_pt_selection_prefix, jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, round(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# ### Embedded Hybrid vs True vs Det level $\Delta R$ - Fixed hybrid jet pt

substructure_variable = "delta_R"
jet_pt_selection_prefix = "hybrid"
_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable=substructure_variable, jet_pt_selection_prefix=jet_pt_selection_prefix, jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# ## Fixed True Jet Pt

# + tags=["remove_cell"]
hists = [
    plot_from_skim.PlotHists(hists=hists_embed, prefix="true", identifier="true", display_label="True",),
    plot_from_skim.PlotHists(hists=hists_embed, prefix="det_level", identifier="det_level", display_label="Det. Level",),
    plot_from_skim.PlotHists(hists=hists_embed, prefix="hybrid", identifier="hybrid", display_label="Hybrid",),
]

# Settings
plot_png = True
jet_pt_selection_prefix = "true"

# Settings to be refactored.
jet_pt_bin = helpers.RangeSelector(min=40, max=140)
text = "Iterative splittings"
text += "\n" + fr"${jet_pt_bin.display_str(label=jet_pt_selection_prefix)}\:\text{{GeV}}/c$"
ratio_label = f"Others/{hists[0].display_label}"
tag = jet_pt_bin.histogram_str(jet_pt_selection_prefix)

for grooming_method in _all_available_methods:
    new_plot_comparison.plot_compare_grooming_methods_for_prefix(
        hists=hists,
        grooming_methods=[grooming_method],
        attr_name="kt",
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"kt_embed_comparison_{jet_pt_selection_prefix}_jet_pt_selection",
            panels=[
                # Main axis.
                pb.Panel(
                    axes=pb.AxisConfig(
                        "y",
                        label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                        log=True,
                    ),
                    text=pb.TextConfig(x=0.96, y=0.96, text=text),
                    legend=pb.LegendConfig(location="lower left"),
                ),
                # Ratio.
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig("y", label=ratio_label, range=(-0.2, 4)),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12}),
        ),
        output_dir=_output_dir,
        plot_png=plot_png,
    )
    new_plot_comparison.plot_compare_grooming_methods_for_prefix(
        hists=hists,
        grooming_methods=[grooming_method],
        attr_name="delta_R",
        tag=tag,
        jet_pt_bin=jet_pt_bin,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"delta_R_embed_comparison_{jet_pt_selection_prefix}_jet_pt_selection",
            panels=[
                # Main axis.
                # NOTE: This intentionally cuts off the normalization bin
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", range=(0, 0.41)),
                        pb.AxisConfig(
                            "y", label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\Delta R$", range=(-0.4, 19.1)
                        ),
                    ],
                    text=pb.TextConfig(x=0.04, y=0.96, text=text),
                    legend=pb.LegendConfig(location="upper left", anchor=(0.02, 0.79)),
                ),
                # Ratio.
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$\Delta R$"),
                        pb.AxisConfig("y", label=ratio_label, range=(0, 2)),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.12}),
        ),
        output_dir=_output_dir,
        plot_png=plot_png,
    )


# + tags=["remove_cell"]
def comparison_filename(grooming_method: str, variable: str, jet_pt_selection_prefix: str, jet_pt_bin: helpers.JetPtRange, ) -> str:
    # kt_embed_comparison_hybrid_jet_pt_selection_jetPt_40_120_dynamical_kt_true_det_level_hybrid_iterative_splittings.pdf
    return f"{variable}_embed_comparison_{jet_pt_selection_prefix}_jet_pt_selection_{jet_pt_bin}_{grooming_method}_true_det_level_hybrid_iterative_splittings"


# -

# ### Embedded Hybrid vs True vs Det level $k_{\text{T}}$ - Fixed true jet pt

substructure_variable = "kt"
jet_pt_selection_prefix = "true"
_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable=substructure_variable, jet_pt_selection_prefix=jet_pt_selection_prefix, jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# ### Embedded Hybrid vs True vs Det level $\Delta R$ - Fixed true jet pt

substructure_variable = "delta_R"
jet_pt_selection_prefix = "true"
_comparisons = []
for grooming_method in _all_available_methods:
    _comparisons.append(comparison_filename(grooming_method=grooming_method, variable=substructure_variable, jet_pt_selection_prefix=jet_pt_selection_prefix, jet_pt_bin=jet_pt_bin))
nb_utils.display_images([
    _comparisons[3*i:3*(i+1)] for i in range(0, int(len(_comparisons)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)

# ## Hybrid vs True vs Det Level Summary
#
# It seems much better to judge this based on a fixed true jet pt selection. Some observations for this case:
#
# - In general, the trends seem rather similar to semi-central R = 0.4.
#
# For $k_{\text{T}}$,
#
# - Depletion at high $k_{\text{T}}$ seems to be consistent until it runs out of stats. This is presumably due to lost subjets.
#
# For $\Delta R$,
#
# - Most are pretty consistent, although I think part of this is due to the small deltaR range. Some bigger differences for high z cut, but I think it's fine

# # Subjet Matching

# ## Prong Matching

# +
# First, any required imports for embedding
from jet_substructure.base import skim_analysis_objects

# Next, any helper functions
def grooming_name(grooming_method: str, prefixes: Sequence[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "embed_pythia"
#prefix = "hybrid"
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
#text_label += "\n" + f"${jet_pt_bin.display_str(label=prefix)}$"
#tag = jet_pt_bin.histogram_str(prefix)
system_label = pb.label_to_display_string["collision_system"][collision_system].format(main_system=r"30-50\%")

# Setup
#_all_available_methods = list(nb_utils.all_grooming_methods)
_method_groups = [
    #nb_utils.leading_kt_grooming_methods,
    nb_utils.dynamical_grooming_methods,
    nb_utils.soft_drop_grooming_methods,
    nb_utils.dynamical_grooming_with_z_cut,
    nb_utils.z_cut_02_grooming_methods,
    #_all_available_methods,
]
# We want to skip the leading_kt methods, but not redefine all_grooming_methods in nb_utils,
# so we combine all of the above
_all_available_methods = [_method for _group in _method_groups for _method in _group]
# Add the add the all methods to the method groups
_method_groups.append(_all_available_methods)

# Helpers for plotting responses
_matching_name_to_axis_value: Dict[str, int] = {
    "all": 0,
    "pure": 1,
    "leading_untagged_subleading_correct": 2,
    "leading_correct_subleading_untagged": 3,
    "leading_correct_subleading_mistag": 4,
    "leading_mistag_subleading_correct": 5,
    "leading_untagged_subleading_mistag": 6,
    "leading_mistag_subleading_untagged": 7,
    "swap": 8,
    "both_untagged": 9,
}
_response_types = [
    skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="det_level"),
    skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="true"),
    skim_analysis_objects.ResponseType(measured_like="det_level", generator_like="true"),
]

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "2023-02-HP" / "semi_central_R02" / "pass3"
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_embedding = {}
_successfully_loaded_methods = []
#for dataset_name in ["LHC19f4_embedded_into_LHC18qr_central_6338_6357", "LHC19f4_embedded_into_LHC18qr_central_R02_6456_6475"]:
#for dataset_name in ["LHC20g4_embedded_into_LHC18qr_semi_central_R02_6932_6951"]:
for dataset_name in ["LHC20g4_embedded_into_LHC18qr_semi_central_R02_0067"]:
    for grooming_method in nb_utils.all_grooming_methods:
        try:
            hists_embedding.update(nb_utils.load_histograms(
                filename=f"{dataset_name}_{grooming_name(grooming_method, prefixes=['hybrid', 'true', 'det_level'])}_response.root",
                collision_system=collision_system,
                tag="RDF",
                base_path=Path("output")
            ))
            _successfully_loaded_methods.append(grooming_method)
        except FileNotFoundError:
            logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_embedding = {k: v.to_boost_histogram() for k, v in hists_embedding.items()}
# -

logger.info(_successfully_loaded_methods)

# + jupyter={"outputs_hidden": true} tags=["remove_cell"]
plot_from_skim.plot_prong_matching(
    hists=hists_embedding,
    grooming_methods=_all_available_methods,
    matching_types=list(_matching_name_to_axis_value.keys()),
    output_dir=_output_dir,
    rdf_plots=True,
    plot_png=False,
    min_kt_hybrid_values=[-1, 1, 1.5, 2],
    system_label=system_label,
)


# -

def comparison_filename(matching_level: str, grooming_method: str, prefix: str, jet_pt_bin: helpers.JetPtRange, min_kt_hybrid: float = -1) -> str:
    suffix = f"_min_kt_hybrid_{min_kt_hybrid}" if min_kt_hybrid > 0 else ""
    #return f"subjet_matching_hybrid_det_level_pt_hybrid_jetPt_40_120_{grooming_method}_single_figure_jet_pt_hybrid_40_120"
    return f"subjet_matching_{matching_level}_pt_{grooming_method}_single_figure_{jet_pt_bin.histogram_str(label=prefix)}{suffix}"


# ### Hybrid <-> Detector Level Prong Matching, No $k_{\text{T}}^{\text{hybrid}}$ cut

# +
_matching_summaries = []
_matching_level = "hybrid_det_level"
for grooming_method in _all_available_methods:
    _matching_summaries.append(
        comparison_filename(matching_level=_matching_level, grooming_method=grooming_method, prefix="hybrid", jet_pt_bin=jet_pt_bin)
    )

nb_utils.display_images([
    _matching_summaries[3*i:3*(i+1)] for i in range(0, round(len(_matching_summaries)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Hybrid <-> Detector Level Prong Matching, $k_{\text{T}}^{\text{hybrid}} > 1$ GeV

# + tags=["remove_cell"]
_matching_summaries = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 1
for grooming_method in _all_available_methods:
    _matching_summaries.append(
        comparison_filename(matching_level=_matching_level, grooming_method=grooming_method, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _matching_summaries[3*i:3*(i+1)] for i in range(0, round(len(_matching_summaries)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Hybrid <-> Detector Level Prong Matching, $k_{\text{T}}^{\text{hybrid}} > 1.5$ GeV

# +
_matching_summaries = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 1.5
for grooming_method in _all_available_methods:
    _matching_summaries.append(
        comparison_filename(matching_level=_matching_level, grooming_method=grooming_method, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _matching_summaries[3*i:3*(i+1)] for i in range(0, round(len(_matching_summaries)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Hybrid <-> Detector Level Prong Matching, $k_{\text{T}}^{\text{hybrid}} > 2$ GeV

# +
_matching_summaries = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 2
for grooming_method in _all_available_methods:
    _matching_summaries.append(
        comparison_filename(matching_level=_matching_level, grooming_method=grooming_method, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _matching_summaries[3*i:3*(i+1)] for i in range(0, round(len(_matching_summaries)/3))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Hybrid <-> Detector Level Prong Matching, Comparison by Grooming Method for $k_{\text{T}}^{\text{hybrid}}$ cuts
#
# Here, we'll consider one grooming method per row, with increasing $k_{\text{T}}^{\text{hybrid}}$ when going from left to right.

# +
_matching_summaries = []
_matching_level = "hybrid_det_level"
_min_kt_hybrids = [-1, 1, 1.5, 2]
for grooming_method in _all_available_methods:
    for _min_kt_hybrid in _min_kt_hybrids:
        _matching_summaries.append(
            comparison_filename(matching_level=_matching_level, grooming_method=grooming_method, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
        )

nb_utils.display_images([
    _matching_summaries[len(_min_kt_hybrids)*i:len(_min_kt_hybrids)*(i+1)] for i in range(0, int(len(_matching_summaries)/len(_min_kt_hybrids)))
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# Increasing the kt cut increases the number of swaps in almost all cases. Purity is pretty consistently on the order of 80-90%

# ## Subjet Purity

# + jupyter={"outputs_hidden": true} tags=["remove_cell"]
for _temp_grooming_methods in _method_groups:
    plot_from_skim.plot_prong_matching_purity(
        hists=hists_embedding,
        grooming_methods=_temp_grooming_methods,
        all_available_methods=_all_available_methods,
        output_dir=_output_dir,
        plot_png=False,
        min_kt_hybrid_values=[-1, 1, 1.5, 2],
    )


# -

# ### Leading Subjet Purity

def comparison_filename(subjet_for_purity: str, matching_level: str, grooming_methods: Sequence[str], prefix: str, jet_pt_bin: helpers.JetPtRange, min_kt_hybrid: float = -1) -> str:
    suffix = f"_min_kt_hybrid_{min_kt_hybrid}" if min_kt_hybrid > 0 else ""
    #return f"subjet_matching_hybrid_det_level_pt_hybrid_jetPt_40_120_{grooming_method}_single_figure_jet_pt_hybrid_40_120"
    #return subjet_matching_subleading_purity_hybrid_det_level_leading_kt_leading_kt_z_cut_02_leading_kt_z_cut_04_dynamical_z_dynamical_kt_dynamical_time_soft_drop_z_cut_02_soft_drop_z_cut_04_jet_pt_hybrid_40_120_min_kt_hybrid_2.pdf
    grooming_methods_label = "_".join(grooming_methods)
    return f"subjet_matching_{subjet_for_purity}_purity_{matching_level}_{grooming_methods_label}_{jet_pt_bin.histogram_str(label=prefix)}{suffix}"


# ### Leading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > -1$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = -1
_subjet_for_purity = "leading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Leading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > 1$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 1
_subjet_for_purity = "leading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Leading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > 1.5$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 1.5
_subjet_for_purity = "leading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Leading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > 2$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 2
_subjet_for_purity = "leading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Subleading Subjet Purity

# ### Subleading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > -1$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = -1
_subjet_for_purity = "subleading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Subleading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > 1$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 1
_subjet_for_purity = "subleading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Subleading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > 1.5$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 1.5
_subjet_for_purity = "subleading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# ### Subleading Subjet Purity, $k_{\text{T}}^{\text{hybrid}} > 2$ GeV

# +
_purity_plots = []
_matching_level = "hybrid_det_level"
_min_kt_hybrid = 2
_subjet_for_purity = "subleading"
for grooming_methods in _method_groups:
    _purity_plots.append(
        comparison_filename(subjet_for_purity=_subjet_for_purity, matching_level=_matching_level, grooming_methods=grooming_methods, prefix="hybrid", jet_pt_bin=jet_pt_bin, min_kt_hybrid=_min_kt_hybrid)
    )

nb_utils.display_images([
    _purity_plots,
], fig_output_dir=_fig_output_dir, embed_with_base64=embed_images)
# -

# Leading purity looks fine. For subleading purity, kt > 1, we get subleading purity ~ 90%, and everything except for $z > 0.4$ looks tenable. Higher kt cuts don't seem to do any better

# # Residuals and Response

# +
# First, any required imports for embedding
from jet_substructure.base import skim_analysis_objects

# Next, any helper functions
def grooming_name(grooming_method: str, prefixes: Sequence[str]) -> str:
    return f"{grooming_method}_prefixes_{'_'.join(prefixes)}"

# Settings
collision_system = "embedPythia"
#prefix = "hybrid"
jet_pt_bin = helpers.JetPtRange(min=40, max=120)
text_label = "Iterative splittings"
#text_label += "\n" + f"${jet_pt_bin.display_str(label=prefix)}$"
#tag = jet_pt_bin.histogram_str(prefix)

# Setup
_all_available_methods = list(nb_utils.all_grooming_methods)
_method_groups = [
    nb_utils.leading_kt_grooming_methods,
    nb_utils.dynamical_grooming_methods,
    nb_utils.soft_drop_grooming_methods,
    _all_available_methods,
]

# Helpers for plotting responses
_matching_name_to_axis_value: Dict[str, int] = {
    "all": 0,
    #"pure": 1,
    #"leading_untagged_subleading_correct": 2,
    #"leading_correct_subleading_untagged": 3,
    #"leading_correct_subleading_mistag": 4,
    #"leading_mistag_subleading_correct": 5,
    #"leading_untagged_subleading_mistag": 6,
    #"leading_mistag_subleading_untagged": 7,
    #"swap": 8,
    #"both_untagged": 9,
}
_response_types = [
    #skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="det_level"),
    skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="true"),
    #skim_analysis_objects.ResponseType(measured_like="det_level", generator_like="true"),
]

# Output
_output_dir = output_dir / collision_system / "RDF" / "jupyter" / "semi_central_R02" / "pass3"
_output_dir.mkdir(parents=True, exist_ok=True)
_fig_output_dir = _output_dir / "png"

# Load data for comparison
hists_embedding = {}
_successfully_loaded_methods = []
#for dataset_name in ["LHC19f4_embedded_into_LHC18qr_central_6338_6357", "LHC19f4_embedded_into_LHC18qr_central_R02_6456_6475"]:
for dataset_name in ["LHC20g4_embedded_into_LHC18qr_semi_central_R02_6932_6951"]:
    for grooming_method in nb_utils.all_grooming_methods:
        try:
            hists_embedding.update(nb_utils.load_histograms(
                filename = f"{dataset_name}_{grooming_name(grooming_method, prefixes=['hybrid', 'true', 'det_level'])}_response.root", collision_system=collision_system,
                tag = "RDF", base_path = Path("output")
            ))
            _successfully_loaded_methods.append(grooming_method)
        except FileNotFoundError:
            logger.info(f"Skipping grooming method {grooming_method} because the output file isn't available")
        
# Convert to boost_histogram because it simplifies projections later.
hists_embedding = {k: v.to_boost_histogram() for k, v in hists_embedding.items()}
# -

logger.info(_successfully_loaded_methods)



from importlib import reload
reload(plot_from_skim)

# + jupyter={"outputs_hidden": true}
plot_from_skim.plot_response_by_matching_type(
    hists=hists_embedding,
    grooming_methods=_all_available_methods,
    matching_types=list(_matching_name_to_axis_value.keys()),
    response_types=_response_types,
    output_dir=_output_dir,
    rdf_plots=True,
    plot_png=True,
)


# -

def comparison_filename(matching_level: str, grooming_method: str, prefix: str, jet_pt_bin: helpers.JetPtRange, min_kt_hybrid: float = -1) -> str:
    suffix = f"_min_kt_hybrid_{min_kt_hybrid}" if min_kt_hybrid > 0 else ""
    #return f"subjet_matching_hybrid_det_level_pt_hybrid_jetPt_40_120_{grooming_method}_single_figure_jet_pt_hybrid_40_120"
    return f"subjet_matching_{matching_level}_pt_{grooming_method}_single_figure_{jet_pt_bin.histogram_str(label=prefix)}{suffix}"


from importlib import reload
reload(plot_from_skim)

_matching_name_to_axis_value: Dict[str, int] = {
    "all": 0,
    "pure": 1,
    "leading_untagged_subleading_correct": 2,
    "leading_correct_subleading_untagged": 3,
    "leading_correct_subleading_mistag": 4,
    "leading_mistag_subleading_correct": 5,
    "leading_untagged_subleading_mistag": 6,
    "leading_mistag_subleading_untagged": 7,
    "swap": 8,
    "both_untagged": 9,
}

# + jupyter={"outputs_hidden": true}
plot_from_skim.plot_subjet_momentum_fraction_in_hybrid(
    hists=hists_embedding,
    grooming_methods=_all_available_methods,
    matching_types=list(_matching_name_to_axis_value.keys()),
    output_dir=_output_dir,
    rdf_plots=True,
    plot_png=True,
)

# + jupyter={"outputs_hidden": true}
plot_from_skim.plot_residuals(
    hists=hists_embedding,
    grooming_methods=_all_available_methods,
    #matching_types=list(_matching_name_to_axis_value.keys()),
    #response_types=_response_types,
    output_dir=_output_dir,
    rdf_plots=True,
    plot_png=True,
)
# -


