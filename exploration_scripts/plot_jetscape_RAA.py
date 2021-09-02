#!/usr/bin/env python3

"""Plot jetscape jet RAA predictions

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import hist
import matplotlib.pyplot as plt
import pachyderm.plot
import seaborn as sns
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb


pachyderm.plot.configure()

def format_R(R: float) -> str:
    return f"{round(R * 100):03}"


def get_hists(filename: Path) -> Dict[str, hist.Hist]:
    hists = {}
    with uproot.open(Path(filename)) as f:
        for k in f.keys(cycle=False):
            hists[k] = f[k].to_hist()

    return hists


def combine_spectra_in_cent_bins(hists: Mapping[str, hist.Hist], jet_type: str, jet_R: float, a: str, b: str) -> hist.Hist:
    name = f"{jet_type}_jetR{format_R(jet_R)}_n_events"

    a_n_events = hists[f"PbPb_{a}"][name]
    b_n_events = hists[f"PbPb_{b}"][name]
    name = f"{jet_type}_jetR{format_R(jet_R)}_jet_pt"
    a_jet_pt = hists[f"PbPb_{a}"][name]
    b_jet_pt = hists[f"PbPb_{b}"][name]

    return ((a_jet_pt / a_n_events.counts()[0]) + (b_jet_pt / b_n_events.counts()[0])) / 2


def plot(output_dir: Path,
         jet_R_values: Optional[Sequence[float]] = None,
         jet_types: Optional[Sequence[str]] = None,
         ) -> None:
    # Validation
    if jet_R_values is None:
        jet_R_values = [0.2, 0.4, 0.6]
    if jet_types is None:
        jet_types = ["charged", "full"]
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)

    hists = {}
    hists["pp"] = get_hists(filename=Path("jetscape_RAA_output/pp/jetscape_RAA.root"))
    hists["PbPb_00_05"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_00_05.root"))
    hists["PbPb_05_10"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_05_10.root"))
    hists["PbPb_30_40"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_30_40.root"))
    hists["PbPb_40_50"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_40_50.root"))

    labels = {
        "pp": "pp",
        "PbPb_00_05": r"0-5\% Pb-Pb",
        "PbPb_05_10": r"5-10\% Pb-Pb",
        "PbPb_30_40": r"30-40\% Pb-Pb",
        "PbPb_40_50": r"40-50\% Pb-Pb",
        # Derived hists
        "PbPb_00_10": r"0-10\% Pb-Pb",
        "PbPb_30_50": r"30-50\% Pb-Pb",
    }

    RAA_hists = {
        "PbPb_00_10": {},
        "PbPb_30_50": {},
    }

    with sns.color_palette("Set2"):
        for jet_R in jet_R_values:
            for jet_type in jet_types:
                fig, ax = plt.subplots(figsize=(10, 8))
                fig_scaled, ax_scaled = plt.subplots(figsize=(10, 8))
                fig_RAA, ax_RAA = plt.subplots(figsize=(10, 8))

                text = fr"{jet_type.capitalize()} jets, $R$ = {jet_R}"
                # Just for some user feedback
                print(text)

                # Finish labeling
                text += "\n" + r"JETSCAPE MATTER + LBT"
                text += "\n" + r"$\alpha_{s} = 0.3$, $Q_{\text{switch}} = 2$ GeV"

                plot_config = pb.PlotConfig(
                    name=f"jet_pt_{jet_type}_R{format_R(jet_R)}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                                    log=True,
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.03, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
                )
                plot_config_scaled = pb.PlotConfig(
                    name=f"jet_pt_{jet_type}_R{format_R(jet_R)}_scaled",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                                    log=True,
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.03, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
                )
                plot_config_RAA = pb.PlotConfig(
                    name=f"jet_RAA_{jet_type}_R{format_R(jet_R)}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$R_{\text{AA}}$",
                                    range=(0.0, 1.4),
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T,jet}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
                )

                # Get scaled pp ref for RAA
                name = f"{jet_type}_jetR{format_R(jet_R)}_n_events"
                h_pp_ref_n_events = hists["pp"][name]
                name = f"{jet_type}_jetR{format_R(jet_R)}_jet_pt"
                # Scale immediately, since we're going to do it anyway
                h_pp_ref_jet_pt = hists["pp"][name] / h_pp_ref_n_events.counts()[0]

                for system, v in hists.items():
                    name = f"{jet_type}_jetR{format_R(jet_R)}_n_events"
                    h_n_events = v[name]
                    name = f"{jet_type}_jetR{format_R(jet_R)}_jet_pt"
                    h_jet_pt = v[name]

                    (h_jet_pt[::hist.rebin(5)] / 5).plot(ax=ax, label=labels[system], linewidth=2)

                    h_jet_pt_scaled = h_jet_pt / h_n_events.counts()[0]
                    (h_jet_pt_scaled[::hist.rebin(5)] / 5).plot(ax=ax_scaled, label=labels[system], linewidth=2)

                    # Skip for now just to reduce the number of curves
                    if system != "pp" and False:
                        # why da faq doesn't this work...
                        #h_RAA = h_pp_ref_jet_pt / h_jet_pt_scaled
                        #(h_RAA[::hist.rebin(5)] / 5).plot(ax=ax_RAA, label=labels[system], linewidth=2)
                        # Some bug somewhere. Not my problem... Back to binned_data...
                        h_RAA = (
                            binned_data.BinnedData.from_existing_data((h_jet_pt_scaled[::hist.rebin(5)] / 5))
                            / binned_data.BinnedData.from_existing_data((h_pp_ref_jet_pt[::hist.rebin(5)] / 5))
                        )
                        ax_RAA.errorbar(
                            h_RAA.axes[0].bin_centers,
                            h_RAA.values,
                            xerr=h_RAA.axes[0].bin_widths / 2,
                            yerr=h_RAA.errors,
                            label=labels[system],
                        )

                # Calculate 0-10%
                h_PbPb_00_10_jet_pt = combine_spectra_in_cent_bins(hists=hists, jet_type=jet_type, jet_R=jet_R, a="00_05", b="05_10")
                (h_PbPb_00_10_jet_pt[::hist.rebin(5)] / 5).plot(ax=ax_scaled, label=labels["PbPb_00_10"], linewidth=2)

                # Calculate 30-50%
                h_PbPb_30_50_jet_pt = combine_spectra_in_cent_bins(hists=hists, jet_type=jet_type, jet_R=jet_R, a="30_40", b="40_50")
                (h_PbPb_30_50_jet_pt[::hist.rebin(5)] / 5).plot(ax=ax_scaled, label=labels["PbPb_30_50"], linewidth=2)

                # Calculate RAA for calculate centralities
                # 0-10%
                #h_RAA = h_pp_ref_jet_pt / h_PbPb_00_10_jet_pt
                #(h_RAA[::hist.rebin(5)] / 5).plot(ax=ax_RAA, label=labels["PbPb_00_10"], linewidth=2)
                h_RAA = (
                    binned_data.BinnedData.from_existing_data((h_PbPb_00_10_jet_pt[::hist.rebin(10)] / 10))
                    / binned_data.BinnedData.from_existing_data((h_pp_ref_jet_pt[::hist.rebin(10)] / 10))
                )
                RAA_hists["PbPb_00_10"][f"{jet_type}_R{format_R(jet_R)}"] = h_RAA
                ax_RAA.errorbar(
                    h_RAA.axes[0].bin_centers,
                    h_RAA.values,
                    xerr=h_RAA.axes[0].bin_widths / 2,
                    yerr=h_RAA.errors,
                    label=labels["PbPb_00_10"],
                )
                # 30-50%
                #h_RAA = h_pp_ref_jet_pt / h_PbPb_30_50_jet_pt
                #(h_RAA[::hist.rebin(5)] / 5).plot(ax=ax_RAA, label=labels["PbPb_30_50"], linewidth=2)
                h_RAA = (
                    binned_data.BinnedData.from_existing_data((h_PbPb_30_50_jet_pt[::hist.rebin(10)] / 10))
                    / binned_data.BinnedData.from_existing_data((h_pp_ref_jet_pt[::hist.rebin(10)] / 10))
                )
                RAA_hists["PbPb_30_50"][f"{jet_type}_R{format_R(jet_R)}"] = h_RAA
                ax_RAA.errorbar(
                    h_RAA.axes[0].bin_centers,
                    h_RAA.values,
                    xerr=h_RAA.axes[0].bin_widths / 2,
                    yerr=h_RAA.errors,
                    label=labels["PbPb_30_50"],
                )

                plot_config.apply(fig=fig, ax=ax)
                filename = f"{plot_config.name}"
                fig.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig)

                plot_config_scaled.apply(fig=fig_scaled, ax=ax_scaled)
                filename = f"{plot_config_scaled.name}"
                fig_scaled.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig_scaled)

                plot_config_RAA.apply(fig=fig_RAA, ax=ax_RAA)
                filename = f"{plot_config_RAA.name}"
                fig_RAA.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig_RAA)

    # Plot RAA as a function of R
    with sns.color_palette("Set2"):
        for jet_type in jet_types:
            for system in ["PbPb_00_10", "PbPb_30_50"]:
                fig, ax = plt.subplots(figsize=(10, 8))

                text = fr"{labels[system]}, {jet_type.capitalize()} jets"
                # Just for some user feedback
                print(text)

                # Finish labeling
                text += "\n" + r"JETSCAPE MATTER + LBT"
                text += "\n" + r"$\alpha_{s} = 0.3$, $Q_{\text{switch}} = 2$ GeV"

                plot_config = pb.PlotConfig(
                    name=f"jet_RAA_R_{jet_type}_{system}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$R_{\text{AA}}$",
                                    range=(0.0, 1.4),
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T,jet}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
                )
                for jet_R in jet_R_values:
                    h_RAA = RAA_hists[system][f"{jet_type}_R{format_R(jet_R)}"]
                    ax.errorbar(
                        h_RAA.axes[0].bin_centers,
                        h_RAA.values,
                        xerr=h_RAA.axes[0].bin_widths / 2,
                        yerr=h_RAA.errors,
                        label=fr"$R$ = {jet_R}",
                    )

                plot_config.apply(fig=fig, ax=ax)
                filename = f"{plot_config.name}"
                fig.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig)


if __name__ == "__main__":
    plot(output_dir=Path("jetscape_RAA_output/plots"))
