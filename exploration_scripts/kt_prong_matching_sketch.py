""" Sketches for INT workshop in August 2021.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pachyderm.plot

# Since I want a sketch, I don't really want the proper pachyderm style here.
# However, I want the latex support, etc. So I first configure, and then I'll
# apply some seaborn styling

pachyderm.plot.configure()

def plot_subleading_subjet_purity_sketch() -> None:
    sns.set_style("white")
    sns.despine()
    x = np.linspace(0.5, 150.5, 150)
    with sns.color_palette("Set2"):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(x, 2/np.pi*np.arctan(x/50), label=r"$k_{\text{T}}$ inclusive", linewidth=2)
        ax.plot(x, 2/np.pi*np.arctan(x/10), label=r"$k_{\text{T}} >$ X GeV", linewidth=2)

        ax.annotate("", xy=(30, 0.75), xytext=(30, 0.45), xycoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=5))

    ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax.set_ylabel("Subleading subjet purity", fontsize=22)
    ax.set_ylim([-0.03, 1.2])
    ax.set_xlabel(r"$p_{\text{T,jet}}$ (GeV/$c$)", fontsize=22)
    ax.legend(frameon=False, loc="lower right", fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    adjust_default_args = dict(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        #left=0.12,
        bottom=0.16,
        right=0.98,
        top=0.98,
    )
    fig.subplots_adjust(**adjust_default_args)
    fig.savefig("subleading_subjet_purity_sketch.pdf")


def plot_kinematic_efficiency_sketch() -> None:
    sns.set_style("white")
    sns.despine()
    x = np.linspace(0, 12, 120)
    with sns.color_palette("Set2"):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(x, 2.25/np.pi*np.arctan(x/1) * (1-np.exp(x - 10)), label=r"$k_{\text{T}}$ inclusive", linewidth=2)
        ax.plot(x, 2.25/np.pi*np.arctan(x/2.25) * (1-np.exp(x - 10)), label=r"$k_{\text{T}} >$ Y GeV", linewidth=2)

        ax.annotate("", xy=(3, 0.7), xytext=(3, 0.85), xycoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=5))

    ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax.set_ylabel("Kinematic Eff.", fontsize=22)
    ax.set_ylim([-0.03, 1.2])
    ax.set_xlabel(r"$k_{\text{T}}$ (GeV/$c$)", fontsize=22)
    ax.legend(frameon=False, loc="lower center", fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    adjust_default_args = dict(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        #left=0.12,
        bottom=0.16,
        right=0.98,
        top=0.98,
    )
    fig.subplots_adjust(**adjust_default_args)
    fig.savefig("kinematic_efficiency_sketch.pdf")


if __name__ == "__main__":
    plot_subleading_subjet_purity_sketch()
    plot_kinematic_efficiency_sketch()
