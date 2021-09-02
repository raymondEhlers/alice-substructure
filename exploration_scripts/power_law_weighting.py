
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pachyderm.plot

pachyderm.plot.configure()

def power_law(x: float, a: float) -> float:
    return x ** a

def plot() -> None:
    with sns.color_palette("Set2"):
        fig, ax = plt.subplots()

        pt = np.linspace(1, 100, 496)

        ax.plot(pt, power_law(pt, -5), label="Pythia default a=-5")
        ax.plot(pt, power_law(pt, -5) * power_law(pt, 3.7), label="reweighted a=-5 + 3.7")

        ax.set_xlabel(r"$\widehat{p}_{\text{T}}$")
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="upper right")

        fig.tight_layout()
        fig.savefig("power_law.pdf")

if __name__ == "__main__":
    plot()
