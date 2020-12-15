"""Rename unfolding outputs to more descriptive names.

This is all pretty ad-hoc, but I needed some more control of renaming them.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import pprint
import shutil
from pathlib import Path

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def rename_semi_central(grooming_method: str, debug: bool) -> bool:
    input_dir = Path("output") / "PbPb" / "unfolding" / "parsl" / "bak_rename"
    input_files = input_dir.glob(f"*{grooming_method}*")

    rename_map = {}

    new_tag = "semi_central_R04"
    for original_filename in input_files:
        new_filename = str(original_filename.name).replace("broadTrueBins", new_tag)
        extra_tag = ""
        if new_filename.count(new_tag) == 0:
            extra_tag = f"{new_tag}_"
        new_filename = (
            new_filename.replace("Rmax06", "Rmax060")
            # To ensure that the label gets in, we need to rename including other terms
            .replace("truncation", f"{extra_tag}truncation")
            .replace("TrackingEff", f"{extra_tag}tracking_efficiency")
            .replace("reweight_prior", f"{extra_tag}reweight_prior")
            .replace("random_binning", f"{extra_tag}random_binning")
        )
        # Skip central
        if new_tag not in new_filename and "central" in new_filename:
            logger.info(f"Skipping central file {new_filename}")
            continue
        assert new_tag in new_filename, new_filename
        assert new_filename.count(new_tag) == 1, new_filename
        rename_map[original_filename] = original_filename.parent.parent / new_filename

    if not debug:
        # Perform the actual copy
        for source, dest in rename_map.items():
            shutil.copy(source, dest)
    else:
        pprint.pprint(rename_map)

    return True


def rename_central(grooming_method: str, debug: bool) -> bool:
    input_dir = Path("output") / "PbPb" / "unfolding" / "parsl" / "bak_rename"
    input_files = input_dir.glob(f"*{grooming_method}*")

    rename_map = {}

    new_tag = "central_R02"
    for original_filename in input_files:
        if not ("2_6" in str(original_filename) or "2_8" in str(original_filename)):
            continue

        assert "central" in str(original_filename)

        new_filename = (
            str(original_filename.name)
            .replace("central", new_tag)
            .replace("Rmax06", "Rmax060")
            # To ensure that the label gets in, we need to rename including other terms
            .replace("truncation", "truncation")
            .replace("TrackingEff", "tracking_efficiency")
        )
        assert new_tag in new_filename, new_filename
        assert new_filename.count(new_tag) == 1, new_filename
        rename_map[original_filename] = original_filename.parent.parent / new_filename

    if not debug:
        # Perform the actual copy
        for source, dest in rename_map.items():
            shutil.copy(source, dest)
    else:
        pprint.pprint(rename_map)

    return True


if __name__ == "__main__":
    debug = False
    helpers.setup_logging()
    # semi-central
    rename_semi_central("dynamical_z", debug=debug)
    rename_semi_central("dynamical_kt", debug=debug)
    rename_semi_central("dynamical_time", debug=debug)
    rename_semi_central("leading_kt", debug=debug)
    # central
    rename_central("leading_kt", debug=debug)
