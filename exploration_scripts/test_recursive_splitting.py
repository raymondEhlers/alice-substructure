#!/usr/bin/env python3

""" Basic tests for storing recursive splittings.

"""

from typing import Any, Dict, Sequence, Tuple

import pprint
import IPython
import uproot

def pretty_print_tree(d: Dict[int, Any], indent: int = 0) -> None:
    """ Convenience function for pretty printing the splitting tree.

    From: https://stackoverflow.com/a/3229493.

    Args:
        d: Dictionary containing the splittings.
        indent: How far to indent (effectively how far we are into the recursion).

    Returns:
        None.
    """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_tree(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def convert_flat_to_tree(parent_label: int, relationships: Sequence[Tuple[int, int]]) -> Dict[int, Any]:
    """ Convert the flat array to the tree.

    Slightly modified from: https://stackoverflow.com/a/43728268

    Args:
        parent_label: Label of the root parent (usually -1).
        relationships: Relationships from child to parent. Of the form (child index, parent index).
    Returns:
        Tree representing these relationships.
    """
    return { p: convert_flat_to_tree(p, relationships) for p in [index for index, parent in relationships if parent == parent_label]}

def run() -> None:
    f = uproot.open("../temp/AnalysisResults.root")
    t = f["AliAnalysisTaskJetDynamicalGrooming_RawTree_Data_ConstSub_Incl"]
    arrays = t.arrays(namedecode="utf-8")

    #splitting_tree = convert_flat_to_tree(0, list(enumerate(arrays["data.fSubjets.fSplittingNodeIndex"][0])))
    splitting_tree = convert_flat_to_tree(-1, list(enumerate(arrays["data.fJetSplittings.fParentIndex"][0])))
    pretty_print_tree(splitting_tree)

    IPython.embed()

if __name__ == "__main__":
    run()
