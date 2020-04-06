""" Basic shared functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import logging
import typing
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import attr
import numpy as np


logger = logging.getLogger(__name__)

# Typing helpers
T = TypeVar("T")


class UprootArray(Collection[T]):
    """ Effectively a protocol for the UprootArray type.

    The main advantage is that it allows us to keep track of the types. I don't believe
    that they're closely checked, but if nothing else, they're useful as sanity checks
    for me.

    These definitely _aren't_ comprehensive, but they're a good start.
    """

    @typing.overload
    def __getitem__(self, key: UprootArray[bool]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: UprootArray[int]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: Tuple[slice, slice]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: bool) -> T:
        ...

    @typing.overload
    def __getitem__(self, key: int) -> T:
        ...

    def __getitem__(self, key):  # type: ignore
        raise NotImplementedError("Just typing information.")

    @typing.overload
    def __truediv__(self, other: float) -> UprootArray[T]:
        ...

    @typing.overload
    def __truediv__(self, other: UprootArray[T]) -> UprootArray[T]:
        ...

    def __truediv__(self, other):  # type: ignore
        raise NotImplementedError("Just typing information.")

    def argmax(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def offsets(self) -> np.ndarray:
        raise NotImplementedError("Just typing information.")

    def flatten(self, axis: Optional[int] = ...) -> np.ndarray:
        raise NotImplementedError("Just typing information.")

    @property
    def localindex(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def count_nonzero(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def __lt__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __le__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __gt__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __ge__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __and__(self, other: UprootArray[bool]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __add__(self, other: int) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def __or__(self, other: UprootArray[bool]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __invert__(self) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def pad(self, value: int) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def fillna(self, value: Any) -> UprootArray[T]:
        """ Fill na values with the given values.

        Note:
            This is a bit of a white lie. The types in the array can be a Union[T, Type[value]],
            but including such types makes it a good deal more complicated. So we just don't
            mention the other values. In practice, they will usually be the same type.
        """
        raise NotImplementedError("Just typing information.")

    def ones_like(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")


# Additional typing helpers
ArrayOrScalar = Union[UprootArray[T], T]
UprootArrays = Mapping[str, UprootArray[Any]]


def pretty_print_tree(d: Mapping[int, Any], indent: int = 0) -> None:
    """ Convenience function for pretty printing the splitting tree.

    From: https://stackoverflow.com/a/3229493.

    Args:
        d: Dictionary containing the splittings.
        indent: How far to indent (effectively how far we are into the recursion).

    Returns:
        None.
    """
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, Mapping):
            pretty_print_tree(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


def convert_flat_to_tree(parent_label: int, relationships: Sequence[Tuple[int, int]]) -> Dict[int, Any]:
    """ Convert the flat array to the tree.

    Slightly modified from: https://stackoverflow.com/a/43728268

    Args:
        parent_label: Label of the root parent (usually -1).
        relationships: Relationships from child to parent. Of the form (child index, parent index).
    Returns:
        Tree representing these relationships.
    """
    return {
        p: convert_flat_to_tree(p, relationships)
        for p in [index for index, parent in relationships if parent == parent_label]
    }


@attr.s(frozen=True)
class RangeSelector:
    min: float = attr.ib()
    max: float = attr.ib()

    def mask_attribute(self, df: UprootArrays, attribute_name: str) -> UprootArray[bool]:
        """ Create a mask to given attribute to the provided range.

        Args:
            df: Data to be used to define the mask. May be a pandas DataFrame or output from loading arrays
                via uproot.
            attribute_name: Name of the attrbute (column) to be used in the mask.
        Returns:
            Mask of the df for the attribute values within the stored range.
        """
        # Range defined by what is shown in the paper.
        return self.mask_array(df[attribute_name])

    def mask_array(self, array: UprootArray[T]) -> UprootArray[bool]:
        return (array >= self.min) & (array < self.max)

    @classmethod
    def full_range_over_selections(cls: Type["RangeSelector"], selections: Sequence[RangeSelector]) -> "RangeSelector":
        """ Extract the min and max range value over all of the selections.

        The DataFrames can be reduced to only contain these values.

        One could be more efficient in the case that there are gaps in the selections, but this
        is sufficient for our purposes.

        Args:
            selections: Selections to be applied.

        Returns:
            Minimum and maximum values.
        """
        return cls(min=min(selections, key=lambda v: v.min).min, max=max(selections, key=lambda v: v.max).max,)

    def __str__(self) -> str:
        return f"jetPt_{self.min}_{self.max}"

    def display_str(self, label: str = "") -> str:
        return fr"{self.min} < p_{{\text{{T,jet}}}}^{{\text{{{label}}}}} < {self.max}"


def split_tree(
    filename: Union[str, Path],
    tree_name: str = "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
    number_of_chunks: int = 4,
) -> List[Path]:
    """ Split tree into a given number of chunks.

    Args:
        filename: Name of the file to split.
        tree_name: Name of the tree to split. Default: "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
        number_of_chunks: Number of chunks to split the file into. Default: 5.

    Returns:
        Filenames of the chunked files.
    """
    # Validation
    filename = Path(filename)

    # Setup input tree
    # Delayed import because we want to depend on ROOT as little as possible.
    import ROOT

    input_file = ROOT.TFile(str(filename), "READ")
    input_tree = input_file.Get(tree_name)

    number_of_entires = input_tree.GetEntries()
    print(f"Total of {number_of_entires} in the tree. Splitting into {number_of_chunks} chunks.")

    output_filenames = []
    for n in range(number_of_chunks):
        start = int((number_of_entires / number_of_chunks) * n)
        end = int((number_of_entires / number_of_chunks) * (n + 1))

        new_filename = filename.with_name(f"{filename.stem}.Chunk{n+1}.root")
        output_filenames.append(new_filename)
        new_file = ROOT.TFile(str(new_filename), "RECREATE")
        new_tree = input_tree.CloneTree(0)
        ROOT.gROOT.cd()

        print(f"Fill tree {new_filename} with entries {start}-{end}")
        for i in range(start, end):
            if i % 10000 == 0:
                print(f"Done: {(i-start)/(end-start) * 100:.03g}%")
            input_tree.GetEntry(i)
            new_tree.Fill()

        new_tree.AutoSave()
        new_file.Close()

    return output_filenames


def split_tree_entry_point() -> None:
    """ Entry point for splitting a tree into chunks.

    Args:
        None. It can be configured through command line arguments.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=f"Split tree into chunks.")

    parser.add_argument("-f", "--filename", required=True, type=str)
    parser.add_argument(
        "-t",
        "--treeName",
        default="AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        type=str,
    )
    parser.add_argument("-n", "--nChunks", default=5, type=int)
    args = parser.parse_args()

    split_tree(filename=args.filename, tree_name=args.treeName, number_of_chunks=args.nChunks)
