""" Basic shared functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import itertools
import logging
import typing
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

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

    def __add__(self, other: int) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def __mul__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __rmul__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __truediv__(self, other: Union[float, UprootArray[T]]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __pow__(self, p: float) -> UprootArray[T]:
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
        ...

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


def setup_logging() -> None:
    # Delayed import since we may not want this as a hard dependency in such a base module.
    import coloredlogs

    # Basic setup
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    # Quiet down BinndData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)


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


def dict_product(input_dict: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """ Like `itertools.product`, but with a dictionary containing lists.

    By way of example:

    >>> list(product_dict({"a": [1, 2], "b": [3], "c": [4, 5]}))
    [{'a': 1, 'b': 3, 'c': 4}, {'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 4}, {'a': 2, 'b': 3, 'c': 5}]

    It will give us all possible combinations of the list values with their associated keys.

    From: https://stackoverflow.com/a/40623158/12907985

    Args:
        kwargs: Dictionary for the product.
    Returns:
        Product of the dict keys and values.
    """
    return (dict(zip(input_dict.keys(), values)) for values in itertools.product(*input_dict.values()))


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


def expand_wildcards_in_filenames(paths: Sequence[Path]) -> List[Path]:
    return_paths: List[Path] = []
    for path in paths:
        p = str(path)
        if "*" in str(p):
            # Glob all associated filenames.
            return_paths.extend(list(Path(path.parent).glob(path.name)))
        else:
            return_paths.append(path)

    # Sort in the expected order (just according to alphabetical, which should handle numbers
    # fine as long as they have leading 0s (ie. 03 instead of 3)).
    return_paths = sorted(return_paths, key=lambda p: str(p))
    return return_paths


def split_tree(
    filenames: Sequence[Union[str, Path]],
    tree_name: str = "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
    number_of_chunks: int = -1,
) -> Dict[Path, List[Path]]:
    """ Split tree into a given number of chunks.

    It will also skip storing bad entries in the new files.

    Note:
        To only repair the file, use only one chunk.

    Note:
        Even if we are chunking the file, this method will still try to avoid storing bad entries
        in the new files.

    Args:
        filenames: Name(s) of the file to split.
        tree_name: Name of the tree to split. Default: "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
        number_of_chunks: Number of chunks to split the file into. Pass -1 to auto calculate the number of
            chunks (ie keeping each file before 1.25 GB). Default: -1.

    Returns:
        Filenames of the chunked files.
    """
    # Validation
    validated_filenames = expand_wildcards_in_filenames([Path(f) for f in filenames])
    # Skip files which were already repaired.
    validated_filenames = [f for f in validated_filenames if "repaired" not in str(f) and "chunk" not in str(f)]
    # Automatically calculate the number of chunks, with the intention of keeping all
    # files below (approximately) 1.25 GB.
    auto_calculate_number_of_chunks = False
    if number_of_chunks < 0:
        logger.info("Automatically calculating chunk size")
        auto_calculate_number_of_chunks = True

    # Setup
    # Delayed import since we may not want this as a hard dependency in such a base module.
    import enlighten

    # Delayed import because we want to depend on ROOT as little as possible.
    import ROOT

    # Just in case we enable multithreading later.
    ROOT.ROOT.EnableImplicitMT()
    # Finish setup
    output_filenames: Dict[Path, List[Path]] = {}
    progress_manager = enlighten.get_manager()

    with progress_manager.counter(total=len(validated_filenames), desc="Processing", unit="file") as file_counter:
        for filename in file_counter(validated_filenames):
            # Determine number of chunks if requested.
            if auto_calculate_number_of_chunks:
                size = Path(filename).stat().st_size
                # Close enough to 1.25 GB
                number_of_chunks = int(np.ceil(size / 1.25e9))

            # Setup input tree
            input_file = ROOT.TFile(str(filename), "READ")
            logger.debug(f"Keys in input_file: {list(input_file.GetListOfKeys())}")
            input_tree = input_file.Get(tree_name)

            number_of_entires = input_tree.GetEntries()
            logger.info(
                f"File: {filename}: Total of {number_of_entires} in the tree. Splitting into {number_of_chunks} chunk(s)."
            )

            output_filenames[filename] = []
            for n in range(number_of_chunks):
                start = int((number_of_entires / number_of_chunks) * n)
                end = int((number_of_entires / number_of_chunks) * (n + 1))

                # If we have only 1 chunk, then we're just trying to repair the file.
                if number_of_chunks == 1:
                    new_filename = filename.with_name(f"{filename.stem}.repaired.root")
                else:
                    new_filename = filename.with_name(f"{filename.stem}.chunk{n+1}.root")
                output_filenames[filename].append(new_filename)
                new_file = ROOT.TFile(str(new_filename), "RECREATE")
                new_tree = input_tree.CloneTree(0)
                ROOT.gROOT.cd()

                logger.info(f"Fill tree {new_filename} with entries {start}-{end}")
                with progress_manager.counter(
                    total=end - start, desc="Converting", unit="event", leave=False
                ) as event_counter:
                    for i in event_counter(range(start, end)):
                        ret_val = input_tree.GetEntry(i)
                        if ret_val < 0:
                            # Skip this entry - something is wrong with it! (Probably a compression error).
                            # This shouldn't happen _too_ often, so we may as well log when it does.
                            logger.debug(
                                f"Skipping entry {i}, as it appears to be bad. GetEntry return value < 0: {ret_val}"
                            )
                            continue
                        new_tree.Fill()

                new_tree.AutoSave()
                new_file.Close()

    progress_manager.stop()
    return output_filenames


def split_tree_entry_point() -> None:
    """ Entry point for splitting a tree into chunks.

    Args:
        None. It can be configured through command line arguments.

    Returns:
        None.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description=f"Split tree into chunks.")

    parser.add_argument("-f", "--filenames", required=True, nargs="+", default=[])
    parser.add_argument(
        "-t",
        "--treeName",
        default="AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        type=str,
    )
    parser.add_argument("-n", "--nChunks", default=-1, type=int)
    args = parser.parse_args()

    output_filenames = split_tree(filenames=args.filenames, tree_name=args.treeName, number_of_chunks=args.nChunks)

    import pprint

    logger.info(f"File output: {pprint.pformat(output_filenames)}")
