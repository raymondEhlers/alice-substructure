
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import attr
import awkward as ak
import h5py
import numpy as np
import pandas as pd
import uproot

# Typing helpers
TTree = Any
UprootArray = Union[np.ndarray, ak.JaggedArray]
UprootArrays = Dict[str, UprootArray]

@attr.s
class RangeSelector:
    min: float = attr.ib()
    max: float = attr.ib()

    def mask_attribute(self, df: Union[pd.DataFrame, UprootArrays], attribute_name: str) -> Union[pd.DataFrame, pd.Series, UprootArrays]:
        """ Create a mask to given attribute to the provided range.

        Args:
            df: Data to be used to define the mask. May be a pandas DataFrame or output from loading arrays
                via uproot.
            attribute_name: Name of the attrbute (column) to be used in the mask.
        Returns:
            Mask of the df for the attribute values within the stored range.
        """
        # Range defined by what is shown in the paper.
        return ((df[attribute_name] >= self.min) & (df[attribute_name] < self.max))

def full_range_extent(selections: Sequence[RangeSelector]) -> RangeSelector:
    """ Extract the min and max range value over all of the selections.

    The DataFrames can be reduced to only contain these values.

    One could be more efficient in the case that there are gaps in the selections, but this
    is sufficient for our purposes.

    Args:
        selections: Selections to be applied.

    Returns:
        Minimum and maximum values.
    """
    return RangeSelector(
        min = min(selections, key=lambda v: v.min).min,
        max = max(selections, key=lambda v: v.max).max,
    )

def get_tree(filename: Path, name: str) -> TTree:
    """ Get the specified tree from a given filename.

    Args:
        filename: Filename of the file containing the tree.
        name: Name of the tree in the file.

    Returns:
        The extract tree.
    """
    # TODO: Cache?
    # Can't use with here because it doesn't fully load the tree here. It just makes it available to be loaded later.
    # If we want to close the file, we need to fully convert to a pandas df immediately. I think even converting to
    # arrays wouldn't be sufficient.
    #with uproot.open(filename) as f:
    f = uproot.open(filename)
    return f[name]

def _concatenate_jagged_array(arrays: Sequence[ak.JaggedArray]) -> ak.JaggedArray:
    """ Concatenate jagged arrays (perhaps from different files).

    Code is from [here](https://github.com/scikit-hep/awkward-array/issues/21#issuecomment-433621906).

    Args:
        arrays: Jagged arrays to be combined.

    Returns:
        Concatenate jagged array.
    """
    contents = np.concatenate([j.flatten() for j in arrays])
    counts = np.concatenate([j.counts for j in arrays])
    return ak.JaggedArray.fromcounts(counts, contents)

def awkward_to_hdf5(trees: Sequence[TTree], array_names: Sequence[str], path: Path, filename: str = "data.h5") -> bool:
    """ Store awkward arrays in HDF5.

    These arrays can come from a list of trees. If so, they will be concatenated together.

    Args:
        trees: Trees or dicts of arrays containing the data of interest.
        array_names: Names of the array keys to be stored.
        path: Path to the HDF5 file.
        filename: HDF5 filename.

    Returns:
        True if the data was written successfully.
    """
    path = path / filename
    with h5py.file(path, "w") as f:
        storage = ak.hdf5(f)
        # Store one array at a time in an attempt to keep memory usage reasonable.
        for name in array_names:
            values = []
            for tree in trees:
                values.append(tree.array([name], namedecode="utf-8"))
            full_array = _concatenate_jagged_array(values)
            storage[name] = full_array

    return True

def hdf5_to_awkward(array_names: Sequence[str], path: Path, filename: str = "data.h5") -> UprootArrays:
    """ Retrieve arrays from an HDF5 array.

    Args:
        array_names: Names of the array keys to be stored.
        path: Path to the HDF5 file.
        filename: HDF5 filename.

    Returns:
        Awkward arrays constructed from stored data.
    """
    data: UprootArrays = {}

    path = path / filename
    with h5py.file(path, "r") as f:
        storage = ak.hdf5(f)
        for array_name in array_names:
            data[array_name] = storage[array_name]

    return data
