
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import attr
import awkward
import numpy as np
import pandas as pd
import uproot

# Typing helpers
TTree = Any
UprootArray = Union[np.ndarray, awkward.JaggedArray]
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
    # Can't use with here because it doesn't fully load the tree here. It just makes it available to be loaded later.
    # If we want to close the file, we need to fully convert to a pandas df immediately. I think even converting to
    # arrays wouldn't be sufficient.
    #with uproot.open(filename) as f:
    f = uproot.open(filename)
    return f[name]

