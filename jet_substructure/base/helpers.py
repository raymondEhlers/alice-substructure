""" Basic shared functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence, Type, Union

import attr
import awkward as ak
import numpy as np


logger = logging.getLogger(__name__)

# Typing helpers
UprootArray = Union[np.ndarray, ak.JaggedArray]
UprootArrays = Mapping[str, UprootArray]
# Arrays = Union[UprootArrays, pd.DataFrame]
# More ideally, I would like:
# It's supposed to carry the semantics of Union[np.ndarray, ak.JaggedArray]
# class NDArray(Generic[T]):
#    def argmax(self) -> T: ...
# class UprootArrayTyped(Generic[T]):
#    def argmax(self) -> NDArray[int]: ...
#    @overload
#    def __getitem__(self, key: NDArray[bool]) -> Union[UprootArrayTyped[T], NDArray[T]]: ...
#    @overload
#    def __getitem__(self, key: NDArray[int]) -> NDArray[T]: ...


@attr.s(frozen=True)
class RangeSelector:
    min: float = attr.ib()
    max: float = attr.ib()

    def mask_attribute(self, df: UprootArrays, attribute_name: str) -> UprootArrays:
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

    def mask_array(self, array: UprootArray) -> UprootArray:
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
