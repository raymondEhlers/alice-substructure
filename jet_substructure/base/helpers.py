
import logging
from pathlib import Path
from types import TracebackType
from typing import Any, ContextManager, Dict, List, Mapping, Optional, Sequence, Type, TypeVar, Union

import attr
import awkward as ak
import boost_histogram as bh
import h5py
import numpy as np
import pandas as pd
import uproot

from pachyderm import histogram


logger = logging.getLogger(__name__)

# Typing helpers
TTree = Any
UprootArray = Union[np.ndarray, ak.JaggedArray]
UprootArrays = Dict[str, UprootArray]
Arrays = Union[UprootArrays, pd.DataFrame]

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
        return self.mask_array(df[attribute_name])

    def mask_array(self, array: UprootArray) -> UprootArray:
        return ((array >= self.min) & (array < self.max))

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
    # TODO: Perhaps can be replaced with ak.concatenate(arrays)
    contents = np.concatenate([j.flatten() for j in arrays])
    counts = np.concatenate([j.counts for j in arrays])
    return ak.JaggedArray.fromcounts(counts, contents)

def awkward_to_hdf5(trees: Sequence[TTree], dataset_name: str, array_names: Sequence[str], hdf5_file: h5py.File) -> bool:
    """ Store awkward arrays in HDF5.

    These arrays can come from a list of trees. If so, they will be concatenated together.

    Args:
        trees: Trees or dicts of arrays containing the data of interest.
        dataset_name: Name of the dataset (under which it is stored)
        array_names: Names of the array keys to be stored.
        path: Path to the HDF5 file.
        filename: HDF5 filename.

    Returns:
        True if the data was written successfully.
    """
    dataset = hdf5_file.require_group(dataset_name)
    storage = ak.hdf5(dataset)
    # Store one array at a time in an attempt to keep memory usage reasonable.
    for name in array_names:
        logger.debug(f"Extracting {name}")
        values = []
        for tree in trees:
            values.append(tree.array(name))
        #full_array = _concatenate_jagged_array(values)
        # TODO: If this works, remove the dedicated function above.
        full_array = ak.concatenate(values)
        storage[name] = full_array

    return True

def hdf5_to_awkward(hdf5_file: h5py.File, dataset_name: str, array_names: Sequence[str]) -> UprootArrays:
    """ Retrieve arrays from an HDF5 array.

    Args:
        array_names: Names of the array keys to be stored.
        path: Path to the HDF5 file.
        filename: HDF5 filename.

    Returns:
        Awkward arrays constructed from stored data.
    """
    data: UprootArrays = {}

    dataset = hdf5_file[dataset_name]
    storage = ak.hdf5(dataset)
    for array_name in array_names:
        data[array_name] = storage[array_name]

    return data

def _normalize_key(key: str) -> str:
    separator = "_"
    key = key.replace(".f", separator)
    index = key.find(separator) + len(separator)
    # +1 to skip over the latter that's being modified.
    return key[:index] + key[index].lower() + key[index + 1:]


_T = TypeVar("_T")

def normalize_array_names(arrays: Mapping[str, _T]) -> Dict[str, _T]:
    return {_normalize_key(k): v for k, v in arrays.items()}


def _bin_widths(bin_edges: np.ndarray) -> np.ndarray:
    """ Bin widths calculated from the bin edges.

    Args:
        bin_edges: Bin edges.
    Returns:
        Array of the bin widths.
    """
    return bin_edges[1:] - bin_edges[:-1]

def _bin_centers(bin_edges: np.ndarray, bin_widths: np.ndarray) -> np.ndarray:
    """ Bin centers.

    Args:
        bin_edges: Bin edges.
        bin_widths: Bin widths.
    Returns:
        Array of the bin centers.
    """
    half_bin_widths = bin_widths / 2
    return bin_edges[:-1] + half_bin_widths

@attr.s
class BinnedData2D:
    x_bin_edges: np.ndarray = attr.ib()
    y_bin_edges: np.ndarray = attr.ib()
    values: np.ndarray = attr.ib()
    errors_squared: np.ndarray = attr.ib()

    @property
    def x_bin_widths(self) -> np.ndarray:
        """ Bin widths calculated from the bin edges.

        Returns:
            Array of the bin widths.
        """
        return _bin_widths(self.x_bin_edges)

    @property
    def y_bin_widths(self) -> np.ndarray:
        """ Bin widths calculated from the bin edges.

        Returns:
            Array of the bin widths.
        """
        return _bin_widths(self.y_bin_edges)

    @property
    def x(self) -> np.ndarray:
        """ The x bin centers (``x``).

        This property caches the x value so we don't have to calculate it every time.

        Args:
            None
        Returns:
            Array of center of bins.
        """
        try:
            return self._x
        except AttributeError:
            self._x: np.ndarray = _bin_centers(bin_edges = self.x_bin_edges, bin_widths=self.x_bin_widths)

        return self._x

    @property
    def y(self) -> np.ndarray:
        """ The y bin centers (``y``).

        This property caches the x value so we don't have to calculate it every time.

        Args:
            None
        Returns:
            Array of center of bins.
        """
        try:
            return self._y
        except AttributeError:
            self._y: np.ndarray = _bin_centers(bin_edges = self.y_bin_edges, bin_widths=self.y_bin_widths)

        return self._y


def histogram_from_array(df: Union[pd.DataFrame, UprootArrays], observable_name: str, axis: bh.axis.Regular) -> histogram.Histogram1D:
    bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
    bh_hist.fill(df[observable_name].to_numpy())
    h = histogram.Histogram1D(
        bin_edges = bh_hist.axes[0].edges,
        y = bh_hist.view().value,
        errors_squared = np.copy(bh_hist.view().variance),
    )
    # Scale by bin width
    h /= h.bin_widths
    return h

def response_from_array(df: Union[pd.DataFrame, UprootArrays], hybrid_observable_name: str, particle_observable_name: str, axes: Sequence[bh.axis.Regular]) -> BinnedData2D:
    bh_hist = bh.Histogram(*axes, storage=bh.storage.Weight())
    bh_hist.fill(df[hybrid_observable_name].to_numpy(), df[particle_observable_name].to_numpy())

    # TODO: Scaling!
    # Scale by bin width
    #h /= h.bin_widths
    # Scale by njets.
    # TODO: Careful - this may not be the correct n_jets
    #h /= len(df_in)
    x, y = bh_hist.axes.edges
    view = bh_hist.view()
    return BinnedData2D(
        x_bin_edges = x,
        y_bin_edges = y,
        values = view.counts,
        errors_squared = view.variance,
    )

def _get_branches_from_tree(filename: Path, tree_name: str) -> List[str]:
    keys: List[str] = []
    with uproot.open(filename) as f:
        keys = f[tree_name].allkeys()
    return keys


class UprootFiles(ContextManager[Sequence[uproot.rootio.ROOTDirectory]]):
    def __init__(self, filenames: Sequence[str]):
        self._filenames = filenames
        self._open_files: List[uproot.rootio.ROOTDirectory] = []

    def __enter__(self) -> Sequence[uproot.rootio.ROOTDirectory]:
        for filename in self._filenames:
            self._open_files.append(uproot.open(filename))
        return self._open_files

    def __exit__(self, execption_type: Optional[Type[BaseException]],
                 exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        for f in self._open_files:
            f.close()

        # To ensure that any other exceptions that were raised are re-raised, we don't return any value.

def root_tree_to_hdf5(dataset_name: str, filenames: Sequence[str], tree_name: str, array_names: Sequence[str], hdf5_file: h5py.File) -> None:
    trees = []
    with UprootFiles(filenames) as files:
        for f in files:
            trees.append(f[tree_name])

        # Convert trees to hdf5
        awkward_to_hdf5(trees = trees, dataset_name=dataset_name, array_names=array_names, hdf5_file=hdf5_file)

def load_data(collision_system: str, dataset_config: Dict[str, Any],
              available_datasets_config: Dict[str, Any], force_reload_dataset: bool = False,
              hdf5_filename: Path = Path("trains/HDF5/data.h5")) -> UprootArrays:
    # Setup
    dataset_name = dataset_config["name"]
    array_names = dataset_config.get("branches", None)
    selected_dataset_config = available_datasets_config[dataset_name]
    filenames = selected_dataset_config["files"]
    tree_name = selected_dataset_config["tree_name"]

    with h5py.File(hdf5_filename, "a") as f:
        if dataset_name not in f or force_reload_dataset:
            # Convert tree to hdf5 via uproot and awkward-array
            logger.info(f"Converting dataset {dataset_name} from ROOT to HDF5.")
            root_tree_to_hdf5(dataset_name=dataset_name, filenames=filenames, tree_name=tree_name, array_names=array_names, hdf5_file=f)
        else:
            # Ensure that all requested branches are in the data.
            missing_arrays = []
            for array_name in array_names:
                if array_name not in f[dataset_name]:
                    missing_arrays.append(array_name)

            # If they're not available, load them into the store.
            if missing_arrays:
                logger.info(f"Missing columns: {missing_arrays}. Will convert from ROOT to HDF5.")
                root_tree_to_hdf5(dataset_name=dataset_name, filenames=filenames, tree_name=tree_name, array_names=missing_arrays, hdf5_file=f)

        # Finally, all the data is available and we can actually load it.
        logger.info(f"Loading dataset \"{dataset_name}\" from HDF5.")
        data = hdf5_to_awkward(hdf5_file=f, dataset_name=dataset_name, array_names=array_names)

    return data

