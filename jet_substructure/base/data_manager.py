import logging
from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import Mapping, MutableMapping
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Type, TypeVar, Union, cast

import attr
import awkward as ak
import h5py
import numpy as np
import uproot
from typing_extensions import Protocol


logger = logging.getLogger(__name__)

# TODO: Consolidate.
UprootArray = Union[np.ndarray, ak.JaggedArray]
UprootArrays = Dict[str, UprootArray]
_T = TypeVar("_T")


def _ensure_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return [Path(p) for p in paths]


def _ensure_hdf5_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return [Path(p).with_suffix(".h5") for p in paths]


@attr.s
class AwkwardArrayWrapper:
    ...


@attr.s
class HDF5Wrapper:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_paths)
    name: str = attr.ib()
    branches: Iterable[str] = attr.ib()
    _stored_branches: Set[str] = attr.ib(factory=list)

    # def __attrs_post_init__(self) -> None:
    #    self._uproot_tree = LRUCache(5)
    #    self._hdf5_tree = LRUCache(5)

    def files(self, mode: str = "r") -> Iterator[h5py.File]:
        for filename in self.filenames:
            with h5py.File(filename, mode) as f:
                yield f

    def trees(self) -> Iterator[Dict[str, Any]]:
        # TODO: Define __iter__?
        with h5py.File(self.filenames[0], "r") as f:
            root_branches = f[self.name].attrs["root_branches"]
            root_filenames = f[self.name].attrs["root_filenames"]

        # TODO: Add cache, other args.
        uproot_trees = uproot.iterate(path=root_filenames, treepath=self.name, branches=root_branches)
        for f, uproot_tree in zip(self.files(), uproot_trees):
            storage = ak.hdf5(f[self.name])
            hdf5_tree = storage[self.branches]
            self._uproot_tree = uproot_tree
            self._hdf5_tree = hdf5_tree
            yield {**uproot_tree, **hdf5_tree}

    def __getitem__(self, key: str) -> Any:
        try:
            return self._uproot_tree[key]
        except:
            return self._hdf5_tree[key]


@attr.s
class DatasetOld:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_paths)
    name: str = attr.ib()
    branches: Set[str] = attr.ib(converter=set)

    @property
    def root_filenames(self) -> Sequence[Path]:
        return self._filenames

    @property
    def hdf5_filenames(self) -> Sequence[Path]:
        return [f.with_suffix("h5") for f in self.filenames]

    def _uproot_dataset_properties(self) -> Set[str]:
        """ Determine dataset properties of a given file.

        Args:
            path: Path to the file with the properties of interest.
        Returns:
            Branches in the tree.
        """
        with uproot.open(self.filenames[0]) as f:
            branches = set(f[self.name].allkeys())
        return branches

    def _hdf5_dataset_properties(self) -> Set[str]:
        """ Determine dataset properties of a given file.

        Args:
            path: Path to the file with the properties of interest.
        Returns:
            Branches in the tree.
        """
        with h5py.File(self.hdf5_filenames[0], "r") as f:
            storage = ak.hdf5(f[self.name])
            branches = set(storage.keys())
        return branches

    def calculate_derived_data(self) -> bool:
        uproot_branches = self._uproot_dataset_properties()
        hdf5_branches = self._hdf5_dataset_properties()
        needed_branches = self.branches - uproot_branches
        calculate_these_branches = needed_branches - hdf5_branches

        return True

    def data_for_analysis(self) -> Iterator[Dict[str, UprootArrays]]:
        # We assume that the branches are the same in all of the files.
        # NOTE: We don't use the generator here because we would need to instantiate it to get these properties.
        uproot_branches = self._uproot_dataset_properties()
        hdf5_branches = self._hdf5_dataset_properties()
        # Check for missing branches
        missing_branches = self.branches - uproot_branches - hdf5_branches
        if missing_branches:
            raise RuntimeError(f"Missing branches: {missing_branches}. Calculate derived data.")

        # Setup and yield the generators
        hdf5_trees = HDF5Wrapper(filenames=self.hdf5_filenames, name=self.name, branches=hdf5_branches)

        for uproot_tree, hdf5_tree in zip(uproot_trees, hdf5_trees):
            yield {**uproot_tree, **hdf5_tree}

    def setup(self) -> None:
        """

        """
        # ...
        uproot_branches = self._uproot_dataset_properties()
        # Store these branches in the HDF5 metadata
        hdf5_trees = HDF5Wrapper(filenames=self.hdf5_filenames, name=self.name, branches=uproot_branches)

        for f in hdf5_trees.files():
            group = f.require_group(self.name, track_order=True)
            # Add ROOT metadata.
            group.attrs["root_filenames"] = self.root_filenames
            group.attrs["root_branches"] = uproot_branches

        # Now, calculate the other needed branches.
        # TODO: Constituent indices
        # TODO: kt, zg, etc??
        ...


###################
# Ignore above...
###################


@attr.s
class TreeWrapper(MutableMapping[str, UprootArray]):
    _tree: MutableMapping[str, UprootArray] = attr.ib()
    _branches: Set[str] = attr.ib(converter=set, default=set())

    @property
    def branches(self) -> Set[str]:
        """ Branches stored in the tree.

        Accessing them in this manner allows them to be set externally, but then we calculate
        them if they're not already set.
        """
        if len(self._branches) != 0:
            return self._branches
        else:
            self._branches = set(cast(Iterable[str], self._tree.keys()))
        return self._branches

    def __len__(self) -> int:
        return len(self._tree)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tree)

    def __getitem__(self, key: Union[str, Iterable[str]]) -> UprootArrays:
        if not isinstance(key, str):
            # Take an intersection with the requested keys, since we can only take onces
            # which actually exist in the tree.
            branches_to_return = self.branches.intersection(list(key))
            logger.debug(
                f"Branches: "
                f"\n\tRequested: {key}"
                f"\n\tReturning: {branches_to_return}"
                f"\n\tdifference: {set(key) - branches_to_return}"
            )
            # We ignore the typing here because our typing of the tree isn't quite right. The
            # typing of the tree is such that it get item of either a str or an iterable of strings.
            return dict(zip(branches_to_return, self._tree[branches_to_return]))  # type: ignore

        return {key: self._tree[key]}

    def __setitem__(self, key: str, item: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"The key must be a string. Passed: {key}")
        self._tree.__setitem__(key, item)

    def __delitem__(self, key: str) -> None:
        self._tree.__delitem__(key)


@attr.s
class UprootTreeWrapper(TreeWrapper):
    # NOTE: Not adding new fields - just updating the types.
    _tree: MutableMapping[str, UprootArrays]

    @property
    def branches(self) -> Set[str]:
        if self._branches is not None:
            return self._branches
        else:
            # NOTE: It's allkeys instead of just keys()
            self._branches = set(self._tree.allkeys())
        return self._branches

    def __setitem__(self, key: Union[str, Iterable[str]], item: Any) -> None:
        raise TypeError(f"Cannot write key {key} to the uproot TTree. Instead, write to the HDF5 tree.")


@attr.s
class HDF5TreeWrapper(TreeWrapper):
    # NOTE: Not adding new fields - just updating the types.
    _tree: ak.hdf5


@attr.s
class Tree(MutableMapping[str, UprootArrays]):
    _uproot_tree: UprootTreeWrapper = attr.ib()
    _hdf5_tree: HDF5TreeWrapper = attr.ib()

    @property
    def _trees(self) -> List[TreeWrapper]:
        return [self._uproot_tree, self._hdf5_tree]

    @property
    def branches(self) -> Set[str]:
        try:
            return self._branches
        except AttributeError:
            self._branches: Set[str] = reduce(set.union, [tree.branches for tree in self._trees])
        return self._branches

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self) -> Iterator[str]:
        return iter(self.branches)

    def __getitem__(self, key: Union[str, Iterable[str]]) -> UprootArrays:
        if isinstance(key, str):
            for tree in self._trees:
                if key in tree.branches:
                    return {key: tree[key]}
            raise ValueError(f"Could not retrieve branch {key}")
        else:
            missing_branches = set(key) - self.branches
            if missing_branches:
                raise ValueError(
                    "Not all requested branches are available. Missing: {missing_branches}. Requested branches: {key}"
                )

            # We rely on each tree ignoring branches that aren't relevant to it.

            return dict(ChainMap(*[tree[key] for tree in self._trees]))

    def __setitem__(self, key: str, item: Any) -> None:
        # Can only store data in the HDF5 file.
        self._hdf5_tree[key] = item

    def __delitem__(self, key: str) -> None:
        del self._hdf5_tree[key]


# @attr.s
# class FileWrapper:
#    _filename: Path = attr.ib()
#    _file: None
#
#    def __attrs_post_init__(self) -> None:
#        ...
#
#    @property
#    def _uproot_filename(self) -> Path:
#        return self._filename
#
#    @property
#    def _root_filename(self) -> Path:
#        return self.uproot_filename
#
#    @property
#    def _hdf5_filename(self) -> Path:
#        return self._filename
#
#    def _uproot_branches(self) -> Set[str]:
#        """ Determine uproot branches of the file.
#
#        Args:
#            None.
#        Returns:
#            Branches in the tree.
#        """
#        with uproot.open(self._uproot_filename) as f:
#            return set(f[self.name].allkeys())
#
#    def _hdf5_branches(self) -> Set[str]:
#        """ Determine hdf5 branches of the file.
#
#        Args:
#            path: Path to the file with the properties of interest.
#        Returns:
#            Branches in the tree.
#        """
#        with h5py.File(self.hdf5_filename, "r") as f:
#            storage = ak.hdf5(f[self.name])
#            return set(storage.keys())


# class TreeIterator(Protocol):
#    def iterate(self) -> Iterator[Union[UprootArrays, ak.hdf5]]:
#        ...


@attr.s
class UprootTreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_paths)
    _tree_name: str = attr.ib()

    def __iter__(self) -> Iterator[UprootTreeWrapper]:
        # TODO: Add additional arguments!
        for tree in uproot.iterate(path=self._filenames, treepath=self._tree_name):
            yield UprootTreeWrapper(tree)


@attr.s
class HDF5TreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_hdf5_paths)
    _tree_name: str = attr.ib()
    _file_mode: str = attr.ib(default="r")

    def __iter__(self) -> Iterator[HDF5TreeWrapper]:
        for filename in self._filenames:
            with h5py.File(filename, self._file_mode) as f:
                storage = ak.hdf5(f.require_group(self._tree_name))
                yield HDF5TreeWrapper(storage)


@attr.s
class IterateTrees:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_paths)
    tree_name: str = attr.ib()
    branches: Set[str] = attr.ib(converter=set)
    _current_tree: Optional[Tree] = attr.ib(default=None)

    # @property
    # def _uproot_filenames(self) -> Path:
    #    return self._filenames

    # @property
    # def _root_filename(self) -> Path:
    #    return self.uproot_filenames

    # @property
    # def _hdf5_filename(self) -> Path:
    #    return [f.with_suffix("h5") for f in self._filenames]

    # def _uproot_iterate(self) -> Iterator[UprootArrays]:
    #    return uproot.iterate(path=self.root_filenames, treepath=self.name)

    # def _hdf5_iterate(self, mode: str = "r") -> Iterator[h5py.File]:
    #    for filename in self._hdf5_filenames:
    #        with h5py.File(filename, mode) as f:
    #            storage = ak.hdf5(f[self.name])
    #            yield storage

    def data_for_analysis(self) -> Iterator[Dict[str, UprootArrays]]:
        for uproot_tree, hdf5_tree in zip(
            UprootTreeIterator(filenames=self._filenames, tree_name=self.tree_name),
            HDF5TreeIterator(filenames=self._filenames, tree_name=self.tree_name),
        ):
            _current_tree = Tree(uproot_tree, hdf5_tree)

            # yield {**uproot_tree, **hdf5_tree}
            yield _current_tree[self.branches]

    # def __iter__(self) -> Iterator[UprootArrays]:
    #    _uproot_branches = self._uproot_branches()
    #    _hdf5_branches = self._hdf5_branches()

    # def desired_interface(self) -> Iterator[UprootArrays]:
    #    for f in self.files:
    #        # f is a current file
    #        yield f[self.branches]

    #    ...
