""" Manager access to datasets and trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from collections import ChainMap
from functools import partial, reduce
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Union

import attr
import awkward as ak
import h5py
import uproot

from jet_substructure.base.helpers import UprootArray, UprootArrays


logger = logging.getLogger(__name__)


def _ensure_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return [Path(p) for p in paths]


def _ensure_hdf5_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return [Path(p).with_suffix(".h5") for p in paths]


@attr.s
class TreeWrapper(MutableMapping[str, UprootArray]):
    _tree: MutableMapping[str, UprootArray] = attr.ib()
    _branches: FrozenSet[str] = attr.ib(converter=frozenset, default=frozenset())

    @property
    def branches(self) -> FrozenSet[str]:
        """ Branches stored in the tree.

        Accessing them in this manner allows them to be set externally, but then we calculate
        them if they're not already set.
        """
        if len(self._branches) != 0:
            return self._branches
        else:
            self._branches = frozenset(self._tree.keys())
        return self._branches

    def __len__(self) -> int:
        return len(self._tree)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tree)

    def _retrieve_branches(self, key: Iterable[str]) -> UprootArrays:
        return {k: self._tree[k] for k in key}

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
            # Retrieve all of the branches.
            return self._retrieve_branches(branches_to_return)

        return {key: self._tree[key]}

    def __setitem__(self, key: str, item: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"The key must be a string. Passed: {key}")
        self._tree.__setitem__(key, item)

    def __delitem__(self, key: str) -> None:
        self._tree.__delitem__(key)


@attr.s
class UprootTreeWrapper(TreeWrapper):
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
    def branches(self) -> FrozenSet[str]:
        try:
            return self._branches
        except AttributeError:
            self._branches: FrozenSet[str] = reduce(frozenset.union, [tree.branches for tree in self._trees])
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
                    f"Not all requested branches are available. Missing: {missing_branches}. Requested branches: {key}"
                )

            # We rely on each tree ignoring branches that aren't relevant to it.
            return dict(ChainMap(*[tree[key] for tree in self._trees]))

    def __setitem__(self, key: str, item: Any) -> None:
        # Can only store data in the HDF5 file.
        self._hdf5_tree[key] = item

    def __delitem__(self, key: str) -> None:
        del self._hdf5_tree[key]


@attr.s
class UprootTreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_paths)
    _tree_name: str = attr.ib()
    _cache: MutableMapping[str, UprootArray] = attr.ib(factory=partial(uproot.ThreadSafeArrayCache, "1 GB"))
    _key_cache: MutableMapping[str, UprootArray] = attr.ib(factory=partial(uproot.ThreadSafeArrayCache, "100 MB"))

    def __iter__(self) -> Iterator[UprootTreeWrapper]:
        # NOTE: If the file sizes get too big, can set entrysteps to something like `entrysteps=100000`, which
        #       is large, but less than the size of the file. We want to keep it as large as possible.
        for tree in uproot.iterate(
            path=self._filenames,
            treepath=self._tree_name,
            namedecode="utf-8",
            cache=self._cache,
            keycache=self._key_cache,
        ):
            yield UprootTreeWrapper(tree)


@attr.s
class HDF5TreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_hdf5_paths)
    _tree_name: str = attr.ib()
    _file_mode: str = attr.ib(default="r")

    def __iter__(self) -> Iterator[HDF5TreeWrapper]:
        for filename in self._filenames:
            # Need to ensure that the file is created if it doesn't already exist.
            # Best way to do so is via "a"
            file_mode = "a" if not filename.exists() else self._file_mode
            with h5py.File(filename, file_mode) as f:
                storage = ak.hdf5(f.require_group(self._tree_name))
                yield HDF5TreeWrapper(storage)


@attr.s
class IterateTrees:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_paths)
    tree_name: str = attr.ib()
    branches: FrozenSet[str] = attr.ib(converter=set)
    _current_tree: Optional[Tree] = attr.ib(default=None)

    def data_for_analysis(self) -> Iterator[Mapping[str, UprootArrays]]:
        for uproot_tree, hdf5_tree in zip(
            UprootTreeIterator(filenames=self._filenames, tree_name=self.tree_name),
            HDF5TreeIterator(filenames=self._filenames, tree_name=self.tree_name),
        ):
            _current_tree = Tree(uproot_tree, hdf5_tree)

            yield _current_tree[self.branches]
