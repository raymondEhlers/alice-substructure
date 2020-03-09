""" Manager access to datasets and trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import typing
from collections import ChainMap
from functools import partial, reduce
from pathlib import Path
from typing import FrozenSet, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Union

import attr
import awkward as ak
import h5py
import uproot

from jet_substructure.base.helpers import UprootArray, UprootArrays


logger = logging.getLogger(__name__)


def _convert_wildcards(paths: Sequence[Path]) -> List[Path]:
    return_paths: List[Path] = []
    for path in paths:
        p = str(path)
        if "*" in str(p):
            # Glob all associated filenames.
            loc = p.find("*")
            return_paths.extend(Path(p[:loc]).glob(p[loc:]))
        else:
            return_paths.append(path)

    # Sort in the expected order.
    return_paths = sorted(return_paths, key=lambda p: int("".join(filter(str.isdigit, str(p)))))
    return return_paths


def _ensure_and_expand_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return _convert_wildcards([Path(p) for p in paths])


def _ensure_and_expand_hdf5_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return _convert_wildcards([Path(p).with_suffix(".h5") for p in paths])


@attr.s
class TreeWrapper(MutableMapping[str, UprootArray]):
    """ Wrapper around an open tree.

    It keeps track of the tree itself, as well as tree metadata such as the available branches
    or the filename where it is stored.
    """

    _tree: MutableMapping[str, UprootArray] = attr.ib()
    _filename: Path = attr.ib(converter=Path)
    _tree_name: str = attr.ib()
    _branches: FrozenSet[str] = attr.ib(converter=frozenset, default=frozenset())

    @property
    def filename(self) -> Path:
        """ Filename of the current tree. """
        return self._filename

    @property
    def tree_name(self) -> str:
        """ Name of the current tree. """
        return self._tree_name

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

    @typing.overload
    def __getitem__(self, key: str) -> UprootArray:
        ...

    @typing.overload
    def __getitem__(self, key: Iterable[str]) -> UprootArrays:
        ...

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

        return self._tree[key]

    def __setitem__(self, key: str, item: UprootArray) -> None:
        self._tree.__setitem__(key, item)

    def __delitem__(self, key: str) -> None:
        self._tree.__delitem__(key)


@attr.s
class UprootTreeWrapper(TreeWrapper):
    def __setitem__(self, key: Union[str, Iterable[str]], item: UprootArray) -> None:
        raise TypeError(f"Cannot write key {key} to the uproot TTree. Instead, write to the HDF5 tree.")


@attr.s
class HDF5TreeWrapper(TreeWrapper):
    # NOTE: Not adding new fields - just updating the types.
    _tree: ak.hdf5


@attr.s
class UprootTreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_and_expand_paths)
    _tree_name: str = attr.ib()
    _cache: MutableMapping[str, UprootArray] = attr.ib(factory=partial(uproot.ThreadSafeArrayCache, "1 GB"))
    _key_cache: MutableMapping[str, UprootArray] = attr.ib(factory=partial(uproot.ThreadSafeArrayCache, "100 MB"))

    def __iter__(self) -> Iterator[UprootTreeWrapper]:
        # NOTE: If the file sizes get too big, can set entrysteps to something like `entrysteps=100000`, which
        #       is large, but less than the size of the file. We want to keep it as large as possible.
        for filename, tree in zip(
            self._filenames,
            uproot.iterate(
                path=self._filenames,
                treepath=self._tree_name,
                namedecode="utf-8",
                cache=self._cache,
                keycache=self._key_cache,
            ),
        ):
            yield UprootTreeWrapper(tree=tree, filename=filename, tree_name=self._tree_name)


@attr.s
class HDF5TreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_and_expand_hdf5_paths)
    _tree_name: str = attr.ib()
    _file_mode: str = attr.ib(default="r")

    def __iter__(self) -> Iterator[HDF5TreeWrapper]:
        for filename in self._filenames:
            # Need to ensure that the file is created if it doesn't already exist. Best way to do so is via "a"
            file_mode = "a" if not filename.exists() else self._file_mode
            with h5py.File(filename, file_mode) as f:
                storage = ak.hdf5(f.require_group(self._tree_name))
                yield HDF5TreeWrapper(tree=storage, filename=filename, tree_name=self._tree_name)


@attr.s
class Tree(MutableMapping[str, UprootArrays]):
    _uproot_tree: UprootTreeWrapper = attr.ib()
    _hdf5_tree: HDF5TreeWrapper = attr.ib()

    @property
    def filename(self) -> Path:
        """ The filename of the (uproot) tree.

        The HDF5 filename is the same, just with the extension replaced with `.h5`.
        """
        return self._uproot_tree.filename

    @property
    def tree_name(self) -> str:
        """ The filename of the (uproot) tree.

        The HDF5 tree name is the same.
        """
        return self._uproot_tree.tree_name

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

    @typing.overload
    def __getitem__(self, key: str) -> UprootArray:
        ...

    @typing.overload
    def __getitem__(self, key: Iterable[str]) -> UprootArrays:
        ...

    def __getitem__(self, key: Union[str, Iterable[str]]) -> Union[UprootArray, UprootArrays]:
        if isinstance(key, str):
            for tree in self._trees:
                if key in tree.branches:
                    return tree[key]
            raise KeyError(f"Could not retrieve branch {key}")
        else:
            missing_branches = set(key) - self.branches
            if missing_branches:
                raise ValueError(
                    f"Not all requested branches are available. Missing: {missing_branches}. Requested branches: {key}"
                )

            # We rely on each tree ignoring branches that aren't relevant to it.
            return dict(ChainMap(*[tree[key] for tree in self._trees]))

    def __setitem__(self, key: str, item: UprootArray) -> None:
        # Can only store data in the HDF5 file.
        self._hdf5_tree[key] = item

    def __delitem__(self, key: str) -> None:
        del self._hdf5_tree[key]


@attr.s
class IterateTrees:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_and_expand_paths)
    tree_name: str = attr.ib()
    branches: FrozenSet[str] = attr.ib(converter=set)
    _current_tree: Optional[Tree] = attr.ib(default=None)

    def __len__(self) -> int:
        return len(self._filenames)

    def __contains__(self, key: str) -> bool:
        return Path(key) in self._filenames

    # def data_for_analysis(self) -> Iterator[Mapping[str, UprootArrays]]:
    def __iter__(self) -> Iterator[Tree]:
        for uproot_tree, hdf5_tree in zip(
            UprootTreeIterator(filenames=self._filenames, tree_name=self.tree_name),
            HDF5TreeIterator(filenames=self._filenames, tree_name=self.tree_name),
        ):
            self._current_tree = Tree(uproot_tree, hdf5_tree)

            # yield _current_tree[self.branches]
            yield self._current_tree
