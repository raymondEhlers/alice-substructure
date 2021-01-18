#!/usr/bin/env python3

from pathlib import Path
from typing import Any

import awkward1 as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from pachyderm import yaml


def convert(manual: bool) -> None:

    collision_system = "embedPythia"
    y = yaml.yaml()
    with open("config/new_config.yaml", "r") as f:
        full_config = y.load(f)
    base_dir = Path(full_config["base_directory"])
    print(base_dir)
    config = full_config["execution"][collision_system]["dataset"]

    f = uproot.open("trains/embedPythia/5966/AnalysisResults.18q.repaired.root")
    tree = f[  # type: ignore
        "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
    ]
    arrays = tree.arrays(config["branches"])

    # Convert and write
    if manual:
        table = pa.Table.from_arrays([ak.to_arrow(arr) for arr in ak.unzip(arrays)], ak.keys(arrays))
        pq.write_table(table, "trains/embedPythia/5966/AnalysisResults.18q.repaired.parquet")
    else:
        # Write directly
        ak.to_parquet(arrays, "trains/embedPythia/5966/AnalysisResults.18q.repaired.direct.parquet")

    f.close()

    # import IPython; IPython.embed()


def test_read() -> Any:
    table = pq.read_table("trains/embedPythia/5966/AnalysisResults.18q.repaired.parquet")
    arrays = {k: ak.from_arrow(v) for k, v in zip(table.column_names, table.columns)}

    return arrays


def test_direct_read() -> Any:
    arrays = ak.from_parquet("trains/embedPythia/5966/AnalysisResults.18q.repaired.direct.parquet")

    return arrays


if __name__ == "__main__":
    import time

    # for manual in [False, True]:
    #    t1 = time.time()
    #    convert(manual=manual)
    #    t2 = time.time()
    #    print(f"Manual: {manual}, time elapsed: {t2 - t1}")

    t1 = time.time()
    arrays = test_read()
    means = ak.mean(arrays["data.fSubjets.fConstituentIndices"], axis=-1)
    t2 = time.time()
    print(f"test_read, time elapsed: {t2 - t1}")

    t1 = time.time()
    arrays = test_direct_read()
    means_direct = ak.mean(arrays["data.fSubjets.fConstituentIndices"], axis=-1)
    t2 = time.time()
    print(f"test_direct_read, time elapsed: {t2 - t1}")

    # Sanity check
    np.allclose(ak.to_numpy(ak.flatten(means)), ak.to_numpy(ak.flatten(means_direct)))

    import IPython

    IPython.embed()
