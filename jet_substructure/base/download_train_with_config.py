#!/usr/bin/env python3

""" Download files based on the config.yaml

"""

import argparse
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Sequence

import pachyderm.alice.utils as alice_utils
from pachyderm import yaml


logger = logging.getLogger(__name__)


_possible_merging_stages = ["merged", "manual", "single_run_manual", "Stage_1", "Stage_2", "Stage_5"]


def add_files_from_xml_file(
    alien_xml_file: Path, local_xml_file: Path, local_train_dir: Path, child_label: str, additional_label: str = ""
) -> Dict[str, str]:
    output = {}
    # Download the XML file...
    logger.info(f"Downloading {alien_xml_file.name} file: {alien_xml_file} to file://{local_xml_file}")
    subprocess.run(
        ["alien_cp", str(alien_xml_file), f"file://{str(local_xml_file)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Open local XML file
    tree = ET.parse(str(local_xml_file))
    root = tree.getroot()
    collection = root[0]
    # Extract the filenames
    for node in collection:
        try:
            lfn = node[0].attrib["lfn"]
            alien_file = lfn.replace("root_archive.zip", "AnalysisResults.root")
            i = int(node.attrib["name"])
            if additional_label:
                label = f"{additional_label}.{i:02}"
            else:
                label = f"{i:02}"
            local_file = local_train_dir / f"AnalysisResults.{child_label}.{label}.root"
            # print(f"Adding alien://{alien_file} : {local_file}")
            output[str(alien_file)] = str(local_file)
            # print(f"Downloading alien://{alien_file} to {local_file}")
            # process = subprocess.run(
            #    ["alien_cp", f"alien://{str(alien_file)}", str(local_file)],
            #    stdout = subprocess.PIPE, stderr = subprocess.PIPE
            # )
        except IndexError:
            pass

    return output


def download(trains: Sequence[int]) -> None:  # noqa: C901

    y = yaml.yaml()

    output = {}
    for train_number in trains:
        logger.info(f"Processing train {train_number}")
        local_train_dir = Path(str(train_number))
        config_filename = local_train_dir / "config.yaml"
        with open(config_filename, "r") as f:
            config = y.load(f)

        # Sanity check
        # If this fails, it means I probably forgot to update the run number in the YAML config.
        assert int(config["number"]) == train_number

        # Determine train properties.
        base_alien_path = Path("/alice/cern.ch/user/a/alitrain/")
        PWG = config.get("PWG", "PWGJE")
        train_name = config.get("train")
        base_alien_path = base_alien_path / PWG / train_name
        train_directories_on_alien = alice_utils.list_alien_dir(base_alien_path)
        possible_directories = [dir for dir in train_directories_on_alien if dir.startswith(str(train_number))]
        # NOTE: There could be more than 1 directory if there are children. We only have to check for whether it's empty.
        if len(possible_directories) == 0:
            logger.warning(f"Can't find any directories for train. Skipping {train_number}.")
            continue

        alien_output_info = config["alien_output_info"]
        for child_name, child_info in alien_output_info.items():
            # Validation
            child_name = child_name.lower()

            # Determine the corresponding AliEn directory.
            possible_child_directories = [dir for dir in possible_directories if dir.endswith(child_name)]
            if len(possible_child_directories) != 1:
                logger.debug(f"Could not find train directory corresponding to child {child_name}. Continuing")
                continue

            # Up until now, the possible directories have been relative to the base_alien_path.
            # Since we're getting close to downloading or saving, we add it back it so we have an absolute path.
            likely_child_directory = possible_child_directories[0]
            alien_dir = base_alien_path / Path(likely_child_directory) / "merge"
            # Extract values from the config needed to finally determine the files to download.
            # Last successful merging stage, which we will use to determine what to download.
            # We force the user to record the stage (rather than determine it automatically) so the record will be saved.
            stage_to_download = child_info["stage_to_download"]
            # Validation
            if stage_to_download not in _possible_merging_stages:
                raise ValueError(
                    f"Invalid last successful merging stage. Provided: {stage_to_download}. Possible values: {_possible_merging_stages}"
                )
            # Child label (such as LHC18q)
            child_label = child_info.get("name", child_name)
            # Validation
            child_label = child_label.replace("LHC", "").replace("lhc", "")

            if stage_to_download == "merged":
                # We just want the final merge files.
                local_file = local_train_dir / f"AnalysisResults.{child_label}.root"
                # So we just save it.
                output[str(alien_dir / "AnalysisResults.root")] = str(local_file)
            elif stage_to_download == "manual":
                manual_config = child_info["manual"]
                for run_number, manual_stage_to_download in manual_config.items():
                    # Validation
                    if manual_stage_to_download not in _possible_merging_stages:
                        raise ValueError(
                            f"Invalid last successful merging stage. Provided: {manual_stage_to_download}. Possible values: {_possible_merging_stages}"
                        )
                    # NOTE: This is somewhat LHC18{q,r} specific
                    # Example: /alice/data/2018/LHC18r/000296934/pass1/PWGJE/Jets_EMC_PbPb/5902_20200515-1910_child_1
                    manual_dir = (
                        Path("/alice/data/2018/")
                        / f"LHC{child_label}"
                        / f"000{run_number}"
                        / "pass1"
                        / PWG
                        / train_name
                        / likely_child_directory
                    )
                    if manual_stage_to_download == "merged":
                        local_file = local_train_dir / f"AnalysisResults.{child_label}.{run_number}.root"
                        output[str(manual_dir / "AnalysisResults.root")] = str(local_file)
                    elif "Stage" in manual_stage_to_download:
                        # Use a stage of the merging.
                        alien_xml_file = manual_dir / f"{manual_stage_to_download}.xml"
                        local_xml_file = local_train_dir / f"{run_number}_{manual_stage_to_download}_{child_name}.xml"
                        result = add_files_from_xml_file(
                            alien_xml_file=alien_xml_file,
                            local_xml_file=local_xml_file,
                            local_train_dir=local_train_dir,
                            child_label=child_label,
                            additional_label=str(run_number),
                        )
                        output.update(result)
                    else:
                        # Didn't even get to a stage of the merging. Take whatever is there...
                        directories_with_output_files = alice_utils.list_alien_dir(manual_dir)
                        directories_with_output_files = [manual_dir / d for d in directories_with_output_files]
                        _additional_label = str(run_number)
                        for d in directories_with_output_files:
                            # Use having a suffix as a proxy for a file, and without a suffix as a directory.
                            # (I think we can query AliEn, but that seems like overkill just for determining directories...
                            if d.suffix:
                                continue

                            # AliEn filename
                            alien_file = d / "AnalysisResults.root"
                            # Local filename
                            i = int(d.name)
                            if _additional_label:
                                label = f"{_additional_label}.{i:02}"
                            else:
                                label = f"{i:02}"
                            local_file = local_train_dir / f"AnalysisResults.{child_label}.{label}.root"
                            print(f"Adding alien://{alien_file} : {local_file}")
                            output[str(alien_file)] = str(local_file)

            else:
                alien_xml_file = alien_dir / f"{stage_to_download}.xml"
                local_xml_file = local_train_dir / f"{stage_to_download}_{child_name}.xml"
                result = add_files_from_xml_file(
                    alien_xml_file=alien_xml_file,
                    local_xml_file=local_xml_file,
                    local_train_dir=local_train_dir,
                    child_label=child_label,
                )
                output.update(result)

                ## Download the XML file...
                # print(f"Downloading {stage_to_download}.xml file: alien://{alien_xml_file} to {local_xml_file}")
                # process = subprocess.run(
                #    ["alien_cp", f"alien://{str(alien_xml_file)}", str(local_xml_file)],
                #    stdout=subprocess.PIPE,
                #    stderr=subprocess.PIPE,
                # )

                ## Open local XML file
                # tree = ET.parse(str(local_xml_file))
                # root = tree.getroot()
                # collection = root[0]
                ## Extract the filenames
                # for node in collection:
                #    try:
                #        lfn = node[0].attrib["lfn"]
                #        alien_file = lfn.replace("root_archive.zip", "AnalysisResults.root")
                #        label = int(node.attrib["name"])
                #        local_file = local_train_dir / f"AnalysisResults.{child_label}.{label:02}.root"
                #        # print(f"Adding alien://{alien_file} : {local_file}")
                #        output[str(alien_file)] = str(local_file)
                #        # print(f"Downloading alien://{alien_file} to {local_file}")
                #        # process = subprocess.run(
                #        #    ["alien_cp", f"alien://{str(alien_file)}", str(local_file)],
                #        #    stdout = subprocess.PIPE, stderr = subprocess.PIPE
                #        # )
                #    except IndexError:
                #        pass

    # Write out the files
    with open("files_to_download.yaml", "w") as f:
        y.dump(output, f)


def entry_point() -> None:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description=f"Split tree into chunks.")
    parser.add_argument(
        "--train", type=int, help="Single train to process, or first train to process.",
    )
    parser.add_argument(
        "--maxTrain",
        type=int,
        default=0,
        help="Max train number. It will include all trains between train and maxTrain (ie. upper limit is inclusive).",
    )
    args = parser.parse_args()

    # Determine what to download
    trains = []
    if not args.maxTrain:
        trains = [args.train]
    else:
        # NOTE: We add +1 so that we're inclusive on this upper limit. I think this will be more intuitive.
        trains = list(range(args.train, args.maxTrain + 1))

    logger.info(f"Downloading trains: {trains}")

    download(trains=trains)


if __name__ == "__main__":
    entry_point()
