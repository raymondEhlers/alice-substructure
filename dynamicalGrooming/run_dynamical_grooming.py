#!/usr/bin/env python3

""" Run the event extractor AliPhysics task.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import enum
import uuid
from pathlib import Path
from typing import Any, List, Type, TypeVar

import attr
import numpy as np

import ROOT

@attr.s
class CppEnum:
    value: str = attr.ib()

    def __str__(self) -> str:
        """ Return the str without quotes by calling `str`.

        This ensures that ROOT will evaluate the enum correctly when evaluating the AddTask.
        """
        return str(self.value)

def _normalize_period(period: str) -> str:
    """ Normalize period name to have "LHC" in all caps.

    Args:
        period: Period name to be normalized.
    Returns:
        Normalized period name.
    """
    return period[:3].upper() + period[3:]

def _is_run2_data(period: str) -> bool:
    """ Determine if the period was data taken during Run 2.

    Performed by checking for the years LHC{15..18}. In principle, this should fail for MC productions
    (which is the desired behavior - the year of the MC production doesn't dictate the year of the data
    taking period).

    Args:
        period: Run period to be checked.
    Returns:
        True if the run period is in Run2.
    """
    for year in [15, 16, 17, 18]:
        if period.startswith(f"LHC{year}"):
            return True
    return False

class DataType(enum.Enum):
    """ ALICE data type.

    Either AOD or ESD.
    """
    AOD = 0
    ESD = 1

_T_BeamType = TypeVar("_T_BeamType", bound="BeamType")

class BeamType(enum.Enum):
    """ Beam type.

    Uses the AliAnalysisTaskEmcal enum.
    """
    pp = ROOT.AliAnalysisTaskEmcal.kpp
    pPb = ROOT.AliAnalysisTaskEmcal.kpA
    PbPb = ROOT.AliAnalysisTaskEmcal.kAA

    @classmethod
    def from_period(cls: Type[_T_BeamType], period: str) -> _T_BeamType:
        """ Determine the beam type from the run period.

        Note:
            This is based on a enumerated list of PbPb and pPb run periods
            and will need to be updated as new periods are added.

        Args:
            period: Run period.
        Returns:
            The determined beam type.
        """
        # Validation
        period = _normalize_period(period)
        PbPb_run_periods = [
            "LHC10h", "LHC11h", "LHC15o", "LHC18q", "LHC18r"
        ]
        pPb_run_periods = [
            "LHC12g", "LHC13b", "LHC13c", "LHC13d", "LHC13e", "LHC13f", "LHC16q", "LHC16r", "LHC16s", "LHC16t"
        ]
        # Initialized via str to help out mypy...
        if period in PbPb_run_periods:
            return cls["PbPb"]
        if period in pPb_run_periods:
            return cls["pPb"]
        return cls["pp"]

def _run_add_task_macro(task_path: Path, task_class_name: str, *args: Any) -> Any:
    """ Run a given add task macro.

    Note:
        This is a cute idea, but it's also rather fragile. See the comments in the code for details how this
        is remotely possible.

    Args:
        task_path: Path to the AddTask macro to be executed.
        task_class_name: Name of the class that the AddTask returns.
        args: All of the arguments to be passed to the AddTask macro.

    Returns:
        The task returned by the AddTask.
    """
    bool_map = {
        False: "false",
        True: "true",
    }
    task_args = ", ".join([f'"{v}"' if isinstance(v, str) else
                           bool_map[v] if isinstance(v, bool) else
                           str(v) for v in args])
    print(f"Running: {task_path}({task_args})")
    address = ROOT.gROOT.ProcessLine(f".x {task_path}({task_args})")
    # Need to convert the address into the task. Unfortunately, we can't cast the address directly into an object.
    # Instead, we use cling to perform the reinterpret_cast for us, and then we retrieve that task.
    # This is super convoluted, but it appears to work okay...
    # For a UUID to be a valid c++ variable, we need:
    #  - Prefix the name so that it starts with a letter.
    #  - To replace "-" with "_"
    cpp_temp_task_name = "temp_" + str(uuid.uuid4()).replace("-","_")
    # Cast the task
    ROOT.gInterpreter.ProcessLine(f"auto * {cpp_temp_task_name} = reinterpret_cast<{task_class_name}*>({address});")
    # And then retrieve and return the actual task.
    return getattr(ROOT, cpp_temp_task_name)

def run_dynamical_grooming(task_name: str,
                           period: str,
                           physics_selection: int,
                           data_type: DataType,
                           is_MC: bool) -> ROOT.AliAnalysisManager:
    """ Run the event extractor.

    Args:
        task_name: Name of the analysis (ie. given to the analysis manager).
        period: Run period.
        physics_selection: Physics selection to apply to the analysis.
        data_type: ALICE data type over which the analysis will run.
        is_MC: True if analyzing MC.

    Returns:
        The analysis manager.
    """
    # Validation
    period = _normalize_period(period)
    is_run2_data = _is_run2_data(period) if not is_MC else False
    # Determine the beam type from the period.
    beam_type = BeamType.from_period(period)

    analysis_manager = ROOT.AliAnalysisManager(task_name)

    if data_type == DataType.AOD:
        input_handler = ROOT.AliAnalysisTaskEmcal.AddAODHandler()
    else:
        input_handler = ROOT.AliAnalysisTaskEmcal.AddESDHandler()

    # Physics selection task
    # Enable pileup rejection (second argument) for pp
    #physics_selection_task = _run_add_task_macro(
    #    "$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C", "AliPhysicsSelectionTask",
    #    is_MC, beam_type == BeamType.pp
    #)
    physics_selection_task = ROOT.AliPhysicsSelectionTask.AddTaskPhysicsSelection(is_MC, beam_type == BeamType.pp)

    # AliMultSelection
    # Works for both pp and PbPb for the periods that it is calibrated
    # However, I seem to have trouble with pp MCs
    if is_run2_data:
        #multiplicity_selection_task = _run_add_task_macro(
        #    "$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C",
        #    "AliMultSelectionTask",
        #    False
        #)
        multiplicity_selection_task = ROOT.AliMultSelectionTask.AddTaskMultSelection(False)
        multiplicity_selection_task.SelectCollisionCandidates(physics_selection)

    ################
    # Debug settings
    ################
    #ROOT.AliLog.SetClassDebugLevel("AliEmcalCorrectionComponent", AliLog::kDebug+3)
    #ROOT.AliLog.SetClassDebugLevel("AliAnalysisTaskEmcalJetHCorrelations", AliLog::kDebug+1)
    #ROOT.AliLog.SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHPerformance", ROOT.AliLog::kDebug-1)
    #ROOT.AliLog.SetClassDebugLevel("AliJetContainer", AliLog::kDebug+7);

    # Setup
    #binning = np.array(
    #    [0, 5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235, 1000], dtype=np.int32
    #)
    #pt_hard_binning = ROOT.TArrayI()
    #pt_hard_binning.Set(len(binning));
    #for i, v in enumerate(binning):
    #    pt_hard_binning[i] = v

    # Dynamical grooming
    #task = _run_add_task_macro(
    #    "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetDynamicalGrooming.C", "PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming",
    #    "Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub",
    #    "Jet_AKTChargedR040_tracks_pT0150_E_scheme", "", "", 0.4, "Rho",
    #    "tracksSubR02", "tracks", "", "", "", "TPC", "V0M", physics_selection,
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kData"),
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kConstSub"),
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kInclusive"),
    #    0, 0, 0.6,
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kSecondOrder"),
    #    "Raw"
    #)
    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.AddTaskJetDynamicalGrooming(
        "Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub",
        "Jet_AKTChargedR040_tracks_pT0150_E_scheme", "", "", 0.4, "Rho",
        "tracksSubR02", "tracks", "", "", "", "TPC", "V0M", physics_selection,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kData,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kConstSub,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kInclusive,
        0, 0, 0.6,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kSecondOrder,
        "Raw"
    )
    #event_extractor_task.SetNumberOfPtHardBins(len(pt_hard_binning) - 1)
    #event_extractor_task.SetUserPtHardBinning(pt_hard_binning)

    tasks = analysis_manager.GetTasks()
    for i in range(tasks.GetEntries()):
        task = tasks.At(i)
        if not task:
            continue
        if task.InheritsFrom("AliAnalysisTaskEmcal"):
            task.SetForceBeamType(beam_type.value)
            print(f"Setting beam type {beam_type.name} for task {task.GetName()}")
        if task.InheritsFrom("AliEmcalCorrectionTask"):
            # TODO: This may not work because the typing isn't correct...
            task.SetForceBeamType(beam_type.value)
            print(f"Setting beam type {beam_type.name} for task {task.GetName()}")

    # Abort if the initialization fails.
    if not analysis_manager.InitAnalysis():
        return

    analysis_manager.PrintStatus()
    analysis_manager.SetUseProgressBar(True, 250)

    # Write out the analysis manager.
    out_file = ROOT.TFile("train.root", "RECREATE")
    out_file.cd()
    analysis_manager.Write()
    out_file.Close()

    return analysis_manager

def start_analysis_manager(analysis_manager: ROOT.AliAnalysisManager,
                           mode: str,
                           n_events: int,
                           input_files: List[Path]) -> None:
    """ Start the given analysis manager.

    Note:
        The only mode current supported is local.

    Args:
        analysis_manager: Analysis manager to start.
        mode: Execution mode for the analysis manager.
        n_events: Number of events to run.
        input_files: Input files to be run over.

    Returns:
        None. The analysis manager is executed.
    """
    if mode == "local":
        print("Starting Analysis...")
        # Create chian from input files
        chain = ROOT.TChain("aodTree")
        for filename in input_files:
            chain.AddFile(str(filename))
        # Start the analysis
        analysis_manager.StartAnalysis("local", chain, n_events)
    elif mode == "grid":
        raise RuntimeError("Not implemented yet!")

def run() -> None:
    # Setup and validation
    task_name = "DynamicalGrooming"
    period = _normalize_period("LHC18b8")
    physics_selection = ROOT.AliVEvent.kAnyINT
    data_type = DataType.AOD
    is_MC = False
    is_run2_data = _is_run2_data(period) if not is_MC else False
    ROOT.AliTrackContainer.SetDefTrackCutsPeriod(period)

    analysis_manager = run_dynamical_grooming(
        task_name=task_name,
        period=period,
        physics_selection=physics_selection,
        data_type=data_type,
        is_MC=is_MC,
    )

    exit(0)

    start_analysis_manager(analysis_manager = analysis_manager,
                           mode = "local",
                           n_events = 1000,
                           input_files = [
                               Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0001/AliAOD.root")
                           ])

if __name__ == "__main__":
    run()
