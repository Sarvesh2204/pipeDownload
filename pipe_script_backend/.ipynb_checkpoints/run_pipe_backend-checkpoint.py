# ==========================================================
# run_pipe_backend.py
# Final Stable Version
# - Frontend provides final version folder
# - Extract version from folder name
# - Set pps.version correctly
# - Hardcoded reference path
# - Structured CSV logging
# ==========================================================

import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
import os
from pipe import conf
from pipe import PipeParam, PipeControl
import time
import numpy as np
import traceback
from datetime import datetime
from astropy.io import fits
from pathlib import Path


# ==========================================================
# 🔒 HARD-CODED REFERENCE PATH
# ==========================================================

ref_datapath = '/media/team_workspaces/CHEOPS-CNN/PIPE/Ref'

conf.ref_lib_data = ref_datapath
conf.ref_path = ref_datapath
conf.calib_path = ref_datapath


# ==========================================================
# Recursive Finder (for RAW files)
# ==========================================================

def find_file_in_tree(root, substrings, warn=True):
    root = Path(root)

    for substring in substrings:
        for file in root.rglob("*"):
            if file.is_file() and substring in file.name:
                print(f"Found {substring} → {file}")
                return str(file)

    if warn:
        print(f"WARNING: '{substrings[0]}' not found in {root}")

    return None


# ==========================================================
# Logging Function
# ==========================================================

def log_pipe_status(log_file, target, visit, version, status, message=""):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("target,visit,version,timestamp,status,message\n")

    with open(log_file, "a") as f:
        f.write(
            f"{target},{visit},{version},{timestamp},{status},{message}\n"
        )


# ==========================================================
# MAIN EXECUTION FUNCTION
# ==========================================================

def run_pipe(input_dir, run_output_dir, target, visit, nthreads=5):

    STATE = {}

    try:

        input_path = Path(input_dir).resolve()
        output_path = Path(run_output_dir).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found:\n{input_path}")

        if not output_path.exists():
            raise FileNotFoundError(
                f"Output directory does not exist:\n{output_path}"
            )

        # --------------------------------------------------
        # Extract version from folder name
        # --------------------------------------------------

        if not output_path.name.isdigit():
            raise RuntimeError(
                f"Output directory name must be integer version. Got: {output_path.name}"
            )

        version = output_path.name
        STATE["version"] = version

        print("\n========== PIPE START ==========")
        print("Input Path:", input_path)
        print("Output Path:", output_path)
        print("Detected Version:", version)
        print("Reference Path:", ref_datapath)
        print("================================")

        # --------------------------------------------------
        # Initialize PipeParam
        # --------------------------------------------------

        pps = PipeParam(
            target,
            visit,
            outdir=str(output_path),
            version='1',
            datapath=str(input_path),
            calibpath=ref_datapath
        )

        # IMPORTANT: Override version from frontend
        pps.version = int(version)

        # Thread & PIPE settings
        pps.nthreads = nthreads
        pps.psf_min_num = 5
        pps.klip = 1
        pps.fit_bgstars = True

        pps.sa_optimise = True
        pps.sa_test_klips = [1, 3, 5]

        pps.im_optimise = True
        pps.im_test_klips = [1, 3, 5]

        # --------------------------------------------------
        # Override RAW file detection
        # --------------------------------------------------

        print("\nOverriding RAW file detection...")

        pps.file_att = find_file_in_tree(input_path, ("SCI_RAW_Attitude", "attitude."))
        pps.file_sa_raw = find_file_in_tree(input_path, ("RAW_SubArray", "raw."))
        pps.file_hk = find_file_in_tree(input_path, ("SCI_RAW_HkExtended",))
        pps.file_im = find_file_in_tree(input_path, ("SCI_RAW_Imagette", "imagettes."), warn=False)
        pps.file_starcat = find_file_in_tree(input_path, ("EXT_PRE_StarCatalogue", "starcat."))

        if pps.file_att is None or pps.file_sa_raw is None or pps.file_hk is None:
            raise RuntimeError("Missing required RAW files.")

        # --------------------------------------------------
        # Log file (stored one level above version folder)
        # --------------------------------------------------

        log_file = output_path.parent / "pipe_process_log.csv"

        # --------------------------------------------------
        # Run PIPE
        # --------------------------------------------------

        pc = PipeControl(pps)

        try:
            pc.process_eigen()

            log_pipe_status(
                log_file,
                target,
                visit,
                version,
                "Success"
            )

            return {
                "status": "success",
                "output_dir": str(output_path),
                "version": version
            }

        except Exception as e:

            traceback.print_exc()

            # Fallback attempt
            print("Retrying with reduced settings...")

            pps.fit_bgstars = False
            pps.remove_satellites = False

            pc = PipeControl(pps)
            pc.process_eigen()

            log_pipe_status(
                log_file,
                target,
                visit,
                version,
                "Success_Fallback",
                str(e)
            )

            return {
                "status": "success_fallback",
                "output_dir": str(output_path),
                "version": version
            }

    except Exception as e:

        traceback.print_exc()

        try:
            log_file = Path(run_output_dir).parent / "pipe_process_log.csv"
            log_pipe_status(
                log_file,
                target,
                visit,
                STATE.get("version", "NA"),
                "Failed",
                str(e)
            )
        except:
            pass

        return {
            "status": "error",
            "message": str(e)
        }