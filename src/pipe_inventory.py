from pathlib import Path
import pandas as pd
import numpy as np

from joblib import Parallel, delayed

import zipfile
import shutil
import os
from astropy.io import fits
from datetime import datetime
from tqdm import tqdm



def query_pipe_visit_time(visit_path):
    """
    Extract timing metadata from a single PIPE visit.

    Returns:
        dict with timing info or None
    """

    visit_path = Path(visit_path)
    visit_id = visit_path.name
    visit_info = parse_visitid(visit_id)

    # ---- Step 1: Find run folders ----
    run_folders = [
        folder for folder in visit_path.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    ]

    if not run_folders:
        return None

    # ---- Step 2: Select latest run ----
    latest_run_folder = max(
        run_folders,
        key=lambda x: int(x.name)
    )

    # ---- Step 3: Detect subarray photometry ----
    subarray_files = list(
        latest_run_folder.glob(f"{visit_id}_*_sa.fits")
    )

    if not subarray_files:
        return None

    lc_file = subarray_files[0]

    try:
        with fits.open(lc_file) as hdul:
            header = hdul[1].header

            t_start = header.get("T_STRT_U")
            t_stop = header.get("T_STOP_U")

            if not t_start or not t_stop:
                return None

            # Convert to datetime
            t_start_dt = datetime.fromisoformat(t_start)
            t_stop_dt = datetime.fromisoformat(t_stop)

            return {
                "ProgramType": visit_info["ProgramType"],
                "ProgramID": visit_info["ProgramID"],
                "ObsID": visit_info["ObsID"],
                "VisitCounter": visit_info["VisitCounter"],
                "visit_id": visit_id,
                "pipe_path": str(visit_path),
                "latest_run": int(latest_run_folder.name),
                "t_start_utc": t_start_dt,
                "t_stop_utc": t_stop_dt,
                "duration_hours": (t_stop_dt - t_start_dt).total_seconds() / 3600,
                "Targets":header.get("TARGNAME").strip() if header.get("TARGNAME") else header.get("TARGNAME")
            }

    except Exception as e:
        print(f"Error reading {lc_file}: {e}")
        return None
    
    
    
    

def build_pipe_inventory(pipe_root, n_jobs=-1):
    """
    Parallel scan of PIPE root directory.

    Returns:
        dict keyed by visit_id
    """

    pipe_root = Path(pipe_root)

    visit_folders = [
        folder for folder in pipe_root.iterdir()
        if folder.is_dir()
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(query_pipe_visit_time)(visit_folder)
        for visit_folder in tqdm(visit_folders, desc="Scanning PIPE visits")
    )

    pipe_dict = {
        r["visit_id"]: r
        for r in results
        if r is not None
    }

    return pipe_dict



def pipe_dict_to_sorted_df(pipe_dict):
    """
    Convert PIPE dictionary to sorted DataFrame.
    """

    df = pd.DataFrame.from_dict(pipe_dict, orient="index")

    df = df.sort_values("t_start_utc").reset_index(drop=True)

    return df


def parse_visitid(visit_id):

    visit_id = str(visit_id)

    return {

        "ProgramType":
            visit_id[:2],

        "ProgramID":
            visit_id[2:6],

        "ObsID":
            visit_id[6:10],

        "VisitCounter":
            visit_id[10:12]
    }

def filter_inventory(
    df,
    program_type=None,
    program_id=None,
    obsid=None,
    visit_counter=None,
    visit_id=None,
    target=None,
):
    """
    Filter PIPE inventory dataframe.

    Identifier fields:
        ProgramType  -> 2 digits
        ProgramID    -> 4 digits
        ObsID        -> 4 digits
        VisitCounter -> 2 digits
        visit_id     -> exact 12-digit visit ID

    Targets use partial text matching.

    Multiple filters are combined using AND logic.
    """

    result = df.copy()

    # ---------------------------
    # Program Type (2 digits)
    # ---------------------------
    if program_type is not None:
        program_type = str(program_type).zfill(2)

        result = result[
            result["ProgramType"] == program_type
        ]

    # ---------------------------
    # Program ID (4 digits)
    # ---------------------------
    if program_id is not None:
        program_id = str(program_id).zfill(4)

        result = result[
            result["ProgramID"] == program_id
        ]

    # ---------------------------
    # ObsID (4 digits)
    # ---------------------------
    if obsid is not None:
        obsid = str(obsid).zfill(4)

        result = result[
            result["ObsID"] == obsid
        ]

    # ---------------------------
    # Visit Counter (2 digits)
    # ---------------------------
    if visit_counter is not None:
        visit_counter = str(visit_counter).zfill(2)

        result = result[
            result["VisitCounter"] == visit_counter
        ]

    # ---------------------------
    # Full Visit ID (12 digits)
    # ---------------------------
    if visit_id is not None:
        visit_id = str(visit_id)

        result = result[
            result["visit_id"] == visit_id
        ]

    # ---------------------------
    # Target Name
    # ---------------------------
    if target:
        result = result[
            result["Targets"].str.contains(
                str(target),
                case=False,
                na=False
            )
        ]

    return result



def build_pipe_inventory_df(pipe_root, n_jobs=-1):

    pipe_dict = build_pipe_inventory(
        pipe_root,
        n_jobs=n_jobs
    )

    return pipe_dict_to_sorted_df(pipe_dict)


def build_visit_options(df):
    """
    Build user-friendly labels for visit selection.
    """

    return [
        (
            f"PT={row.ProgramType} | "
            f"PID={row.ProgramID} | "
            f"OBS={row.ObsID} | "
            f"VC={row.VisitCounter} | "
            f"{row.visit_id} | "
            f"{row.Targets} | ",
            row.visit_id
        )
        for _, row in df.iterrows()
    ]


from pathlib import Path
from datetime import datetime
import zipfile


def create_visit_zip(
    df_inventory,
    selected_visits,
    output_zip="current_download.zip"
):
    """
    Create ZIP containing selected visit folders.

    Returns
    -------
    str
        Path to created ZIP file.
    """

    # -----------------------------------
    # Remove old PIPE download ZIPs
    # -----------------------------------
    ZIP_DIR = Path("/media/home/my_workspace/pipeDown-tool")
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for old_zip in Path(ZIP_DIR).glob("pipe_download_*.zip"):
        old_zip.unlink()

    # -----------------------------------
    # ZIP naming
    # -----------------------------------

    if len(selected_visits) == 1:

        zip_name = ZIP_DIR/f"{selected_visits[0]}.zip"

    else:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        zip_name = (
            f"pipe_download_{timestamp}.zip"
        )

    # -----------------------------------
    # Build ZIP
    # -----------------------------------

    with zipfile.ZipFile(
        zip_name,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zf:

        for visit_id in selected_visits:

            visit_path = Path(
                df_inventory.loc[
                    df_inventory["visit_id"] == visit_id,
                    "pipe_path"
                ].iloc[0]
            )

            for file in visit_path.rglob("*"):

                if file.is_file():

                    zf.write(
                        file,
                        arcname=file.relative_to(
                            visit_path.parent
                        )
                    )

    return zip_name


# def load_or_update_inventory(
#     pipe_root,
#     master_cache,
#     local_cache
# )

def load_inventory_from_cache(
    master_cache,
    local_cache
):
    """
    Load inventory from cache.

    Priority:
        1. Local cache
        2. Master cache

    Returns
    -------
    pd.DataFrame
    """

    local_cache = Path(local_cache)
    master_cache = Path(master_cache)

    if local_cache.exists():

        print(
            f"Loading local cache: {local_cache}"
        )

        return pd.read_pickle(
            local_cache
        )

    if master_cache.exists():

        print(
            f"Loading master cache: {master_cache}"
        )

        return pd.read_pickle(
            master_cache
        )

    return None


from pathlib import Path
import pandas as pd


def load_cache(cache_path):

    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None

    return pd.read_pickle(cache_path)


def save_cache(df, cache_path):

    cache_path = Path(cache_path)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_pickle(cache_path)
    
    
def get_pipe_visit_ids(pipe_root):
    """
    Return all visit IDs currently present
    in the PIPE visit directory.
    """

    pipe_root = Path(pipe_root)

    return {
        folder.name
        for folder in pipe_root.iterdir()
        if folder.is_dir()
    }



def scan_specific_visits(
    pipe_root,
    visit_ids,
    n_jobs=-1
):
    """
    Scan only the specified visit IDs.

    Parameters
    ----------
    pipe_root : str or Path

    visit_ids : iterable
        Visit IDs to scan.

    Returns
    -------
    pd.DataFrame
    """

    pipe_root = Path(pipe_root)

    visit_folders = [
        pipe_root / visit_id
        for visit_id in visit_ids
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(query_pipe_visit_time)(visit_folder)
        for visit_folder in tqdm(
            visit_folders,
            desc="Scanning new visits"
        )
    )

    results = [
        r
        for r in results
        if r is not None
    ]

    if len(results) == 0:

        return pd.DataFrame()

    df = pd.DataFrame(results)

    return (
        df
        .sort_values("t_start_utc")
        .reset_index(drop=True)
    )


def load_or_update_inventory(
    pipe_root,
    local_cache,
    n_jobs=-1
):
    """
    Load inventory from cache and
    append any newly discovered visits.
    """

    pipe_root = Path(pipe_root)
    local_cache = Path(local_cache)

    # -------------------------
    # Load cache
    # -------------------------

    if local_cache.exists():

        print(
            f"Loading cache: {local_cache}"
        )

        df_inventory = pd.read_pickle(
            local_cache
        )

    else:

        print(
            "No cache found."
        )

        print(
            "Building inventory..."
        )

        df_inventory = build_pipe_inventory_df(
            pipe_root,
            n_jobs=n_jobs
        )

        save_cache(
            df_inventory,
            local_cache
        )

        return df_inventory

    # -------------------------
    # Detect new visits
    # -------------------------

    pipe_ids = get_pipe_visit_ids(
        pipe_root
    )

    cached_ids = set(
        df_inventory["visit_id"]
    )

    candidate_ids = pipe_ids - cached_ids

    print(
        f"{len(candidate_ids)} candidate visits found"
    )

    # -------------------------
    # Scan candidates
    # -------------------------

    if len(candidate_ids) > 0:

        new_df = scan_specific_visits(
            pipe_root,
            candidate_ids,
            n_jobs=n_jobs
        )
        valid_count = len(new_df)
        invalid_count = len(candidate_ids) - valid_count

        print(
            f"{valid_count} valid visits discovered"
        )

        print(
            f"{invalid_count} visits skipped "
            "(missing run folder, SA file, or required headers)"
        )

        if len(new_df) > 0:

            df_inventory = pd.concat(
                [
                    df_inventory,
                    new_df
                ],
                ignore_index=True
            )

            df_inventory = (
                df_inventory
                .sort_values(
                    "t_start_utc"
                )
                .reset_index(
                    drop=True
                )
            )

            save_cache(
                df_inventory,
                local_cache
            )

            print(
                f"Added {len(new_df)} new visits"
            )
        

    return df_inventory