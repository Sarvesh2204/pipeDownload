from pathlib import Path
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm


# visit_id
# program_id
#obs_id
# target
# drp_exists
# pipe_exists
# drp_subarray
# drp_imagettes
# pipe_subarray
# pipe_imagettes
# pipe_latest_run
# drp_path
# pipe_path

DEFAULT_PIPE_PATH = "/data/che_pipe/Output_lc/visits"  # change to your real default
DEFAULT_DRP_PATH = "/data/che_data_dev/rep"  # change to your real default


from pathlib import Path
from astropy.io import fits


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return name.lower().replace("_", " ").strip()


def query_drp_visit(visit_path, target_name):
    """
    Return DRP metadata only if target matches.
    Otherwise return None.
    """

    visit_path = Path(visit_path)
    visit_id = visit_path.name

    l2_path = visit_path / "L2" / "SCI_COR_Lightcurve"
    if not l2_path.exists():
        return None

    matching_files = sorted(
        l2_path.glob("*SCI_COR_Lightcurve-DEFAULT_V03*.fits")
    )

    if not matching_files:
        return None

    lc_file = matching_files[-1]

    try:
        with fits.open(lc_file) as hdul:
            header = hdul[1].header

            target = header.get("TARGNAME")
            target = target.strip() if isinstance(target, str) else None

            if not target:
                return None

            normalized_target = normalize_name(target)
            normalized_query = normalize_name(target_name)

            # Safe prefix matching
            if not normalized_target.startswith(normalized_query):
                return None

            return {
                "visit_id": visit_id,
                "program_type": header.get("PROGTYPE"),
                "program_id": header.get("PROG_ID"),
                "req_id":header.get("REQ_ID"),
                "obs_id": header.get("OBSID"),
                "visit_counter": header.get("VISITCTR"),
                "target": target,
                "drp_subarray": True,
                "drp_imagettes": any(
                    l2_path.glob("*SCI_RAW_Imagette*.fits")
                ),
                "drp_path": visit_path
            }

    except Exception as e:
        print(f"Error reading {lc_file}: {e}")
        return None

from pathlib import Path
from joblib import Parallel, delayed


def scan_drp_directory(drp_root, target_name, n_jobs=-1):
    """
    Parallel DRP scan.
    
    Parameters
    ----------
    drp_root : str or Path
    target_name : str
    n_jobs : int
        Number of parallel jobs (-1 uses all cores)
    """

    drp_root = Path(drp_root)

    visit_folders = [
        folder for folder in drp_root.iterdir()
        if folder.is_dir()
    ]

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(query_drp_visit)(visit_folder, target_name)
        for visit_folder in tqdm(visit_folders, desc="Scanning DRP")
    )

    # Keep only valid matches
    drp_dict = {
        r["visit_id"]: r
        for r in results
        if r is not None
    }

    return drp_dict



###################################

### PIPE

def query_pipe_visit(visit_path, target_name):
    """
    Query a single PIPE visit directory.

    Returns metadata only if target matches.
    Otherwise returns None.
    """

    visit_path = Path(visit_path)
    visit_id = visit_path.name

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

    # ---- Step 3: Detect files ----
    subarray_files = list(
        latest_run_folder.glob(f"{visit_id}_*_sa.fits")
    )

    imagettes_exists = any(
        latest_run_folder.glob(f"{visit_id}_*_im.fits")
    )

    if not subarray_files:
        return None

    # Assume first subarray LC for header reading
    lc_file = subarray_files[0]
    try:
        with fits.open(lc_file) as hdul:
            header = hdul[1].header

            target = header.get("TARGNAME")
            target = target.strip() if isinstance(target, str) else None

            if not target:
                return None

            # ---- Safe prefix matching ----
            normalized_target = normalize_name(target)
            normalized_query = normalize_name(target_name)

            if not normalized_target.startswith(normalized_query):
                return None

            return {
                "visit_id": visit_id,
                "program_type": header.get("PROGTYPE"),
                "program_id": header.get("PROG_ID"),
                "req_id":header.get("REQ_ID"),
                "obs_id": header.get("OBSID"),
                "visit_counter": header.get("VISITCTR"),
                "target": target,
                "pipe_subarray": True,
                "pipe_imagettes": imagettes_exists,
                "pipe_latest_run": int(latest_run_folder.name),
                "pipe_path": visit_path
            }

    except Exception as e:
        print(f"Error reading {lc_file}: {e}")
        return None

    
def scan_pipe_directory(pipe_root, target_name, n_jobs=-1):
    """
    Parallel PIPE directory scan.
    Returns dictionary keyed by visit_id.
    """

    pipe_root = Path(pipe_root)

    visit_folders = [
        folder for folder in pipe_root.iterdir()
        if folder.is_dir()
    ]

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(query_pipe_visit)(visit_folder, target_name)
        for visit_folder in tqdm(visit_folders, desc="Scanning PIPE")
    )


    pipe_dict = {
        r["visit_id"]: r
        for r in results
        if r is not None
    }

    return pipe_dict


# ###################

# Combined
import pandas as pd

import pandas as pd


def merge_drp_pipe(df_drp, df_pipe):
    """
    Merge DRP and PIPE DataFrames into one combined comparison table.
    """

    # Safe copies
    df_drp = df_drp.copy() if not df_drp.empty else pd.DataFrame(columns=["visit_id"])
    df_pipe = df_pipe.copy() if not df_pipe.empty else pd.DataFrame(columns=["visit_id"])

    # Add existence flags explicitly as bool
    if not df_drp.empty:
        df_drp["drp_exists"] = True
    if not df_pipe.empty:
        df_pipe["pipe_exists"] = True

    merged = pd.merge(
        df_drp,
        df_pipe,
        on="visit_id",
        how="outer",
        suffixes=("_drp", "_pipe")
    )

    # Ensure existence columns exist
    if "drp_exists" not in merged:
        merged["drp_exists"] = False
    if "pipe_exists" not in merged:
        merged["pipe_exists"] = False

    # Fill NaN explicitly and cast to bool
    merged["drp_exists"] = merged["drp_exists"].fillna(False).astype(bool)
    merged["pipe_exists"] = merged["pipe_exists"].fillna(False).astype(bool)

    # ---- Unify metadata ----
    metadata_cols = ["program_type", "program_id","req_id", "obs_id", "visit_counter", "target"]

    for col in metadata_cols:
        col_drp = f"{col}_drp"
        col_pipe = f"{col}_pipe"

        if col_drp in merged.columns and col_pipe in merged.columns:
            merged[col] = (
                merged[col_drp]
                .combine_first(merged[col_pipe])
                .infer_objects(copy=False)
            )
        elif col_drp in merged.columns:
            merged[col] = merged[col_drp].infer_objects(copy=False)
        elif col_pipe in merged.columns:
            merged[col] = merged[col_pipe].infer_objects(copy=False)
    # Convert numeric metadata columns back to integers
    int_cols = ["program_type", "program_id", "req_id", "obs_id", "visit_counter","visit_id"]

    for col in int_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # Drop duplicate metadata columns
    drop_cols = []
    for col in metadata_cols:
        drop_cols.extend([f"{col}_drp", f"{col}_pipe"])

    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    merged = merged.sort_values("visit_id").reset_index(drop=True)

    return merged

def run_target_query(drp_root, pipe_root, target_name, n_jobs=4):
    """
    Main backend function.
    Returns df_drp, df_pipe, df_combined.
    """

    # ---- DRP ----
    drp_dict = scan_drp_directory(
        drp_root=drp_root,
        target_name=target_name,
        n_jobs=n_jobs
    )

    df_drp = pd.DataFrame(drp_dict.values())
    if not df_drp.empty:
        df_drp = df_drp.sort_values("visit_id").reset_index(drop=True)

    # ---- PIPE ----
    pipe_dict = scan_pipe_directory(
        pipe_root=pipe_root,
        target_name=target_name,
        n_jobs=n_jobs
    )

    df_pipe = pd.DataFrame(pipe_dict.values())
    if not df_pipe.empty:
        df_pipe = df_pipe.sort_values("visit_id").reset_index(drop=True)

    # ---- MERGE ----
    df_combined = merge_drp_pipe(df_drp, df_pipe)

    return df_drp, df_pipe, df_combined