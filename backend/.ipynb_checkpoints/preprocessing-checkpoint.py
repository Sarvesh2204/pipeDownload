import os
import numpy as np
from astropy.io import fits
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.auto import tqdm
import deroll
import warnings
warnings.filterwarnings("ignore")





def mad_mask(flux, nmad):
    median = np.median(flux)
    nf = flux / median
    mad = np.median(np.abs(np.diff(nf)))
    mask = np.abs(nf - 1) < nmad * mad
    ylim = (1-(nmad*mad),1+(nmad*mad))
    return mask,ylim

def _process_single_visit(row, prefix, aperture, pipe_mode, settings):

    visit_id = row["visit_id"]
    detrended = False
    outlier_clipped = False
    ylim = None

    # ---------------------------------
    # Select file
    # ---------------------------------
    if prefix == "DRP":
        file_path, aperture_used = select_drp_file(row, aperture)
        pipe_type = None
        run_number = None
    else:
        file_path, pipe_type, run_number = select_pipe_file(row, pipe_mode)
        aperture_used = None

    if file_path is None:
        return None

    # ---------------------------------
    # Load arrays (memory mapped)
    # ---------------------------------
    time, flux, roll, status = load_fits_arrays(file_path, prefix)

    if time is None or flux is None or len(time) == 0:
        return None

    # ---------------------------------
    # Initialize mask
    # ---------------------------------
    final_mask = np.ones_like(flux, dtype=bool)

    # ---------------------------------
    # STATUS mask (DRP only)
    # ---------------------------------
    if status is not None:
        final_mask &= (status == 0)

    # ---------------------------------
    # NO PREPROCESSING
    # ---------------------------------
    if settings.get("no_preproc", False):

        time = time[final_mask]
        flux = flux[final_mask]

        if len(flux) == 0:
            return None

        flux = flux / np.median(flux)

        return {
            "visit_id": visit_id,
            "program_type": row["program_type"],
            "program_id": row["program_id"],
            "req_id": row["req_id"],
            "file_path": str(file_path),
            "aperture_used": aperture_used,
            "pipe_type": pipe_type,
            "run_number": run_number,
            "time": time,
            "flux": flux,
            "detrended": False,
            "outlier_clipped": False,
            "y_lim": None,
        }

    # ---------------------------------
    # Initial Outlier Clipping
    # ---------------------------------
    if settings.get("apply_outlier", False):
        out_mask, ylim = mad_mask(flux, settings["nmad_outlier"])
        final_mask &= out_mask
        outlier_clipped = True

    # ---------------------------------
    # Detrending
    # ---------------------------------
    if settings.get("apply_detrend", False) and roll is not None:

        try:
            trend = deroll.deroll(
                roll,
                time,
                flux,
                final_mask,
                tdens=settings["tdens"]
            )

            flux = flux / trend
            detrended = True

        except Exception:
            detrended = False

    # ---------------------------------
    # Residual Clipping
    # ---------------------------------
    if settings.get("apply_residual", False):
        residual_mask, ylim = mad_mask(flux, settings["nmad_residual"])
        final_mask &= residual_mask
        outlier_clipped = True

    # ---------------------------------
    # Apply final mask
    # ---------------------------------
    time = time[final_mask]
    flux = flux[final_mask]

    if len(flux) == 0:
        return None

    # ---------------------------------
    # Normalize
    # ---------------------------------
    flux = flux / np.median(flux)

    return {
        "visit_id": visit_id,
        "program_type": row["program_type"],
        "program_id": row["program_id"],
        "req_id": row["req_id"],
        "file_path": str(file_path),
        "aperture_used": aperture_used,
        "pipe_type": pipe_type,
        "run_number": run_number,
        "time": time,
        "flux": flux,
        "detrended": detrended,
        "outlier_clipped": outlier_clipped,
        "y_lim": ylim,
    }


def run_preprocessing(df_selected, prefix, aperture=None, pipe_mode=None, settings=None):

    if df_selected is None or df_selected.empty:
        return None

    if prefix == "DRP":
        df_pipeline = df_selected[df_selected["drp_exists"] == True]
    else:
        df_pipeline = df_selected[df_selected["pipe_exists"] == True]

    if df_pipeline.empty:
        return None

    rows = df_pipeline.to_dict("records")

    # ---------------------------------------------
    # ---------------------------------------------
    max_workers = min(4, os.cpu_count())  # 4 is optimal for FITS I/O
    n_jobs = min(len(rows), max_workers)

    results = Parallel(
        n_jobs=n_jobs,
        backend="threading",   # better for I/O + NumPy
        prefer="threads"
    )(
        delayed(_process_single_visit)(
            row, prefix, aperture, pipe_mode, settings
        )
        for row in tqdm(rows,desc=f"{prefix} Preprocessing ")
    )

    results = [r for r in results if r is not None]

    if not results:
        return None

    return pd.DataFrame(results)



def select_drp_file(row, aperture):

    folder = Path(row["drp_path"]) / "L2" / "SCI_COR_Lightcurve"

    if not folder.exists():
        return None, None

    if aperture == "DEFAULT":
        aperture_string = "SCI_COR_Lightcurve-DEFAULT_V03"
    else:
        aperture_string = f"SCI_COR_Lightcurve-R{aperture}_V03"

    file = next(folder.glob(f"*{aperture_string}*.fits"), None)

    if file is None:
        return None, None

    return file, aperture

def select_pipe_file(row, pipe_mode):

    try:
        visit_path = Path(row["pipe_path"])
    except Exception:
        return None, None, None

    if not visit_path.exists():
        return None, None, None

    visit_id = visit_path.name

    run_folders = [
        f for f in visit_path.iterdir()
        if f.is_dir() and f.name.isdigit()
    ]

    if not run_folders:
        return None, None, None

    latest_run = max(run_folders, key=lambda x: int(x.name))
    run_number = int(latest_run.name)

    sa_file = next(latest_run.glob(f"{visit_id}_*_sa.fits"), None)
    im_file = next(latest_run.glob(f"{visit_id}_*_im.fits"), None)

    if pipe_mode == "sa":
        return (sa_file, "sa", run_number) if sa_file else (None, None, None)

    if pipe_mode == "im":
        return (im_file, "im", run_number) if im_file else (None, None, None)

    if pipe_mode == "either":
        if im_file:
            return im_file, "im", run_number
        if sa_file:
            return sa_file, "sa", run_number

    return None, None, None



def load_fits_arrays(file_path, prefix):

    from astropy.io import fits

    try:
        with fits.open(str(file_path), memmap=True) as hdul:

            data = hdul[1].data
            colnames = data.columns.names

            if "BJD_TIME" in colnames:
                time = np.array(data["BJD_TIME"])
            else:
                time = np.array(data["TIME"])

            flux = np.array(data["FLUX"])

            if "ROLL" in colnames:
                roll = np.array(data["ROLL"])
            elif "ROLL_ANGLE" in colnames:
                roll = np.array(data["ROLL_ANGLE"])
            else:
                roll = None

            if prefix == "DRP" and "STATUS" in colnames:
                status = np.array(data["STATUS"])
            else:
                status = None

    except Exception:
        return None, None, None, None

    return time, flux, roll, status