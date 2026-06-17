import numpy as np
from tqdm.auto import tqdm


def phase_fold(time, flux, period, t0):
    """
    Compute phase-folded lightcurve.

    Parameters
    ----------
    time : array
    flux : array
    period : float (days)
    t0 : float (mid transit time)

    Returns
    -------
    phase : array
    flux : array (sorted by phase)
    """

    phase = ((time - t0) / period) % 1
    phase = phase - 0.5

    sort_idx = np.argsort(phase)

    return phase[sort_idx], flux[sort_idx]

import pandas as pd

def stack_phase_fold(df_prog, period, t0):
    """
    Stack and phase-fold multiple visits of one program.

    Parameters
    ----------
    df_prog : DataFrame
        Processed dataframe filtered to one program
    period : float
        Orbital period in days
    t0 : float
        Reference mid-transit time (BJD)

    Returns
    -------
    phase : np.ndarray
    flux  : np.ndarray
    """

    all_phase = []
    all_flux  = []

    for _, row in df_prog.iterrows():

        time = np.asarray(row["time"])
        flux = np.asarray(row["flux"])

        # Compute phase in range [-0.5, 0.5]
        phase = ((time - t0) / period) % 1
        phase[phase > 0.5] -= 1

        all_phase.append(phase)
        all_flux.append(flux)

    phase = np.concatenate(all_phase)
    flux  = np.concatenate(all_flux)

    # Sort by phase
    sort_idx = np.argsort(phase)

    return phase[sort_idx], flux[sort_idx]



from joblib import Parallel, delayed
import os


def precompute_phase_all_programs(df_sel, ephem_df, prefix):

    """
    Precompute stacked phase-folded lightcurves per (program_type, program_id).

    Returns:
        dict:
            {
                (program_type, program_id): (phase_array, flux_array),
                ...
            }
    """

    # --------------------------------------------------
    # 1️⃣ Safety Checks
    # --------------------------------------------------

    if df_sel is None or df_sel.empty:
        return {}

    if ephem_df is None or ephem_df.empty:
        return {}

    # --------------------------------------------------
    # 2️⃣ Ensure String Keys (avoid full copy)
    # --------------------------------------------------
    df_sel["program_type"] = df_sel["program_type"].astype(str).str.strip()
    df_sel["program_id"]   = df_sel["program_id"].astype(str).str.strip()
    df_sel["req_id"]   = df_sel["req_id"].astype(str).str.strip()


    ephem_df["program_type"] = ephem_df["program_type"].astype(str).str.strip()
    ephem_df["program_id"]   = ephem_df["program_id"].astype(str).str.strip()
    ephem_df["req_id"]   = ephem_df["req_id"].astype(str).str.strip()


    # --------------------------------------------------
    # 3️⃣ Group Visit Data ONCE
    # --------------------------------------------------

    grouped_data = dict(
        tuple(df_sel.groupby(["program_type", "program_id","req_id"]))
    )
    print("Number groups =", len(grouped_data))


    if not grouped_data:
        return {}

    # --------------------------------------------------
    # 4️⃣ Ephemeris Lookup Dictionary
    # --------------------------------------------------

    ephem_lookup = {
        (row.program_type, row.program_id,row.req_id): (
            float(row.period_days),
            float(row.mid_transit_time)
        )
        for row in ephem_df.itertuples(index=False)
    }

    # --------------------------------------------------
    # 5️⃣ Worker Function
    # --------------------------------------------------

    def compute_one(key):

        if key not in ephem_lookup:
            return (key, None)

        P, t0 = ephem_lookup[key]
        df_prog = grouped_data[key]

        try:
            phase, flux = stack_phase_fold(df_prog, P, t0)
            return (key, (phase, flux))
        except Exception:
            return (key, None)

    keys = list(grouped_data.keys())

    # --------------------------------------------------
    # 6️⃣ Smart Parallel Strategy
    # --------------------------------------------------

    if len(keys) == 1:
        # Single-core faster for tiny workloads
        results = [compute_one(keys[0])]
    else:
        max_cores = os.cpu_count() or 1
        n_jobs = min(len(keys), max_cores, 8)  # cap at 8 for stability

        results = Parallel(
            n_jobs=n_jobs,
            prefer="threads"
        )(
            delayed(compute_one)(key)
            for key in keys
        )

    # --------------------------------------------------
    # 7️⃣ Build Phase Cache Dictionary
    # --------------------------------------------------

    phase_cache = {
        key: value
        for key, value in results
        if value is not None
    }

    return phase_cache