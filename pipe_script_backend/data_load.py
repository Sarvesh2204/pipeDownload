from pathlib import Path
import re
def find_sci_cor_file(input_dir):
    """
    Search for SCI_COR_Lightcurve-DEFAULT_V03*.fits
    directly inside input directory.
    """
    
    input_path = Path(input_dir)
    
    file = next(
        input_path.glob("*SCI_COR_Lightcurve-DEFAULT_V03*.fits"),
        None
    )
    
    if file is None:
        raise FileNotFoundError(
            "No SCI_COR_Lightcurve-DEFAULT_V03 FITS file found."
        )
    
    return file


def extract_visit_id_from_filename(file_path):
    """
    Extract visit ID from CHEOPS SCI_COR filename.

    Example:
    CH_PR100002_TG000901_TU2020-04-18T19-03-30_SCI_COR_Lightcurve-DEFAULT_V0300.fits

    Returns:
    100002000901
    """

    filename = Path(file_path).name

    # Extract PR number
    pr_match = re.search(r"PR(\d+)", filename)

    # Extract TG number
    tg_match = re.search(r"TG(\d+)", filename)

    if not pr_match or not tg_match:
        raise ValueError("Could not extract PR or TG from filename.")

    pr_number = pr_match.group(1)
    tg_number = tg_match.group(1)

    visit_id = f"{pr_number}{tg_number}"

    return visit_id