"""
Analog (NSx) data processing.
Converts raw brpylib analog data into a DataFrame for CSV export.
"""

from __future__ import annotations

import pandas as pd


def parse_analog(analog: dict) -> pd.DataFrame:
    """
    TO-DO:
    parse the raw analog signal into trial based.
    Each row is a trial, each column is a time point.
    --------------------------------------------

    Convert raw analog data from brpylib into a DataFrame.


    Parameters
    ----------
    analog : dict
        The dict returned by brpylib NsxFile.getdata(), containing:
            'data'     : list of arrays, one per segment (channels × samples)
            'elec_ids' : list of electrode IDs

    Returns
    -------
    pandas.DataFrame
        Rows = samples, columns = channel_{elec_id}.
    """
    data = analog["data"][0]          # channels × samples (first segment)
    elec_ids = analog["elec_ids"]
    columns = [f"channel_{eid}" for eid in elec_ids]
    return pd.DataFrame(data.T, columns=columns)  # samples × channels
