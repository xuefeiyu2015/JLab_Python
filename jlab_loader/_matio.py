"""
Shared MATLAB v7.3 (HDF5) writer for the segmented .mat products.

MATLAB's export saves every product with ``-v7.3`` and compression on: these are
dense per-trial arrays that can exceed the 2 GB per-variable cap of the default
format, and they are mostly NaN padding, which gzips ~6x. scipy's ``savemat``
writes MAT-v5, which is uncompressed and caps a variable at 4 GB — a real
session's LFP product goes past that — so everything here goes through
hdf5storage instead.

Orientation matters because MATLAB code reads these files back. MATLAB emits the
per-trial and per-unit vectors as COLUMNS (``nTrials x 1`` / ``nUnit x 1``) and
``relative_time`` as a ROW (``1 x maxSamples``); a bare 1-D numpy array would
land as whichever the writer default happens to be. ``orient_product`` pins each
one explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fields MATLAB stores as column vectors, wherever they appear in a product.
_COLUMN_FIELDS = frozenset(
    {
        "alignedrawtime",
        "Session",
        "Trial_number",
        "Channel_Number",
        "Unit_No",
        "ViolationRate",
    }
)
# Fields MATLAB stores as row vectors.
_ROW_FIELDS = frozenset({"relative_time"})


def _orient(name: str, val):
    """Give a 1-D array the shape MATLAB would have written for that field."""
    if isinstance(val, dict):
        return {k: _orient(k, v) for k, v in val.items()}
    if isinstance(val, np.ndarray) and val.ndim == 1:
        if name in _COLUMN_FIELDS:
            return val.reshape(-1, 1)
        if name in _ROW_FIELDS:
            return val.reshape(1, -1)
    return val


def orient_product(product: dict) -> dict:
    """Recursively pin every vector in a segmented product to MATLAB's orientation."""
    return {k: _orient(k, v) for k, v in product.items()}


def save_mat_v73(path: str | Path, mdict: dict) -> None:
    """Write ``mdict`` as a MATLAB v7.3 (HDF5) .mat file via hdf5storage.

    ``store_python_metadata=False`` keeps the structs clean (no extra
    Python-type fields), and ``oned_as='row'`` matches the scipy-based writers
    for any vector ``orient_product`` did not explicitly pin.
    """
    try:
        import hdf5storage
    except ImportError as e:
        raise ImportError(
            "Saving the segmented .mat products needs MATLAB v7.3 (HDF5) "
            "support, which requires the hdf5storage package. Install it with:\n"
            "  pip install hdf5storage"
        ) from e

    mdict = {k: orient_product(v) if isinstance(v, dict) else v
             for k, v in mdict.items()}
    hdf5storage.savemat(
        str(path),
        mdict,
        format="7.3",
        store_python_metadata=False,
        oned_as="row",
        truncate_existing=True,
    )


def _from_h5(obj):
    """Recursively convert an h5py node into numpy arrays / str / dicts.

    MATLAB writes HDF5 with the dimensions reversed, so a MATLAB
    ``nChan x nTrials x nSamp`` array is stored as ``nSamp x nTrials x nChan``.
    Transposing restores MATLAB's orientation. Char arrays come back as uint16
    code points and are decoded to str.
    """
    import h5py
    import numpy as np

    if isinstance(obj, h5py.Group):
        return {k: _from_h5(obj[k]) for k in obj if k != "#refs#"}

    arr = obj[()]
    if isinstance(arr, np.ndarray):
        # MATLAB char arrays are uint16 with a MATLAB_class attr of 'char'.
        if obj.attrs.get("MATLAB_class", b"") == b"char" or (
            arr.dtype == np.uint16 and arr.ndim == 2 and 1 in arr.shape
        ):
            return "".join(chr(c) for c in arr.ravel())
        arr = arr.T  # undo MATLAB's reversed dimension order
        if arr.size == 1:
            return arr.reshape(-1)[0]
        return np.squeeze(arr) if 1 in arr.shape and arr.ndim > 1 else arr
    return arr


def load_product(path: str | Path, var: str | None = None):
    """Read a segmented .mat product (v7.3) back into nested dicts of arrays.

    Works on both this package's exports and MATLAB's own ``*_matlab.mat``
    files, since both are v7.3. scipy's ``loadmat`` cannot read v7.3 at all,
    which is why this exists.

    Parameters
    ----------
    path : str or Path
        The .mat file.
    var : str, optional
        Top-level variable to return (e.g. "eye", "lfp", "online_spike"). When
        omitted, the file's single top-level variable is returned; if there are
        several, a dict of all of them comes back.

    Returns
    -------
    dict — e.g. ``{"data": (nChan, nTrials, nSamp) array, "timeseq": {...},
    "info": {...}}``, in MATLAB's orientation.

    Examples
    --------
    >>> from jlab_loader import load_product
    >>> eye = load_product("Blackrock_2026-07-24_eye.mat")
    >>> eye["data"].shape, eye["timeseq"]["relative_time"].shape
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        top = [k for k in f if k != "#refs#"]
        if var is not None:
            return _from_h5(f[var])
        if len(top) == 1:
            return _from_h5(f[top[0]])
        return {k: _from_h5(f[k]) for k in top}
