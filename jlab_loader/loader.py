"""
BlackRockLoader — main user-facing class.

Reads the role-split Blackrock files for one session and exports the same
products as the MATLAB BlackrockLoader / BackRockFileLoader.m, with the same
filenames minus the `_matlab` suffix:
  NSP-*.nev  -> experiment comments + comment timing  (-> expmeta .txt, trials .csv)
  NSP-*.ns2  -> eye data (+ photodiode on ch 4-6)     (-> _eye.mat, _photodiode.mat)
  Hub-*.ns2  -> local field potential                 (-> _lfp.mat)
  HUB-*.nev  -> online spike timing                   (-> _spikes.mat)
  HUB-*.nev  -> online spike waveforms (opt-in)       (-> _spikes_waveform.mat)
HUB-*.nev is also the legacy fallback for comments (early sessions wrote comments
and spikes both to the HUB file).

Precision rule, copied from the MATLAB class header: VOLTAGES are single, TIMES
are always double. Segmented sample values, spike waveforms and the 0/1 raster
are float32; every timestamp is float64.

Continuous streams are read as RAW int16 and carry a per-channel `uv_per_digit`
factor alongside; the segmenter applies it only to the samples it keeps. Reading
in µV instead would force the whole multi-hundred-MB array to float64.

Spike pipeline (source-agnostic):
  A source-specific *reader* produces the common per-spike arrays
  ``(times, channel, unit[, waveform, ticks])`` -> the shared, source-agnostic
  ``segment_spikes`` / ``segment_waveforms`` align them to trials -> export.
  Today the only reader is ``_read_online_spikes`` (online HUB NEV). Offline
  (sorted) spikes drop in later by adding a parallel ``_read_offline_spikes`` +
  ``load_offline_spikes`` / ``export_offline_spikes`` that reuse the SAME
  segmenters and the shared ``drop_units`` filter, writing ``offline_spike`` /
  ``offline_spike_waveform`` outputs. Only the information source differs.
"""

from __future__ import annotations

from pathlib import Path

from . import _constants as C
from ._continuous import (
    align_ptp_clock_drift,
    clamp_channels,
    segment_continuous,
    subset_channels,
)
from ._exporter import write_expmeta, write_trials_csv
from ._features import compute_derived_features
from ._matio import save_mat_v73
from ._parser import parse_comments
from ._spikes import drop_units, segment_spikes
from ._waveforms import segment_waveforms


def _shared_time_res(a: float | None, b: float | None) -> float | None:
    """The common tick rate of two files, or None if they do not share one.

    Trial Start/End ticks come from the comment .nev; the samples they are
    matched against come from an .nsx, and the spikes from another .nev. Doing
    the window arithmetic in integer ticks is only valid when those clocks tick
    at the same rate (they do on PTP rigs — 1e9 everywhere). Returning None
    makes the caller fall back to the float path rather than mix clocks.
    """
    if not a or not b or float(a) != float(b):
        return None
    return float(a)


def _patch_numpy_for_brpylib() -> None:
    """brpylib calls numpy chararray.tostring(), removed in numpy 2.0."""
    import numpy as np

    if not hasattr(np.chararray, "tostring"):
        np.chararray.tostring = np.chararray.tobytes  # type: ignore[attr-defined]


class BlackRockLoader:
    """
    Load a Blackrock session and export experiment metadata + trial data, plus
    optional trial-segmented eye / LFP / photodiode (.ns2) and online-spike
    (HUB .nev) products.

    Construction is folder-based, mirroring BackRockFileLoader.m: no filenames
    needed. `session_path` is the directory that holds the `raw_data` and
    `export_data` sub-folders (assemble it in the notebook, e.g.
    ``Path(Basic_Path) / "Monkey <name>" / Location``). Files are picked from
    ``session_path / data_type / date`` by role prefix (the largest match wins).
    Output goes to ``session_path / output_folder / date``.

    Every stream is gated by its own flag; if requested but the prefixed file is
    absent it is skipped with a message (soft failure, like the MATLAB loader).
    Missing comments is a hard error.

    Parameters
    ----------
    session_path : str or Path
        Directory containing the `data_type` and `output_folder` sub-folders.
    date : str
        Year-month-date folder, e.g. "2026-06-17". Also used in output filenames.
    data_type : str
        Raw-data sub-folder name. Default "raw_data".
    nev_filename : str, optional
        Explicit comment .nev filename instead of prefix auto-pick.
    load_eye : bool
        If True, load and segment the NSP-*.ns2 eye file.
    eye_marker : str
        Eye NSx extension without the dot. Default "ns2".
    eye_filename : str, optional
        Explicit eye filename instead of prefix auto-pick.
    load_lfp : bool
        If True, load and segment the Hub-*.ns2 LFP file.
    load_photodiode : bool
        If True, produce the photodiode product. By default it rides on rows
        `photodiode_channels` of the eye ns2, so it costs no extra read and no
        extra segmentation pass; it falls back to a dedicated NSP-*.ns4 when the
        eye stream has too few channels.
    eye_channels, photodiode_channels : sequence of int
        1-based row indices in the eye ns2 (MATLAB convention). Defaults
        (1, 2, 3) and (4, 5, 6).
    photodiode_use_separate_file : bool
        Skip the channel split and read the dedicated NSP-*.ns4 directly.
    to_microvolts : bool
        Load the continuous streams directly in true microVolts. Each NSx
        channel's scale factor produces the unit named in its header, and NPMK
        (so MATLAB) calls the result µV regardless: correct for the LFP
        (`Units = uV`) but off by 1000x for the analog-input eye/photodiode
        channels (`Units = mV`). Default False keeps MATLAB's numbers exactly
        and records the true unit in `info.Unit`. Set True to fold the header
        unit into the scale factor so every product really is µV — physically
        consistent, but eye/photodiode then differ from MATLAB by 1000x.
    load_online_spikes : bool
        If True, load and rasterize online spikes from the HUB-*.nev file.
    spike_nev_filename : str, optional
        Explicit spike .nev filename instead of prefix auto-pick.
    load_online_wave : bool
        If True, also read per-spike waveforms from the HUB-*.nev file and
        segment them into a dense per-trial array (microVolts). Off by default
        because the dense array can be large; implies reading spikes (waveforms
        come from the same file). Also fills info.MeanWaveform on the spike product.
    include_unsorted : bool
        If True, keep every online-spike "unit", including unsorted threshold
        crossings (unit 0) and noise (unit 255). Default False -> only sorted
        units are kept, which applies to BOTH the spike raster and the waveform
        export and keeps the dense waveform array small.
    pre_ms, post_ms : float
        Trial-segmentation buffers (ms): window = [Start - pre, End + post].
        Default 500 each.
    bin_ms : float
        Spike-raster bin width (ms). Default 1.
    isi_violation_ms : float
        Refractory window (ms) for info.ViolationRate. Default 1.
    output_folder : str
        Export sub-folder name. Default "export_data".
    verbose : bool
        Print progress messages. Default True.

    Examples
    --------
    >>> from jlab_loader import BlackRockLoader
    >>> from pathlib import Path
    >>> session = Path(Basic_Path) / "Monkey Porthos" / "in_lab"

    >>> # comments only
    >>> loader = BlackRockLoader(session, "2026-06-17")
    >>> output_dir, files = loader.run()

    >>> # everything
    >>> loader = BlackRockLoader(session, "2026-06-17", load_eye=True,
    ...                          load_lfp=True, load_photodiode=True,
    ...                          load_online_spikes=True, load_online_wave=True)
    >>> output_dir, files = loader.run()

    >>> # several dates at once (or run_batch(session) to do every date folder)
    >>> report = BlackRockLoader.run_batch(session, dates=["2026-06-17", "2026-06-18"],
    ...                                    load_eye=True, load_online_spikes=True)
    """

    def __init__(
        self,
        session_path: str | Path,
        date: str,
        *,
        data_type: str = "raw_data",
        nev_filename: str | None = None,
        load_eye: bool = False,
        eye_marker: str = C.EYE_IDENTIFIER,
        eye_filename: str | None = None,
        load_lfp: bool = False,
        load_photodiode: bool = False,
        eye_channels=C.EYE_CHANNELS,
        photodiode_channels=C.PHOTODIODE_CHANNELS,
        photodiode_use_separate_file: bool = False,
        to_microvolts: bool = False,
        load_online_spikes: bool = False,
        spike_nev_filename: str | None = None,
        load_online_wave: bool = False,
        include_unsorted: bool = False,
        pre_ms: float = C.SEGMENT_PRE_MS,
        post_ms: float = C.SEGMENT_POST_MS,
        bin_ms: float = C.SEGMENT_BIN_MS,
        isi_violation_ms: float = C.SPIKE_ISI_VIOLATION_MS,
        output_folder: str = "export_data",
        verbose: bool = True,
    ) -> None:
        self.session_path = Path(session_path)
        self.date = date
        self.date_str = date
        self.verbose = verbose
        self.data_folder = self.session_path / data_type / date

        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.bin_ms = bin_ms
        self.isi_violation_ms = isi_violation_ms

        self.eye_channels = tuple(eye_channels)
        self.photodiode_channels = tuple(photodiode_channels)
        self.photodiode_use_separate_file = photodiode_use_separate_file
        self.to_microvolts = to_microvolts

        # ── Comment .nev candidates (NSP primary, HUB legacy fallback) ────────
        # The comment source is finalised at load time (the first candidate that
        # actually carries comments wins), matching MATLAB's NSP -> HUB fallback.
        if nev_filename is not None:
            path = self.data_folder / nev_filename
            if not path.exists():
                raise FileNotFoundError(f"NEV file not found: {path}")
            self._comment_candidates: list[Path] = [path]
        else:
            self._comment_candidates = [
                p
                for p in (
                    self._pick_by_prefix(self.data_folder, "*.nev", C.COMMENT_PREFIX_PRIMARY),
                    self._pick_by_prefix(self.data_folder, "*.nev", C.COMMENT_PREFIX_LEGACY),
                )
                if p is not None
            ]
        self.comment_nev_path: Path | None = None  # resolved in load_nev()

        # ── Eye (NSP-*.ns2), gated ───────────────────────────────────────────
        self.eye_marker = eye_marker.lstrip(".").lower()
        self.eye_path: Path | None = None
        if load_eye:
            if eye_filename is not None:
                path = self.data_folder / eye_filename
                if not path.exists():
                    raise FileNotFoundError(f"Eye file not found: {path}")
                self.eye_path = path
            else:
                self.eye_path = self._pick_by_prefix(
                    self.data_folder, f"*.{self.eye_marker}", C.EYE_PREFIX
                )
            if self.eye_path is None and verbose:
                print(
                    f"No {C.EYE_PREFIX}-*.{self.eye_marker} eye file found; "
                    "skipping eye load."
                )

        # ── LFP (Hub-*.ns2), gated. Same extension as the eye file, different
        # prefix, so it resolves independently. ─────────────────────────────
        self.lfp_path: Path | None = None
        if load_lfp:
            self.lfp_path = self._pick_by_prefix(
                self.data_folder, f"*.{C.LFP_IDENTIFIER}", C.LFP_PREFIX
            )
            if self.lfp_path is None and verbose:
                print(
                    f"No {C.LFP_PREFIX}-*.{C.LFP_IDENTIFIER} LFP file found; "
                    "skipping LFP load."
                )

        # ── Photodiode, gated. Resolved at load time: it normally rides in the
        # eye ns2 and only needs its own file as a fallback. ─────────────────
        # Flag is private so it does not shadow the load_photodiode() method.
        self._want_photodiode = load_photodiode
        self.photodiode_path: Path | None = None
        self.photodiode_from_eye = False

        # ── Online spikes / waveforms (HUB-*.nev), gated ─────────────────────
        # Waveforms come from the same HUB-*.nev as spikes, so load_online_wave
        # implies resolving (and reading) the spike file. The spike raster is
        # only exported when load_online_spikes was requested in its own right.
        self._export_online_spikes = load_online_spikes
        self.load_online_wave = load_online_wave
        self.include_unsorted = include_unsorted
        self.spike_nev_path: Path | None = None
        if load_online_spikes or load_online_wave:
            if spike_nev_filename is not None:
                path = self.data_folder / spike_nev_filename
                if not path.exists():
                    raise FileNotFoundError(f"Spike NEV file not found: {path}")
                self.spike_nev_path = path
            else:
                self.spike_nev_path = self._pick_by_prefix(
                    self.data_folder, "*.nev", C.SPIKE_PREFIX
                )
            if self.spike_nev_path is None and verbose:
                print(
                    f"No {C.SPIKE_PREFIX}-*.nev spike file found; "
                    "skipping spike/waveform load."
                )

        self.output_dir = self.session_path / output_folder / date

        if verbose:
            cand = ", ".join(p.name for p in self._comment_candidates) or "(none)"
            print(f"Comment NEV   : {cand}")
            if self.eye_path is not None:
                print(f"Eye file      : {self.eye_path.name}")
            if self.lfp_path is not None:
                print(f"LFP file      : {self.lfp_path.name}")
            if self.spike_nev_path is not None:
                role = "spikes+waveforms" if self.load_online_wave else "spikes"
                print(f"Spike NEV     : {self.spike_nev_path.name} ({role})")
            print(f"Output dir    : {self.output_dir}")

        # Parsed products
        self.experiments: list[dict] | None = None
        self.trials: list[dict] | None = None
        self.trial_ticks: dict | None = None
        self.start_ticks = None
        self.end_ticks = None
        self.comment_time_res: float | None = None
        # Raw continuous streams, keyed by role
        self.raw: dict[str, dict] = {}
        # Segmented continuous products
        self.eye: dict | None = None
        self.lfp: dict | None = None
        self.photodiode: dict | None = None
        # Spikes
        self.spike_times = None
        self.spike_channel = None
        self.spike_unit = None
        self.spike_waveform = None
        self.spike_ticks = None
        self.spike_time_res: float | None = None
        self._tick_rate: int | None = None

    # ── File detection helpers ────────────────────────────────────────────

    @staticmethod
    def _pick_by_prefix(folder: Path, pattern: str, prefix: str) -> Path | None:
        """Largest file in `folder` matching glob `pattern` whose name starts
        with `prefix` (case-insensitive). Returns None if nothing matches.

        Matches MATLAB's pickByPrefix, which also takes the largest byte size —
        not the newest or alphabetically first.
        """
        pre = prefix.lower()
        matches = [p for p in folder.glob(pattern) if p.name.lower().startswith(pre)]
        if not matches:
            return None
        return max(matches, key=lambda p: p.stat().st_size)  # largest wins

    @classmethod
    def list_nev_files(
        cls, session_path: str | Path, date: str, *, data_type: str = "raw_data"
    ) -> list[Path]:
        """
        List the .nev files in a session's raw-data folder (largest first), with
        sizes. Use this to choose a `nev_filename` when several recordings exist.

        Returns the list of paths; also prints them when called interactively.
        """
        return cls._list_files(session_path, date, "*.nev", data_type)

    @classmethod
    def list_ns_files(
        cls,
        session_path: str | Path,
        date: str,
        *,
        ns_marker: str = C.EYE_IDENTIFIER,
        data_type: str = "raw_data",
    ) -> list[Path]:
        """List the NSx files (default .ns2) in a session's raw-data folder (see
        list_nev_files). Pass ns_marker="ns6" etc. for others."""
        marker = ns_marker.lstrip(".").lower()
        return cls._list_files(session_path, date, f"*.{marker}", data_type)

    @staticmethod
    def _list_files(
        session_path: str | Path, date: str, pattern: str, data_type: str
    ) -> list[Path]:
        folder = Path(session_path) / data_type / date
        files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)
        if not files:
            print(f"No {pattern} file found in: {folder}")
        for p in files:
            print(f"{p.stat().st_size / 1e6:8.1f} MB  {p.name}")
        return files

    # ── Batch helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_dates(
        session_path: str | Path, dates: str | list | tuple | None, data_type: str
    ) -> list[str]:
        """Normalise a `dates` argument into a list of date strings.

        A non-empty string becomes a single-element list; a non-empty list/tuple
        is used as-is (order preserved). An empty value or None triggers
        discovery: every ``YYYY-MM-DD`` folder under
        ``session_path / data_type`` (sorted).
        """
        if isinstance(dates, str):
            if dates:
                return [dates]
        elif dates:  # non-empty list/tuple
            return [str(d) for d in dates]

        # Discover date-named folders in the raw-data root.
        root = Path(session_path) / data_type
        if not root.is_dir():
            raise FileNotFoundError(f"Raw-data folder not found: {root}")
        found = sorted(
            p.name
            for p in root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
            if p.is_dir()
        )
        if not found:
            raise FileNotFoundError(f"No YYYY-MM-DD date folders found in: {root}")
        return found

    @staticmethod
    def _exports_exist(
        session_path: str | Path, date: str, output_folder: str
    ) -> bool:
        """True if both the expmeta .txt and trials .csv already exist for `date`."""
        out = Path(session_path) / output_folder / date
        return (
            (out / f"Blackrock_{date}_expmeta.txt").exists()
            and (out / f"Blackrock_{date}_trials.csv").exists()
        )

    # ── Properties ────────────────────────────────────────────────────────
    # Filenames match the MATLAB export exactly, minus the `_matlab` suffix.

    @property
    def txt_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_expmeta.txt"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_trials.csv"

    @property
    def eye_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_eye.mat"

    @property
    def lfp_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_lfp.mat"

    @property
    def photodiode_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_photodiode.mat"

    @property
    def spikes_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_spikes.mat"

    @property
    def waveforms_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_spikes_waveform.mat"

    # ── Loading ────────────────────────────────────────────────────────────

    def load_nev(self) -> "BlackRockLoader":
        """
        Read and parse the comment .nev file. Populates self.experiments,
        self.trials and self.start_ticks. Returns self for method chaining.
        """
        comments, times_s, ticks, source = self._get_comments()
        self.comment_nev_path = source

        if self.verbose:
            print(f"Loading NEV: {source.name}  ({len(comments)} events)")

        self.experiments, self.trials, self.trial_ticks = parse_comments(
            comments, times_s, ticks
        )
        self.start_ticks = self.trial_ticks["Start"]
        self.end_ticks = self.trial_ticks["End"]
        self.comment_time_res = self._tick_rate
        compute_derived_features(self.trials)

        if self.verbose:
            n_complete = sum(t["Save_complete"] for t in self.trials)
            print(
                f"  {len(self.trials)} trials parsed ({n_complete} complete), "
                f"{len(self.experiments)} session(s)"
            )

        return self

    @staticmethod
    def _read_nsx_packets(ns_path: Path, nsx):
        """Read a one-sample-per-packet PTP NSx file as ONE continuous block.

        Returns ``(data (nChan, nSamp) int16, timestamps (nSamp,) uint64)``, or
        None when the file is not one-sample-per-packet (older formats), in
        which case the caller falls back to brpylib's getdata().

        brpylib's getdata() is not usable here for two reasons, both of which
        would silently corrupt the comparison against MATLAB:

        * It starts a new segment at every ``diff(timestamp) > 2 * clk``. In
          these recordings that fires on isolated SINGLE dropped samples (gaps
          of 2.000008 x the sample period), splitting one continuous session
          into dozens of segments — 45 for a 5.5 h LFP file. NPMK's pause
          detection is frame-based (100000 packets, ``maxTickMultiple = 2``), so
          it does not split on those; it sees one segment and repairs the
          accumulated drift in the samplealign step instead.
        * For the final segment it uses ``len(struct_arr) - 1``, so the last
          packet of the file is always dropped.

        Reading the packets directly gives the same view NPMK has.
        """
        import os

        import numpy as np

        eoh = nsx.basic_header["BytesInHeader"]
        n_chan = nsx.basic_header["ChannelCount"]
        dt = np.dtype(
            [
                ("reserved", "uint8"),
                ("timestamps", "uint64"),
                ("num_data_points", "uint32"),
                ("samples", "int16", n_chan),
            ]
        )
        n_pkt = int((os.path.getsize(ns_path) - eoh) // dt.itemsize)
        if n_pkt <= 0:
            return None
        arr = np.memmap(ns_path, dtype=dt, shape=n_pkt, offset=eoh, mode="r")
        # Same check brpylib and NPMK make: every packet carries exactly 1 sample.
        probe = min(n_pkt, 10)
        if not np.all(arr["num_data_points"][:probe] == 1):
            return None

        timestamps = np.asarray(arr["timestamps"])
        # Materialise C-contiguous (nChan, nSamp); the packet layout interleaves
        # channels, so leaving it as a strided memmap view would make the
        # per-trial slicing in segment_continuous crawl.
        data = np.ascontiguousarray(np.asarray(arr["samples"]).T)
        return data, timestamps

    def load_continuous(self, ns_path: str | Path, role: str = "eye") -> dict:
        """
        Read ONE continuous (.nsx) stream — eye, LFP or photodiode.

        Returns a dict with ``data`` (raw int16, channels × samples),
        ``uv_per_digit`` (per-channel µV per digit), ``samplingrate``,
        ``abs_time`` (absolute seconds per sample) and ``timeresolution``, and
        stores it under ``self.raw[role]``.

        The samples stay int16 as stored on disk: converting to µV here would
        force the whole array to float64, and these files run to hundreds of MB.
        ``segment_continuous`` applies ``uv_per_digit`` per trial slice, so only
        the data actually kept is converted.
        """
        import numpy as np

        _patch_numpy_for_brpylib()
        try:
            from brpylib import NsxFile
        except ImportError as e:
            raise ImportError(
                "brpylib is required. Install it with:\n"
                "  pip install -e /path/to/Python-Utilities-main/"
            ) from e

        ns_path = Path(ns_path)
        if self.verbose:
            print(f"Loading {role}: {ns_path.name}")

        nsx = NsxFile(str(ns_path))
        try:
            tick_rate = nsx.basic_header["TimeStampResolution"]
            # Same per-channel factor openNSx applies for 'uv' (MaxAnalogValue
            # over MaxDigitalValue). Look it up by electrode id rather than by
            # header position, so it stays correct if the returned channel set
            # is ever a subset. NPMK calls this field MaxDigiValue.
            factor = {
                h["ElectrodeID"]: float(h["MaxAnalogValue"]) / float(h["MaxDigitalValue"])
                for h in nsx.extended_headers
                if h.get("MaxDigitalValue")
            }
            # The header's Units string is what that factor actually produces.
            header_unit = {
                h["ElectrodeID"]: str(h.get("Units", "")).strip()
                for h in nsx.extended_headers
            }
            packets = self._read_nsx_packets(ns_path, nsx)
            if packets is not None:
                data, stamps = packets
                elec_ids = [h["ElectrodeID"] for h in nsx.extended_headers]
                samplingrate = (
                    nsx.basic_header["SampleResolution"] / nsx.basic_header["Period"]
                )
            else:
                # Older, multi-sample-per-packet file: brpylib's segmentation is
                # meaningful here (one packet == one uninterrupted segment).
                out = nsx.getdata("all", 0, "all", 1, full_timestamps=True)
                if len(out["data"]) > 1:
                    print(
                        f"  WARNING: {len(out['data'])} data segments; "
                        "concatenating (MATLAB's loader assumes one segment)."
                    )
                data = np.concatenate(out["data"], axis=1)
                stamps = np.atleast_1d(out["data_headers"][0]["Timestamp"])
                elec_ids = out["elec_ids"]
                samplingrate = out["samp_per_s"]
        finally:
            nsx.close()

        start_tick = int(stamps[0])

        # Correct PTP clock drift exactly as NPMK's openNSx does, so both
        # readers hand the segmenter the same sample stream.
        data, n_added = align_ptp_clock_drift(data, stamps, samplingrate, tick_rate)
        if n_added and self.verbose:
            verb = "Added" if n_added > 0 else "Removed"
            print(
                f"  {verb} {abs(n_added)} sample(s) for clock drift alignment"
            )

        n_samples = data.shape[1]
        # Reconstruct absolute sample time on the recording clock (same basis as
        # the comment/spike timestamps): start_tick / tick_rate + n / samp_per_s.
        abs_time = start_tick / tick_rate + np.arange(n_samples) / samplingrate
        uv_per_digit = np.array(
            [factor.get(int(e), 1.0) for e in elec_ids], dtype=float
        )
        raw_units = [header_unit.get(int(e), "") for e in elec_ids]

        if self.to_microvolts:
            # Opt in: fold the header unit into the scale factor so the samples
            # come out in true microVolts regardless of stream.
            mult = np.array(
                [C.UNIT_TO_MICROVOLTS.get(u, 1.0) for u in raw_units], dtype=float
            )
            unknown = {u for u, m in zip(raw_units, mult) if u not in C.UNIT_TO_MICROVOLTS}
            if unknown:
                print(
                    f"  WARNING: unrecognised header unit(s) {sorted(unknown)}; "
                    "left unconverted."
                )
            uv_per_digit = uv_per_digit * mult
            unit: str | list[str] = C.MICROVOLTS_LABEL
        else:
            labels = [C.UNIT_LABELS.get(u, u or "unknown") for u in raw_units]
            # A single string when the whole stream shares a unit (always so in
            # practice); a per-channel list otherwise, sliced by subset_channels.
            unit = labels[0] if len(set(labels)) == 1 else labels

        stream = {
            "data": data,
            "uv_per_digit": uv_per_digit,
            "unit": unit,
            "start_tick": start_tick,
            "samplingrate": samplingrate,
            "abs_time": abs_time,
            "timeresolution": tick_rate,
            "elec_ids": elec_ids,
            "source": ns_path.name,
        }
        self.raw[role] = stream

        if self.verbose:
            print(
                f"  {len(elec_ids)} channels, {n_samples} samples, "
                f"{samplingrate} Hz, unit "
                f"{unit if isinstance(unit, str) else 'mixed'}"
            )
        return stream

    def load_eye(self) -> "BlackRockLoader":
        """Load the eye NSx stream (and, by default, the photodiode with it)."""
        if self.eye_path is None:
            raise RuntimeError("No eye file resolved; construct with load_eye=True.")
        self.load_continuous(self.eye_path, "eye")
        return self

    def load_lfp(self) -> "BlackRockLoader":
        """Load the LFP NSx stream."""
        if self.lfp_path is None:
            raise RuntimeError("No LFP file resolved; construct with load_lfp=True.")
        self.load_continuous(self.lfp_path, "lfp")
        return self

    def load_photodiode(self) -> "BlackRockLoader":
        """
        Resolve where the photodiode lives and load it if it needs its own file.

        By default it rides on rows `photodiode_channels` of the eye ns2, which
        is already loaded — nothing is read here, we only record that. Falls back
        to (or, when photodiode_use_separate_file is set, goes straight to) a
        dedicated NSP-*.ns4. Mirrors the photodiode branch of MATLAB loadSession.
        """
        eye = self.raw.get("eye")
        if (
            not self.photodiode_use_separate_file
            and eye is not None
            and eye["data"].shape[0] >= max(self.photodiode_channels)
        ):
            self.photodiode_from_eye = True
            if self.verbose:
                print(
                    f"  Photodiode: channels {list(self.photodiode_channels)} "
                    "of the eye ns2"
                )
            return self

        path = self._pick_by_prefix(
            self.data_folder, f"*.{C.PHOTODIODE_IDENTIFIER}", C.PHOTODIODE_PREFIX
        )
        if path is None:
            raise FileNotFoundError(
                f"No {C.PHOTODIODE_PREFIX}-*.{C.PHOTODIODE_IDENTIFIER} photodiode "
                f"file in {self.data_folder}, and the eye stream does not have "
                f"channel {max(self.photodiode_channels)}."
            )
        self.photodiode_path = path
        self.photodiode_from_eye = False
        self.load_continuous(path, "photodiode")
        return self

    def load_online_spikes(self, nev_path: str | Path | None = None) -> "BlackRockLoader":
        """
        Load online spike timing (HUB-*.nev). Populates self.spike_times
        (seconds), self.spike_channel, self.spike_unit and self.spike_ticks
        (raw uint64), ready for segment_spikes / export_online_spikes. If the
        loader was built with load_online_wave=True, also populates
        self.spike_waveform (microVolts, shape (nSpikes, nSamp)).
        """
        nev_path = Path(nev_path) if nev_path is not None else self.spike_nev_path
        if nev_path is None:
            raise RuntimeError(
                "No spike file resolved; construct with load_online_spikes=True "
                "(or load_online_wave=True)."
            )
        if self.verbose:
            print(f"Loading spikes: {nev_path.name}")

        (
            times_s,
            channel,
            unit,
            waveform,
            ticks,
            time_res,
            n_dropped,
        ) = self._read_online_spikes(
            nev_path,
            read_waveforms=self.load_online_wave,
            include_unsorted=self.include_unsorted,
        )
        self.spike_times = times_s
        self.spike_channel = channel
        self.spike_unit = unit
        self.spike_waveform = waveform
        self.spike_ticks = ticks
        self.spike_time_res = time_res

        if self.verbose:
            msg = f"  {len(times_s)} spikes loaded"
            if n_dropped:
                msg += f" (dropped {n_dropped} unsorted/noise)"
            if waveform is not None:
                msg += f" (waveforms {waveform.shape[1]} samples/spike)"
            print(msg)

        return self

    # ── Segmentation ───────────────────────────────────────────────────────

    def segment_eye_stream(self) -> None:
        """
        Segment the eye ns2 ONCE and split the result by channel row into
        self.eye (eye_channels) and, when the photodiode rides in the same file,
        self.photodiode (photodiode_channels).

        Both products come out of a single segment_continuous pass because they
        are the same samples on the same clock — cutting the stream twice was
        pure duplicated work on a 500 MB file. Memoised on the products already
        existing, so parse_eye and parse_photodiode can be called in either
        order, or individually, and the pass still runs exactly once.
        Port of BlackrockLoader.segmentEyeStream.
        """
        stream = self.raw.get("eye")
        if stream is None:
            self.eye = None
            return
        wants_pd = self._want_photodiode and self.photodiode_from_eye
        if self.eye is not None and (not wants_pd or self.photodiode is not None):
            return  # already segmented

        full = segment_continuous(
            self.trials,
            stream["data"],
            stream["abs_time"],
            stream["samplingrate"],
            self.pre_ms,
            self.post_ms,
            stream["uv_per_digit"],
            stream["unit"],
            start_ticks=self.start_ticks,
            end_ticks=self.end_ticks,
            ref_tick=stream["start_tick"],
            time_res=_shared_time_res(self.comment_time_res, stream["timeresolution"]),
        )

        n_chan = full["data"].shape[0]
        self.eye = subset_channels(
            full, clamp_channels(self.eye_channels, n_chan, "eye_channels")
        )
        if wants_pd:
            self.photodiode = subset_channels(
                full,
                clamp_channels(self.photodiode_channels, n_chan, "photodiode_channels"),
            )
        # The raw stream is the largest thing the loader holds; release it now
        # that both per-trial products exist.
        self.raw.pop("eye", None)

    def parse_eye(self) -> dict | None:
        """Segment the loaded eye stream into per-trial slices (self.eye)."""
        self.segment_eye_stream()
        return self.eye

    def parse_lfp(self) -> dict | None:
        """Segment the loaded LFP stream into per-trial slices (self.lfp)."""
        stream = self.raw.get("lfp")
        if stream is None:
            self.lfp = None
            return None
        self.lfp = segment_continuous(
            self.trials,
            stream["data"],
            stream["abs_time"],
            stream["samplingrate"],
            self.pre_ms,
            self.post_ms,
            stream["uv_per_digit"],
            stream["unit"],
            start_ticks=self.start_ticks,
            end_ticks=self.end_ticks,
            ref_tick=stream["start_tick"],
            time_res=_shared_time_res(self.comment_time_res, stream["timeresolution"]),
        )
        self.raw.pop("lfp", None)
        return self.lfp

    def parse_photodiode(self) -> dict | None:
        """
        Segment the photodiode into per-trial slices (self.photodiode).

        In the default layout this is done by the shared eye-stream pass, so
        calling it after parse_eye is a no-op. Only a dedicated photodiode file
        is segmented separately here.
        """
        if not self._want_photodiode:
            self.photodiode = None
            return None
        if self.photodiode_from_eye:
            self.segment_eye_stream()
            return self.photodiode
        stream = self.raw.get("photodiode")
        if stream is None:
            self.photodiode = None
            return None
        self.photodiode = segment_continuous(
            self.trials,
            stream["data"],
            stream["abs_time"],
            stream["samplingrate"],
            self.pre_ms,
            self.post_ms,
            stream["uv_per_digit"],
            stream["unit"],
            start_ticks=self.start_ticks,
            end_ticks=self.end_ticks,
            ref_tick=stream["start_tick"],
            time_res=_shared_time_res(self.comment_time_res, stream["timeresolution"]),
        )
        self.raw.pop("photodiode", None)
        return self.photodiode

    # ── Export ─────────────────────────────────────────────────────────────

    def export(self) -> tuple[Path, Path]:
        """
        Write expmeta .txt and trials .csv to output_dir.
        Must be called after load_nev().

        Returns
        -------
        (txt_path, csv_path)
        """
        if self.experiments is None or self.trials is None:
            raise RuntimeError("Call load_nev() before export().")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        write_expmeta(self.experiments, self.txt_path)
        write_trials_csv(self.trials, self.csv_path)

        return self.txt_path, self.csv_path

    def _export_continuous(self, product: dict, var: str, path: Path, label: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_mat_v73(path, {var: product})
        if self.verbose:
            print(
                f"  {label} segmented ({product['data'].shape[1]} trials) "
                f"-> {path.name}"
            )
        return path

    def export_eye(self) -> Path:
        """Segment (if needed) and save the eye product as `eye` in _eye.mat."""
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_eye().")
        self.parse_eye()
        if self.eye is None:
            raise RuntimeError("Call load_eye() before export_eye().")
        return self._export_continuous(self.eye, "eye", self.eye_mat_path, "Eye")

    def export_lfp(self) -> Path:
        """Segment (if needed) and save the LFP product as `lfp` in _lfp.mat."""
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_lfp().")
        if self.lfp is None:
            self.parse_lfp()
        if self.lfp is None:
            raise RuntimeError("Call load_lfp() before export_lfp().")
        return self._export_continuous(self.lfp, "lfp", self.lfp_mat_path, "LFP")

    def export_photodiode(self) -> Path:
        """Segment (if needed) and save `photodiode` in _photodiode.mat."""
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_photodiode().")
        if self.photodiode is None:
            self.parse_photodiode()
        if self.photodiode is None:
            raise RuntimeError("No photodiode product; construct with load_photodiode=True.")
        return self._export_continuous(
            self.photodiode, "photodiode", self.photodiode_mat_path, "Photodiode"
        )

    def export_online_spikes(self) -> Path | None:
        """
        Rasterize online spikes into per-trial bins and save as a .mat file.
        Must be called after load_nev() and load_online_spikes().

        The file is written even when every spike was dropped as unsorted/noise:
        MATLAB's parseSpikes always builds the struct and export always saves it,
        yielding a zero-unit product whose per-trial fields are still present.
        Matching that keeps downstream readers from special-casing the file's
        absence.

        When waveforms were loaded they are passed through so the product also
        carries info.MeanWaveform, matching MATLAB.
        """
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_online_spikes().")
        if self.spike_times is None:
            raise RuntimeError("Call load_online_spikes() before export_online_spikes().")
        if len(self.spike_times) == 0 and self.verbose:
            print("  No sorted spikes; writing an empty spike product (as MATLAB does).")

        online_spike = segment_spikes(
            self.trials,
            self.spike_times,
            self.spike_channel,
            self.spike_unit,
            self.pre_ms,
            self.post_ms,
            self.bin_ms,
            self.isi_violation_ms,
            self.spike_waveform,
            spike_ticks=self.spike_ticks,
            start_ticks=self.start_ticks,
            end_ticks=self.end_ticks,
            time_res=_shared_time_res(self.comment_time_res, self.spike_time_res),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_mat_v73(self.spikes_mat_path, {"online_spike": online_spike})
        if self.verbose:
            d = online_spike["data"]
            print(
                f"  Spikes rasterized ({d.shape[0]} units x {d.shape[1]} trials) "
                f"-> {self.spikes_mat_path.name}"
            )
        return self.spikes_mat_path

    def export_online_wave(self) -> Path | None:
        """
        Segment per-spike waveforms into per-trial slices and save as a .mat
        file. Must be called after load_nev() and load_online_spikes() with the
        loader built using load_online_wave=True.

        Returns
        -------
        waveforms_mat_path or None
            None (and nothing written) when there are no spikes to export — e.g.
            all were unsorted/noise and dropped (include_unsorted=False).
        """
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_online_wave().")
        if self.spike_waveform is None:
            raise RuntimeError(
                "No waveforms loaded; construct with load_online_wave=True and "
                "call load_online_spikes() before export_online_wave()."
            )
        if len(self.spike_waveform) == 0:
            if self.verbose:
                print("  No sorted spikes found; skipping waveform export.")
            return None

        online_spike_waveform = segment_waveforms(
            self.trials,
            self.spike_times,
            self.spike_channel,
            self.spike_unit,
            self.spike_waveform,
            self.pre_ms,
            self.post_ms,
            self.spike_ticks,
            self.start_ticks,
            self.spike_time_res,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_mat_v73(
            self.waveforms_mat_path,
            {"online_spike_waveform": online_spike_waveform},
        )
        if self.verbose:
            w = online_spike_waveform["waveform"]
            print(
                f"  Waveforms segmented ({w.shape[0]} units x {w.shape[1]} trials "
                f"x {w.shape[2]} spikes x {w.shape[3]} samples) "
                f"-> {self.waveforms_mat_path.name}"
            )
        return self.waveforms_mat_path

    # ── Orchestration ──────────────────────────────────────────────────────

    def run(self) -> tuple[Path, list[str]]:
        """
        Convenience: load_nev() + export(), then every stream whose file was
        resolved at construction.

        Each optional product fails soft: if its file turns out to be unusable
        the product is skipped with a message and the rest still export, like
        the MATLAB loader. A missing comment file is still a hard error.

        Returns
        -------
        (output_dir, filenames)
        """
        self.load_nev().export()
        filenames = [self.txt_path.name, self.csv_path.name]

        # Eye first, and resolve where the photodiode lives BEFORE segmenting:
        # in the default layout both products come out of one segmentation pass
        # over the eye stream, and that pass releases the raw stream when it is
        # done. Resolving afterwards would find the raw data already gone.
        if self.eye_path is not None:
            try:
                self.load_eye()
            except Exception as e:
                print(f"  Eye loading failed: {e}")

        if self._want_photodiode:
            try:
                self.load_photodiode()
            except Exception as e:
                print(f"  Photodiode loading failed: {e}")
                self._want_photodiode = False

        if self.raw.get("eye") is not None:
            try:
                self.export_eye()
                filenames.append(self.eye_mat_path.name)
            except Exception as e:
                print(f"  Eye export failed: {e}")

        if self._want_photodiode:
            try:
                self.export_photodiode()
                filenames.append(self.photodiode_mat_path.name)
            except Exception as e:
                print(f"  Photodiode export failed: {e}")

        if self.lfp_path is not None:
            try:
                self.load_lfp()
                self.export_lfp()
                filenames.append(self.lfp_mat_path.name)
            except Exception as e:
                print(f"  LFP loading failed: {e}")

        if self.spike_nev_path is not None:
            try:
                self.load_online_spikes()
                if self._export_online_spikes:
                    # Written even with zero sorted spikes, as MATLAB does.
                    self.export_online_spikes()
                    filenames.append(self.spikes_mat_path.name)
                if self.load_online_wave:
                    # MATLAB skips the waveform product when the filtered
                    # waveform matrix is empty, so we do too.
                    if self.export_online_wave() is not None:
                        filenames.append(self.waveforms_mat_path.name)
            except Exception as e:
                print(f"  Spike loading failed: {e}")

        return self.output_dir, filenames

    @classmethod
    def run_batch(
        cls,
        session_path: str | Path,
        dates: str | list[str] | None = None,
        *,
        data_type: str = "raw_data",
        load_eye: bool = False,
        eye_marker: str = C.EYE_IDENTIFIER,
        load_lfp: bool = False,
        load_photodiode: bool = False,
        load_online_spikes: bool = False,
        load_online_wave: bool = False,
        include_unsorted: bool = False,
        pre_ms: float = C.SEGMENT_PRE_MS,
        post_ms: float = C.SEGMENT_POST_MS,
        bin_ms: float = C.SEGMENT_BIN_MS,
        output_folder: str = "export_data",
        skip_existing: bool = False,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Load and export several date folders in one call.

        Builds one single-date BlackRockLoader per date and runs it, reusing the
        same parse/export pipeline. A date that fails (e.g. missing comment .nev,
        parse error) is recorded and skipped so the batch always finishes.

        Parameters
        ----------
        session_path : str or Path
            Directory containing the `data_type` and `output_folder` sub-folders.
        dates : str or list[str] or None
            A year-month-date string, a list of them, or None/empty to discover
            and process every ``YYYY-MM-DD`` folder under
            ``session_path / data_type`` (sorted).
        skip_existing : bool
            Skip dates whose expmeta .txt and trials .csv already exist, so a
            re-run only processes new folders. Default False.
        Everything else
            Forwarded to each per-date loader (see BlackRockLoader).

        Returns
        -------
        list[dict] — one per date, each with keys:
            "date", "status" ("ok" | "skipped" | "failed"),
            and ("output_dir", "files") on success or "error" on failure.
        """
        session_path = Path(session_path)
        date_list = cls._resolve_dates(session_path, dates, data_type)

        results: list[dict] = []
        n = len(date_list)
        if verbose:
            print(f"Batch: {n} date folder(s)")

        for i, d in enumerate(date_list, 1):
            prefix = f"[{i}/{n}] {d}"

            if skip_existing and cls._exports_exist(session_path, d, output_folder):
                print(f"{prefix}  ... SKIPPED (already exported)")
                results.append({"date": d, "status": "skipped"})
                continue

            try:
                loader = cls(
                    session_path,
                    d,
                    data_type=data_type,
                    load_eye=load_eye,
                    eye_marker=eye_marker,
                    load_lfp=load_lfp,
                    load_photodiode=load_photodiode,
                    load_online_spikes=load_online_spikes,
                    load_online_wave=load_online_wave,
                    include_unsorted=include_unsorted,
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    bin_ms=bin_ms,
                    output_folder=output_folder,
                    verbose=verbose,
                )
                output_dir, files = loader.run()
                print(f"{prefix}  ... OK")
                results.append(
                    {"date": d, "status": "ok", "output_dir": output_dir, "files": files}
                )
            except Exception as e:  # continue-on-error
                print(f"{prefix}  ... FAILED ({e})")
                results.append({"date": d, "status": "failed", "error": str(e)})

        n_ok = sum(r["status"] == "ok" for r in results)
        n_skip = sum(r["status"] == "skipped" for r in results)
        n_fail = sum(r["status"] == "failed" for r in results)
        print(f"\nDone: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
        return results

    # ── Reporting ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """
        Print a behavioral summary of the loaded session.
        Must be called after load_nev().
        """
        if self.trials is None:
            raise RuntimeError("Call load_nev() before print_summary().")

        import math

        def _isnan(v):
            try:
                return math.isnan(v)
            except (TypeError, ValueError):
                return False

        # ── Session length ─────────────────────────────────────────────────
        # A recording can hold several sessions; report the count and the overall
        # span (earliest start → latest end across all sessions).
        sessions = self.experiments or []
        starts = [e.get("start") for e in sessions if not _isnan(e.get("start"))]
        ends = [e.get("end") for e in sessions if not _isnan(e.get("end"))]

        print("Session Summary")
        print("─" * 50)
        print(f"Sessions       : {len(sessions)}")
        if starts and ends:
            duration = max(ends) - min(starts)
            mins, secs = int(duration // 60), duration % 60
            print(f"Total length   : {mins} min {secs:.1f} sec")
        print(f"Total trials   : {len(self.trials)}"
              f"  ({sum(t['Save_complete'] == 1 for t in self.trials)} complete)")

        # ── Per-task breakdown ─────────────────────────────────────────────
        tasks: dict = {}
        for t in self.trials:
            task = t.get("Task") or "unknown"
            tasks.setdefault(task, []).append(t)

        for task, trials in tasks.items():
            outcomes = [t.get("Trialoutcome") or "" for t in trials]

            n_wrong   = sum("wrong"   in o.lower() for o in outcomes)
            n_correct = sum("correct" in o.lower() for o in outcomes)

            is_choice = n_wrong > 0
            n_success = (n_correct + n_wrong) if is_choice else n_correct

            print(f"\nTask: {task}{' [choice]' if is_choice else ''}")
            print(f"  Total          : {len(trials)}")
            if is_choice:
                print(f"  Successful     : {n_success}  (correct + error)")
            else:
                print(f"  Correct        : {n_correct}")
            print("  Outcomes:")

            # Count every unique outcome value
            outcome_counts: dict = {}
            for o in outcomes:
                label = o if o else "(none)"
                outcome_counts[label] = outcome_counts.get(label, 0) + 1
            for label, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
                print(f"    {label:<30} : {count}")

    def print_comments(self) -> None:
        """
        Print all raw comment events with their timestamps.
        Useful for inspecting raw NEV events before parsing.
        """
        comments, times_s, _, _ = self._get_comments()
        print(f"{'Time (s)':>20}  Comment")
        print("-" * 60)
        for t, c in zip(times_s, comments):
            print(f"{t:>20.6f}  {c}")

    # ── Internal NEV readers ────────────────────────────────────────────────

    def _get_comments(self) -> tuple[list[str], list[float], list[int], Path]:
        """Return (comments, times_s, ticks, source_path) from the first comment
        candidate that actually carries comments (NSP, then HUB legacy)."""
        if not self._comment_candidates:
            raise FileNotFoundError(
                f"No {C.COMMENT_PREFIX_PRIMARY}-*.nev or {C.COMMENT_PREFIX_LEGACY}-*.nev "
                f"comment file found in: {self.data_folder}"
            )
        last_err: Exception | None = None
        for path in self._comment_candidates:
            try:
                comments, times_s, ticks = self._read_comments(path)
                return comments, times_s, ticks, path
            except ValueError as e:
                last_err = e  # no comments in this file; try the next candidate
        raise ValueError(
            f"No comment events found in any of "
            f"{[p.name for p in self._comment_candidates]}: {last_err}"
        )

    def _read_comments(self, nev_path: Path) -> tuple[list[str], list[float], list[int]]:
        """Open the NEV file via brpylib and extract comment strings, times and
        raw ticks. Raises ValueError if the file carries no comment events."""
        _patch_numpy_for_brpylib()
        try:
            from brpylib import NevFile
        except ImportError as e:
            raise ImportError(
                "brpylib is required. Install it with:\n"
                "  uv pip install -e /path/to/Python-Utilities/"
            ) from e

        nev = NevFile(str(nev_path))
        data = nev.getdata(elec_ids="all", wave_read="no_read")

        self._tick_rate = nev.basic_header["TimeStampResolution"]

        comments_data = data.get("comments", {})
        raw_comments: list = comments_data.get("Data", [])

        if not raw_comments:
            raise ValueError(
                f"No comment events found in {nev_path}. "
                "Check that the file contains behavioral event markers."
            )

        # Use raw TimeStamps (no lag correction) so behavioral events, spikes, and
        # continuous channels all share the same hardware clock reference.
        raw_ticks: list = comments_data.get("TimeStamps", [])

        # Filespec 3.0 stores each comment twice (timestamp packet + color packet),
        # both with identical timestamp and text. NPMK filters by Flag==1; brpylib
        # returns both. Deduplicate by (tick, text), preserving order.
        seen: set = set()
        deduped_comments: list = []
        deduped_ticks: list = []
        for t, c in zip(raw_ticks, raw_comments):
            key = (t, c.strip())
            if key not in seen:
                seen.add(key)
                deduped_comments.append(c)
                deduped_ticks.append(int(t))

        times_s = [t / self._tick_rate for t in deduped_ticks]
        return deduped_comments, times_s, deduped_ticks

    def _read_online_spikes(
        self,
        nev_path: Path,
        read_waveforms: bool = False,
        include_unsorted: bool = False,
    ):
        """Online spike reader: open the HUB NEV via brpylib and extract spike
        timing into the common per-spike representation.

        Returns (times_s, channel, unit, waveform, ticks, time_res, n_dropped) —
        the same shape a future ``_read_offline_spikes`` would return, so both
        feed the shared ``segment_spikes`` / ``segment_waveforms``. ``waveform``
        is None unless ``read_waveforms`` is True, in which case it is an
        (nSpikes, nSamp) float32 array of microVolts (raw int16 scaled per
        electrode, like the MATLAB version). ``ticks`` are the raw uint64
        timestamps, kept so segment_waveforms can subtract in exact integers.
        When ``include_unsorted`` is False, spikes whose unit is in
        C.UNSORTED_UNIT_IDS (0 unsorted, 255 noise) are dropped via the shared
        ``drop_units`` helper before scaling (the memory saver); ``n_dropped``
        counts how many were removed. Raises ValueError if no spikes."""
        import numpy as np

        _patch_numpy_for_brpylib()
        try:
            from brpylib import NevFile
        except ImportError as e:
            raise ImportError(
                "brpylib is required. Install it with:\n"
                "  uv pip install -e /path/to/Python-Utilities/"
            ) from e

        nev = NevFile(str(nev_path))
        data = nev.getdata(
            elec_ids="all", wave_read="read" if read_waveforms else "no_read"
        )
        tick_rate = float(nev.basic_header["TimeStampResolution"])

        spikes = data.get("spike_events", {})
        raw_ticks = spikes.get("TimeStamps", [])
        if len(raw_ticks) == 0:
            raise ValueError(f"No spike timestamps in {nev_path}.")

        # TimeStamps are raw ticks; keep them as uint64 for the exact-integer
        # relative-time path, and divide for the seconds view that everything
        # else works in. (Channel = electrode id.)
        ticks = np.asarray(raw_ticks, dtype=np.uint64)
        times_s = ticks.astype(float) / tick_rate
        channel = np.asarray(spikes.get("Channel", []), dtype=float)
        unit = np.asarray(spikes.get("Unit", []), dtype=float)

        # raw int16 waveforms (kept int until after masking to save memory)
        raw = None
        if read_waveforms:
            raw = spikes.get("Waveforms")
            if raw is None:
                raise ValueError(
                    "Spike NEV has no waveform data (wave_read returned none); "
                    "this file may have been recorded without online waveforms."
                )
            raw = np.asarray(raw)  # (nSpikes, nSamp), int16

        # Drop unsorted (unit 0) / noise (unit 255) before the float scaling so
        # the dense waveform array only ever holds sorted units. drop_units is
        # the shared, source-agnostic filter (a future offline reader reuses it).
        n_dropped = 0
        if not include_unsorted:
            times_s, channel, unit, raw, ticks, n_dropped = drop_units(
                times_s, channel, unit, raw, ticks
            )

        waveform = None
        if read_waveforms:
            waveform = self._scale_waveforms(nev, raw, channel)
        return times_s, channel, unit, waveform, ticks, tick_rate, n_dropped

    @staticmethod
    def _scale_waveforms(nev, raw, channel):
        """Convert raw int16 spike waveforms to microVolts.

        ``raw`` is the (nSpikes, nSamp) int16 array from
        ``spike_events["Waveforms"]`` (already masked by the caller). Each
        electrode's NEUEVWAV extended header carries a DigitizationFactor
        (nV/bit); microVolts = raw * DigitizationFactor / 1000, matching the
        MATLAB BlackrockLoader (DigitalFactor / 1000). The factor is computed in
        float64 and the product taken in float32, as MATLAB does."""
        import numpy as np

        # electrode id -> DigitizationFactor (nV/bit) from NEUEVWAV headers
        factor_map = {
            h["ElectrodeID"]: h["DigitizationFactor"]
            for h in nev.extended_headers
            if "ElectrodeID" in h and "DigitizationFactor" in h
        }
        channel = np.asarray(channel, dtype=float)
        factors = np.array(
            [factor_map.get(int(c), 0.0) / 1000.0 for c in channel], dtype=float
        )
        # microVolts = raw int16 * (nV/bit) / 1000
        return np.asarray(raw, dtype=np.float32) * factors[:, None].astype(np.float32)
