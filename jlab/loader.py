"""
Finally, combine everything
BlackRockLoader — main user-facing class.

Reads the role-split Blackrock files for one session and exports the same
products as the MATLAB BlackrockLoader / BackRockFileLoader.m:
  NSP-*.nev  -> experiment comments + comment timing  (-> expmeta .txt, trials .csv)
  NSP-*.ns2  -> analog/eye data                        (-> segmented analog .mat)
  HUB-*.nev  -> online spike timing                    (-> rasterized spikes .mat)
  HUB-*.nev  -> online spike waveforms (opt-in)        (-> segmented waveforms .mat)
HUB-*.nev is also the legacy fallback for comments (early sessions wrote comments
and spikes both to the HUB file).

Spike pipeline (source-agnostic):
  A source-specific *reader* produces the common per-spike arrays
  ``(times, channel, unit[, waveform])`` -> the shared, source-agnostic
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
from ._analog import segment_analog
from ._exporter import write_expmeta, write_trials_csv
from ._features import compute_derived_features
from ._parser import parse_comments
from ._spikes import drop_units, segment_spikes
from ._waveforms import segment_waveforms


class BlackRockLoader:
    """
    Load a Blackrock session and export experiment metadata + trial data, plus
    optional trial-segmented analog (.ns2) and online-spike (HUB .nev) products.

    Construction is folder-based, mirroring BackRockFileLoader.m: no filenames
    needed. `session_path` is the directory that holds the `raw_data` and
    `export_data` sub-folders (assemble it in the notebook, e.g.
    ``Path(Basic_Path) / "Monkey <name>" / Location``). Files are picked from
    ``session_path / data_type / date`` by role prefix (the largest match wins):
    comments from ``NSP-*.nev`` (legacy fallback ``HUB-*.nev``), analog from
    ``NSP-*.ns2``, online spikes from ``HUB-*.nev``. Output goes to
    ``session_path / output_folder / date``.

    Analog and spikes are gated by `load_analog` / `load_online_spikes`; if requested
    but the prefixed file is absent they are skipped with a message (soft
    failure, like the MATLAB loader). Missing comments is a hard error.

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
    load_analog : bool
        If True, load and segment the NSP-*.ns2 analog file.
    ns_marker : str
        NSx file extension without the dot. Default "ns2".
    ns_filename : str, optional
        Explicit analog filename instead of prefix auto-pick.
    load_online_spikes : bool
        If True, load and rasterize online spikes from the HUB-*.nev file.
    spike_nev_filename : str, optional
        Explicit spike .nev filename instead of prefix auto-pick.
    load_online_wave : bool
        If True, also read per-spike waveforms from the HUB-*.nev file and
        segment them into a dense per-trial array (microVolts). Off by default
        because the dense array can be large; implies reading spikes (waveforms
        come from the same file).
    include_unsorted : bool
        If True, keep every online-spike "unit", including unsorted threshold
        crossings (unit 0) and noise (unit 255). Default False -> only sorted
        units (1-5) are kept, which applies to BOTH the spike raster and the
        waveform export and keeps the dense waveform array small.
    pre_ms, post_ms : float
        Trial-segmentation buffers (ms): window = [Start - pre, End + post].
        Default 500 each.
    bin_ms : float
        Spike-raster bin width (ms). Default 1.
    output_folder : str
        Export sub-folder name. Default "export_data".
    verbose : bool
        Print progress messages. Default True.

    Examples
    --------
    >>> from jlab import BlackRockLoader
    >>> from pathlib import Path
    >>> session = Path(Basic_Path) / "Monkey Porthos" / "in_lab"

    >>> # comments only
    >>> loader = BlackRockLoader(session, "2026-06-17")
    >>> output_dir, files = loader.run()

    >>> # comments + analog + online spikes
    >>> loader = BlackRockLoader(session, "2026-06-17",
    ...                          load_analog=True, load_online_spikes=True)
    >>> output_dir, files = loader.run()

    >>> # several dates at once (or run_batch(session) to do every date folder)
    >>> report = BlackRockLoader.run_batch(session, dates=["2026-06-17", "2026-06-18"],
    ...                                    load_analog=True, load_online_spikes=True)
    """

    def __init__(
        self,
        session_path: str | Path,
        date: str,
        *,
        data_type: str = "raw_data",
        nev_filename: str | None = None,
        load_analog: bool = False,
        ns_marker: str = "ns2",
        ns_filename: str | None = None,
        load_online_spikes: bool = False,
        spike_nev_filename: str | None = None,
        load_online_wave: bool = False,
        include_unsorted: bool = False,
        pre_ms: float = C.SEGMENT_PRE_MS,
        post_ms: float = C.SEGMENT_POST_MS,
        bin_ms: float = C.SEGMENT_BIN_MS,
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

        # ── Analog (NSP-*.ns2), gated ────────────────────────────────────────
        self.ns_marker = ns_marker.lstrip(".").lower()
        self.ns_path: Path | None = None
        if load_analog:
            if ns_filename is not None:
                path = self.data_folder / ns_filename
                if not path.exists():
                    raise FileNotFoundError(f"Analog file not found: {path}")
                self.ns_path = path
            else:
                self.ns_path = self._pick_by_prefix(
                    self.data_folder, f"*.{self.ns_marker}", C.ANALOG_PREFIX
                )
            if self.ns_path is None and verbose:
                print(
                    f"No {C.ANALOG_PREFIX}-*.{self.ns_marker} analog file found; "
                    "skipping analog load."
                )

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
            if self.ns_path is not None:
                print(f"Analog file   : {self.ns_path.name}")
            if self.spike_nev_path is not None:
                role = "spikes+waveforms" if self.load_online_wave else "spikes"
                print(f"Spike NEV     : {self.spike_nev_path.name} ({role})")
            print(f"Output dir    : {self.output_dir}")

        self.experiments: list[dict] | None = None
        self.trials: list[dict] | None = None
        self.analog: dict | None = None
        self.nsx_data = None
        self.nsx_abs_time = None
        self.nsx_samplingrate = None
        self.spike_times = None
        self.spike_channel = None
        self.spike_unit = None
        self.spike_waveform = None
        self._tick_rate: int | None = None
        self._nsx_tick_rate: int | None = None

    # ── File detection helpers ────────────────────────────────────────────

    @staticmethod
    def _pick_by_prefix(folder: Path, pattern: str, prefix: str) -> Path | None:
        """Largest file in `folder` matching glob `pattern` whose name starts
        with `prefix` (case-insensitive). Returns None if nothing matches."""
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
        ns_marker: str = "ns2",
        data_type: str = "raw_data",
    ) -> list[Path]:
        """List the NSx analog files (default .ns2) in a session's raw-data
        folder (see list_nev_files). Pass ns_marker="ns6" etc. for others."""
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

    @property
    def txt_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_expmeta.txt"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_trials.csv"

    @property
    def analog_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_analog.mat"

    @property
    def spikes_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_spikes.mat"

    @property
    def waveforms_mat_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_waveforms.mat"

    # ── Public methods ─────────────────────────────────────────────────────

    def load_nev(self) -> "BlackRockLoader":
        """
        Read and parse the comment .nev file. Populates self.experiments and
        self.trials. Returns self for method chaining.
        """
        comments, times_s, source = self._get_comments()
        self.comment_nev_path = source

        if self.verbose:
            print(f"Loading NEV: {source.name}  ({len(comments)} events)")

        self.experiments, self.trials = parse_comments(comments, times_s)
        compute_derived_features(self.trials)

        if self.verbose:
            n_complete = sum(t["Save_complete"] for t in self.trials)
            print(f"  {len(self.trials)} trials parsed ({n_complete} complete)")

        return self

    def load_analog(
        self,
        ns_path: str | Path | None = None,
        elec_ids="all",
        start_time_s: float = 0,
        data_time_s="all",
        downsample: int = 1,
    ) -> "BlackRockLoader":
        """
        Load the continuous analog NSx file (e.g. .ns2 for eye tracking) and
        reconstruct each sample's absolute time on the recording clock.

        Populates self.analog (raw brpylib dict) plus self.nsx_data
        (channels × samples), self.nsx_abs_time (seconds) and
        self.nsx_samplingrate, ready for segment_analog / export_analog.
        """
        import numpy as np

        if not hasattr(np.chararray, "tostring"):
            np.chararray.tostring = np.chararray.tobytes  # type: ignore[attr-defined]

        try:
            from brpylib import NsxFile
        except ImportError as e:
            raise ImportError(
                "brpylib is required. Install it with:\n"
                "  pip install -e /path/to/Python-Utilities-main/"
            ) from e

        ns_path = Path(ns_path) if ns_path is not None else self.ns_path
        if ns_path is None:
            raise RuntimeError("No analog file resolved; construct with load_analog=True.")
        if self.verbose:
            print(f"Loading analog: {ns_path.name}")

        nsx = NsxFile(str(ns_path))
        self.analog = nsx.getdata(elec_ids, start_time_s, data_time_s, downsample)
        self._nsx_tick_rate = nsx.basic_header["TimeStampResolution"]
        nsx.close()

        # Reconstruct absolute sample time on the recording clock (same basis as
        # the comment/spike timestamps): start_tick / tick_rate + n / samp_per_s.
        self.nsx_data = self.analog["data"][0]  # channels × samples
        self.nsx_samplingrate = self.analog["samp_per_s"]
        start_tick = self.analog["data_headers"][0]["Timestamp"]
        n_samples = self.nsx_data.shape[1]
        start_s = start_tick / self._nsx_tick_rate
        self.nsx_abs_time = start_s + np.arange(n_samples) / self.nsx_samplingrate

        if self.verbose:
            n_ch = len(self.analog["elec_ids"])
            print(
                f"  {n_ch} channels, {n_samples} samples, "
                f"{self.nsx_samplingrate} Hz"
            )

        return self

    def load_online_spikes(self, nev_path: str | Path | None = None) -> "BlackRockLoader":
        """
        Load online spike timing (HUB-*.nev). Populates self.spike_times
        (seconds), self.spike_channel and self.spike_unit, ready for
        segment_spikes / export_online_spikes. If the loader was built with
        load_online_wave=True, also populates self.spike_waveform (microVolts,
        shape (nSpikes, nSamp)) for segment_waveforms / export_online_wave.
        """
        nev_path = Path(nev_path) if nev_path is not None else self.spike_nev_path
        if nev_path is None:
            raise RuntimeError(
                "No spike file resolved; construct with load_online_spikes=True "
                "(or load_online_wave=True)."
            )
        if self.verbose:
            print(f"Loading spikes: {nev_path.name}")

        times_s, channel, unit, waveform, n_dropped = self._read_online_spikes(
            nev_path,
            read_waveforms=self.load_online_wave,
            include_unsorted=self.include_unsorted,
        )
        self.spike_times = times_s
        self.spike_channel = channel
        self.spike_unit = unit
        self.spike_waveform = waveform

        if self.verbose:
            msg = f"  {len(times_s)} spikes loaded"
            if n_dropped:
                msg += f" (dropped {n_dropped} unsorted/noise)"
            if waveform is not None:
                msg += f" (waveforms {waveform.shape[1]} samples/spike)"
            print(msg)

        return self

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
            print(f"  Outcomes:")

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
        comments, times_s, _ = self._get_comments()
        print(f"{'Time (s)':>20}  Comment")
        print("-" * 60)
        for t, c in zip(times_s, comments):
            print(f"{t:>20.6f}  {c}")

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

    def export_analog(self) -> Path:
        """
        Segment the analog stream into trials and save as a .mat file.
        Must be called after load_nev() and load_analog().

        Returns
        -------
        analog_mat_path
        """
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_analog().")
        if self.nsx_data is None:
            raise RuntimeError("Call load_analog() before export_analog().")

        from scipy.io import savemat

        analog = segment_analog(
            self.trials,
            self.nsx_data,
            self.nsx_abs_time,
            self.nsx_samplingrate,
            self.pre_ms,
            self.post_ms,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        savemat(str(self.analog_mat_path), {"analog": analog})
        if self.verbose:
            print(
                f"  Analog segmented ({analog['data'].shape[1]} trials) "
                f"-> {self.analog_mat_path.name}"
            )
        return self.analog_mat_path

    def export_online_spikes(self) -> Path | None:
        """
        Rasterize online spikes into per-trial bins and save as a .mat file.
        Must be called after load_nev() and load_online_spikes(). Returns None
        (and writes nothing) when there are no spikes to export — e.g. all were
        unsorted/noise and dropped (include_unsorted=False).

        Returns
        -------
        spikes_mat_path or None
        """
        if self.trials is None:
            raise RuntimeError("Call load_nev() before export_online_spikes().")
        if self.spike_times is None:
            raise RuntimeError("Call load_online_spikes() before export_online_spikes().")
        if len(self.spike_times) == 0:
            if self.verbose:
                print("  No sorted spikes found; skipping spike export.")
            return None

        from scipy.io import savemat

        online_spike = segment_spikes(
            self.trials,
            self.spike_times,
            self.spike_channel,
            self.spike_unit,
            self.pre_ms,
            self.post_ms,
            self.bin_ms,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        savemat(str(self.spikes_mat_path), {"online_spike": online_spike})
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
        file. Must be called after load_nev() and load_online_spikes() with the loader
        built using load_online_wave=True.

        The dense waveform array (units x trials x spikes x samples) easily
        exceeds scipy savemat's 4 GB MAT-v5 limit, so the file is written as
        MATLAB **v7.3** (HDF5) via hdf5storage, mirroring the MATLAB loader's
        `-v7.3` save. Large arrays are gzip-compressed (the NaN padding
        compresses well), so the file on disk is far smaller than the in-memory
        array.

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
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._savemat_v73(
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

    @staticmethod
    def _savemat_v73(path: Path, mdict: dict) -> None:
        """Write ``mdict`` as a MATLAB **v7.3** (HDF5) .mat file via hdf5storage.

        Used for the spike-waveform export, whose dense array routinely exceeds
        scipy savemat's 4 GB MAT-v5 (uint32 byte-count) limit. ``oned_as='row'``
        and row char strings match the scipy-based analog/spike exporters, so
        MATLAB sees the same field orientations; ``store_python_metadata=False``
        keeps the struct clean (no extra Python-type fields)."""
        try:
            import hdf5storage
        except ImportError as e:
            raise ImportError(
                "Saving spike waveforms needs MATLAB v7.3 (HDF5) support, which "
                "requires the hdf5storage package. Install it with:\n"
                "  pip install hdf5storage"
            ) from e
        hdf5storage.savemat(
            str(path),
            mdict,
            format="7.3",
            store_python_metadata=False,
            oned_as="row",
            truncate_existing=True,
        )

    def run(self) -> tuple[Path, list[str]]:
        """
        Convenience: load_nev() + export(), then analog/spikes if their files
        were resolved at construction (load_analog / load_online_spikes).

        Returns
        -------
        (output_dir, filenames)
            output_dir : Path — the directory where files were written
            filenames  : list[str] — names of all exported files
        """
        self.load_nev().export()
        filenames = [self.txt_path.name, self.csv_path.name]
        if self.ns_path is not None:
            self.load_analog()
            self.export_analog()
            filenames.append(self.analog_mat_path.name)
        if self.spike_nev_path is not None:
            self.load_online_spikes()
            if self.spike_times is None or len(self.spike_times) == 0:
                # Everything was unsorted/noise (include_unsorted=False) -> nothing to export.
                if self.verbose:
                    print("  No sorted spikes found; skipping spike/waveform export.")
            else:
                if self._export_online_spikes:
                    self.export_online_spikes()
                    filenames.append(self.spikes_mat_path.name)
                if self.load_online_wave:
                    self.export_online_wave()
                    filenames.append(self.waveforms_mat_path.name)
        return self.output_dir, filenames

    @classmethod
    def run_batch(
        cls,
        session_path: str | Path,
        dates: str | list[str] | None = None,
        *,
        data_type: str = "raw_data",
        load_analog: bool = False,
        ns_marker: str = "ns2",
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
        data_type, load_analog, ns_marker, load_online_spikes, load_online_wave,
        include_unsorted, pre_ms, post_ms, bin_ms, output_folder, verbose
            Forwarded to each per-date loader (see BlackRockLoader).
        skip_existing : bool
            Skip dates whose expmeta .txt and trials .csv already exist, so a
            re-run only processes new folders. Default False.

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
                    load_analog=load_analog,
                    ns_marker=ns_marker,
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

    # ── Internal NEV readers ────────────────────────────────────────────────

    def _get_comments(self) -> tuple[list[str], list[float], Path]:
        """Return (comments, times_s, source_path) from the first comment
        candidate that actually carries comments (NSP, then HUB legacy)."""
        if not self._comment_candidates:
            raise FileNotFoundError(
                f"No {C.COMMENT_PREFIX_PRIMARY}-*.nev or {C.COMMENT_PREFIX_LEGACY}-*.nev "
                f"comment file found in: {self.data_folder}"
            )
        last_err: Exception | None = None
        for path in self._comment_candidates:
            try:
                comments, times_s = self._read_comments(path)
                return comments, times_s, path
            except ValueError as e:
                last_err = e  # no comments in this file; try the next candidate
        raise ValueError(
            f"No comment events found in any of "
            f"{[p.name for p in self._comment_candidates]}: {last_err}"
        )

    def _read_comments(self, nev_path: Path) -> tuple[list[str], list[float]]:
        """Open the NEV file via brpylib and extract comment strings + times.
        Raises ValueError if the file carries no comment events."""
        # brpylib uses numpy chararray.tostring() which was removed in numpy 2.0.
        # Patch it back before loading so existing brpylib installations work.
        import numpy as np
        if not hasattr(np.chararray, "tostring"):
            np.chararray.tostring = np.chararray.tobytes  # type: ignore[attr-defined]

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
        # analog channels all share the same hardware clock reference.
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
                deduped_ticks.append(t)

        times_s = [t / self._tick_rate for t in deduped_ticks]
        return deduped_comments, times_s

    def _read_online_spikes(
        self,
        nev_path: Path,
        read_waveforms: bool = False,
        include_unsorted: bool = False,
    ):
        """Online spike reader: open the HUB NEV via brpylib and extract spike
        timing into the common per-spike representation.

        Returns (times_s, channel, unit, waveform, n_dropped) — the same shape a
        future ``_read_offline_spikes`` would return, so both feed the shared
        ``segment_spikes`` / ``segment_waveforms``. ``waveform`` is None unless
        ``read_waveforms`` is True, in which case it is an (nSpikes, nSamp) array
        of microVolts (raw int16 scaled per electrode, like the MATLAB version).
        When ``include_unsorted`` is False, spikes whose unit is in
        C.UNSORTED_UNIT_IDS (0 unsorted, 255 noise) are dropped via the shared
        ``drop_units`` helper before scaling (the memory saver); ``n_dropped``
        counts how many were removed. Raises ValueError if no spikes."""
        import numpy as np
        if not hasattr(np.chararray, "tostring"):
            np.chararray.tostring = np.chararray.tobytes  # type: ignore[attr-defined]

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
        tick_rate = nev.basic_header["TimeStampResolution"]

        spikes = data.get("spike_events", {})
        raw_ticks = spikes.get("TimeStamps", [])
        if len(raw_ticks) == 0:
            raise ValueError(f"No spike timestamps in {nev_path}.")

        # TimeStamps are raw ticks; divide by the file's tick rate so spike
        # seconds share the comment/analog clock. (Channel = electrode id.)
        times_s = np.asarray(raw_ticks, dtype=float) / tick_rate
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
            times_s, channel, unit, raw, n_dropped = drop_units(
                times_s, channel, unit, raw
            )

        waveform = None
        if read_waveforms:
            waveform = self._scale_waveforms(nev, raw, channel)
        return times_s, channel, unit, waveform, n_dropped

    @staticmethod
    def _scale_waveforms(nev, raw, channel) -> "np.ndarray":  # noqa: F821
        """Convert raw int16 spike waveforms to microVolts.

        ``raw`` is the (nSpikes, nSamp) int16 array from
        ``spike_events["Waveforms"]`` (already masked by the caller). Each
        electrode's NEUEVWAV extended header carries a DigitizationFactor
        (nV/bit); microVolts = raw * DigitizationFactor / 1000, matching the
        MATLAB BlackrockLoader (DigitalFactor / 1000)."""
        import numpy as np

        raw = np.asarray(raw, dtype=float)  # (nSpikes, nSamp)

        # electrode id -> DigitizationFactor (nV/bit) from NEUEVWAV headers
        factor_map = {
            h["ElectrodeID"]: h["DigitizationFactor"]
            for h in nev.extended_headers
            if "ElectrodeID" in h and "DigitizationFactor" in h
        }
        channel = np.asarray(channel, dtype=float)
        factors = np.array(
            [factor_map.get(int(c), 0.0) for c in channel], dtype=float
        )
        # microVolts = raw int16 * (nV/bit) / 1000
        return raw * factors[:, None] / 1000.0
