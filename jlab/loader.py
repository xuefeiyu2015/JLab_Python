"""
Finally, combine everything
BlackRockLoader — main user-facing class.
Reads a Blackrock .nev file and exports the same .txt and .csv files
produced by BackRockFileLoader.m.
"""

from __future__ import annotations

from pathlib import Path

from ._analog import parse_analog
from ._exporter import write_expmeta, write_trials_csv
from ._features import compute_derived_features
from ._parser import parse_comments


class BlackRockLoader:
    """
    Load a Blackrock session and export experiment metadata + trial data.
    Optionally loads and exports an NSx analog file (e.g. .ns2 for eye tracking).

    Construction is folder-based, mirroring BackRockFileLoader.m: no filenames
    needed. `session_path` is the directory that holds the `raw_data` and
    `export_data` sub-folders (assemble it in the notebook, e.g.
    ``Path(Basic_Path) / Monkey / Location``). The .nev (and optional .ns2) are
    auto-detected from ``session_path / data_type / date`` — the largest file
    wins when several exist — and output goes to
    ``session_path / output_folder / date``.

    Parameters
    ----------
    session_path : str or Path
        Directory containing the `data_type` and `output_folder` sub-folders
        (i.e. ``Basic_Path / "Monkey <name>" / Location``).
    date : str
        Year-month-date folder, e.g. "2026-06-17". Also used as the date string
        in output filenames. To process several dates at once, use the
        ``run_batch`` classmethod instead.
    data_type : str
        Raw-data sub-folder name. Default "raw_data".
    nev_filename : str, optional
        Explicit .nev filename to load instead of auto-picking the largest.
        Use list_nev_files() to see what is available.
    load_analog : bool
        If True, auto-detect an NSx file (the largest one, or `ns_filename` if
        given). Skipped with a message if none found.
    ns_marker : str
        NSx file extension to look for, without the dot. Default "ns2" (eye
        tracking); use "ns6" or others for different sampling rates.
    ns_filename : str, optional
        Explicit NSx filename to load instead of auto-picking the largest.
    output_folder : str
        Export sub-folder name. Default "export_data".
    verbose : bool
        Print progress messages. Default True.

    Examples
    --------
    >>> from jlab import BlackRockLoader
    >>> from pathlib import Path
    >>> session = Path(Basic_Path) / "Monkey Porthos" / "in_lab"

    >>> # See which .nev files are in the folder, then load one
    >>> BlackRockLoader.list_nev_files(session, "2026-06-17")

    >>> # NEV only (largest auto-detected)
    >>> loader = BlackRockLoader(session, "2026-06-17")
    >>> output_dir, files = loader.run()

    >>> # NEV + analog, picking a specific .nev
    >>> loader = BlackRockLoader(session, "2026-06-17",
    ...                          nev_filename="Hub1-Porthos_....nev", load_analog=True)
    >>> output_dir, files = loader.run()  # files includes analog csv name

    >>> # Several dates at once (or run_batch(session) to do every date folder)
    >>> report = BlackRockLoader.run_batch(session, dates=["2026-06-17", "2026-06-18"])
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
        output_folder: str = "export_data",
        verbose: bool = True,
    ) -> None:
        self.session_path = Path(session_path)
        self.date = date
        self.date_str = date
        self.verbose = verbose
        self.data_folder = self.session_path / data_type / date

        # Auto-detect the .nev (largest), or use an explicit nev_filename.
        self.nev_path = self._resolve_file(self.data_folder, "*.nev", nev_filename, "NEV")

        # Auto-detect the NSx analog file only when requested.
        self.ns_marker = ns_marker.lstrip(".").lower()
        self.ns_path: Path | None = None
        if load_analog:
            pattern = f"*.{self.ns_marker}"
            try:
                self.ns_path = self._resolve_file(
                    self.data_folder, pattern, ns_filename, self.ns_marker.upper()
                )
            except FileNotFoundError:
                if verbose:
                    print(f"No .{self.ns_marker} analog file found; skipping analog load.")

        self.output_dir = self.session_path / output_folder / date

        if verbose:
            print(f"NEV file      : {self.nev_path.name}")
            if self.ns_path is not None:
                print(f"Analog file   : {self.ns_path.name}")
            print(f"Output dir    : {self.output_dir}")

        self.experiments: list[dict] | None = None
        self.trials: list[dict] | None = None
        self.analog: dict | None = None
        self._tick_rate: int | None = None

    # ── File detection helpers ────────────────────────────────────────────

    @staticmethod
    def _resolve_file(
        folder: Path, pattern: str, filename: str | None, label: str
    ) -> Path:
        """Return an explicit `filename` in `folder`, else the largest match of
        `pattern`. Raises FileNotFoundError if nothing matches."""
        if filename is not None:
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")
            return path
        matches = sorted(folder.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No {pattern} file found in: {folder}")
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
    def analog_csv_path(self) -> Path:
        return self.output_dir / f"Blackrock_{self.date_str}_analog.csv"

    # ── Public methods ─────────────────────────────────────────────────────

    def load_nev(self) -> "BlackRockLoader":
        """
        Read and parse the NEV file. Populates self.experiments and self.trials.
        Returns self for method chaining.
        """
        if self.verbose:
            print(f"Loading NEV: {self.nev_path.name}")

        comments, times_s = self._read_nev()

        if self.verbose:
            print(f"  {len(comments)} events found")

        self.experiments, self.trials = parse_comments(comments, times_s)
        compute_derived_features(self.trials)

        if self.verbose:
            n_complete = sum(t["Save_complete"] for t in self.trials)
            print(f"  {len(self.trials)} trials parsed ({n_complete} complete)")

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
        comments, times_s = self._read_nev()
        print(f"{'Time (s)':>20}  Comment")
        print("-" * 60)
        for t, c in zip(times_s, comments):
            print(f"{t:>20.6f}  {c}")

    def export(self) -> tuple[Path, Path]:
        """
        Write expmeta .txt and trials .csv to output_dir.
        Must be called after load().

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
        Parse and export analog data to a CSV file.
        Must be called after load_analog().

        Returns
        -------
        analog_csv_path
        """
        if self.analog is None:
            raise RuntimeError("Call load_analog() before export_analog().")

        df = parse_analog(self.analog)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.analog_csv_path, index=False)
        return self.analog_csv_path

    def run(self) -> tuple[Path, list[str]]:
        """
        Convenience: load_nev() + export().
        If ns_path was provided, also runs load_analog() + export_analog().

        Returns
        -------
        (output_dir, filenames)
            output_dir : Path — the directory where files were written
            filenames  : list[str] — names of all exported files
        """
        self.load_nev().export()
        filenames = [self.txt_path.name, self.csv_path.name]
        if self.ns_path is not None:
            self.load_analog(self.ns_path)
            self.export_analog()
            filenames.append(self.analog_csv_path.name)
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
        output_folder: str = "export_data",
        skip_existing: bool = False,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Load and export several date folders in one call.

        Builds one single-date BlackRockLoader per date and runs it, reusing the
        same parse/export pipeline. A date that fails (e.g. missing .nev, parse
        error) is recorded and skipped so the batch always finishes.

        Parameters
        ----------
        session_path : str or Path
            Directory containing the `data_type` and `output_folder` sub-folders
            (i.e. ``Basic_Path / "Monkey <name>" / Location``).
        dates : str or list[str] or None
            A year-month-date string, a list of them, or None/empty to discover
            and process every ``YYYY-MM-DD`` folder under
            ``session_path / data_type`` (sorted).
        data_type, load_analog, ns_marker, output_folder, verbose
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

    def load_analog(
        self,
        ns_path: str | Path,
        elec_ids="all",
        start_time_s: float = 0,
        data_time_s="all",
        downsample: int = 1,
    ) -> "BlackRockLoader":
        """
        Load a continuous analog NSx file (e.g. .ns2 for eye tracking).
        Populates self.analog with the data returned by brpylib.NsxFile.

        Parameters
        ----------
        ns_path : str or Path
            Path to the .ns2 / .ns4 / .ns6 file.
        elec_ids : list or 'all'
            Electrode IDs to load (1-indexed). Default 'all'.
        start_time_s : float
            Start time in seconds. Default 0.
        data_time_s : float or 'all'
            Duration to load in seconds. Default 'all'.
        downsample : int
            Downsampling factor. Default 1 (no downsampling).

        After loading, self.analog contains:
            'data'        : list of arrays, one per segment (channels × samples)
            'elec_ids'    : list of loaded electrode IDs
            'samp_per_s'  : sampling rate in Hz
            'data_headers': list of per-segment metadata dicts
            'start_time_s': requested start time
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

        ns_path = Path(ns_path)
        if self.verbose:
            print(f"Loading analog: {ns_path.name}")

        nsx = NsxFile(str(ns_path))
        self.analog = nsx.getdata(elec_ids, start_time_s, data_time_s, downsample)
        nsx.close()

        if self.verbose:
            n_ch  = len(self.analog["elec_ids"])
            n_seg = len(self.analog["data"])
            hz    = self.analog["samp_per_s"]
            n_samp = self.analog["data"][0].shape[1] if self.analog["data"] else 0
            print(f"  {n_ch} channels, {n_seg} segment(s), {n_samp} samples, {hz} Hz")

        return self


    def _read_nev(self) -> tuple[list[str], list[float]]:
        """Open the NEV file via brpylib and extract comment strings + times."""
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

        nev = NevFile(str(self.nev_path))
        data = nev.getdata(elec_ids="all", wave_read="no_read")

        self._tick_rate = nev.basic_header["TimeStampResolution"]

        # Auto-detect date from NEV header if not provided
        if self.date_str is None:
            origin = nev.basic_header.get("TimeOrigin")
            if origin is not None and hasattr(origin, "strftime"):
                self.date_str = origin.strftime("%Y-%m-%d")
            else:
                from datetime import date
                self.date_str = date.today().isoformat()

        comments_data = data.get("comments", {})
        raw_comments: list = comments_data.get("Data", [])

        if not raw_comments:
            raise ValueError(
                f"No comment events found in {self.nev_path}. "
                "Check that the file contains behavioral event markers."
            )

        # Use raw TimeStamps (no lag correction) so behavioral events, spikes, and
        # analog channels all share the same 30kHz hardware clock reference.
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
