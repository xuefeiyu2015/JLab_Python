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
    Load a Blackrock NEV file and export experiment metadata + trial data.
    Optionally loads and exports an NSx analog file (e.g. .ns2 for eye tracking).

    Parameters
    ----------
    nev_path : str or Path
        Path to the .nev file.
    output_dir : str or Path, optional
        Directory for output files. Defaults to the same directory as nev_path.
    date_str : str, optional
        Date string used in output filenames (YYYY-MM-DD).
        Auto-detected from the NEV header TimeOrigin if not provided.
    ns_path : str or Path, optional
        Path to the analog NSx file (.ns2, .ns4, .ns6).
        If provided, run() will also load and export analog data to
        Blackrock_YYYY-MM-DD_analog.csv.
    verbose : bool
        Print progress messages. Default True.

    Examples
    --------
    >>> from jlab import BlackRockLoader

    >>> # NEV only
    >>> loader = BlackRockLoader("file.nev", output_dir="output/")
    >>> output_dir, files = loader.run()

    >>> # NEV + analog
    >>> loader = BlackRockLoader("file.nev", output_dir="output/", ns_path="file.ns2")
    >>> output_dir, files = loader.run()  # files includes analog csv name
    """

    def __init__(
        self,
        nev_path: str | Path,
        output_dir: str | Path | None = None,
        date_str: str | None = None,
        ns_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.nev_path = Path(nev_path)
        self.output_dir = Path(output_dir) if output_dir else self.nev_path.parent
        self.ns_path = Path(ns_path) if ns_path else None
        self.verbose = verbose
        self.date_str = date_str

        self.experiment: dict | None = None
        self.trials: list[dict] | None = None
        self.analog: dict | None = None
        self._tick_rate: int | None = None

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
        Read and parse the NEV file. Populates self.experiment and self.trials.
        Returns self for method chaining.
        """
        if self.verbose:
            print(f"Loading NEV: {self.nev_path.name}")

        comments, times_s = self._read_nev()

        if self.verbose:
            print(f"  {len(comments)} events found")

        self.experiment, self.trials = parse_comments(comments, times_s)
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
        exp = self.experiment or {}
        start, end = exp.get("start", float("nan")), exp.get("end", float("nan"))

        print("Session Summary")
        print("─" * 50)
        if not _isnan(start) and not _isnan(end):
            duration = end - start
            mins, secs = int(duration // 60), duration % 60
            print(f"Session length : {mins} min {secs:.1f} sec")
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
        if self.experiment is None or self.trials is None:
            raise RuntimeError("Call load_nev() before export().")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        write_expmeta(self.experiment, self.txt_path)
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
