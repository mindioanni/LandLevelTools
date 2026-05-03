
from __future__ import annotations

from pathlib import Path
import importlib
import io
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import contextlib

import paths_config


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = Path(sys.executable)
BATCH_MAIN = PROJECT_DIR / "batch_main.py"

YES_NO = ["y", "n"]


class RTKLIBBaselineGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RTKLIB Baseline Orchestrator")
        self.geometry("1080x880")
        self.minsize(980, 760)

        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.project_name_var = tk.StringVar(value="")
        self.cors_report_var = tk.StringVar(value="")
        self.rover_root_var = tk.StringVar(value="")
        self.base_root_var = tk.StringVar(value="")
        self.products_root_var = tk.StringVar(value="")
        self.output_root_var = tk.StringVar(value="")
        self.rnx2rtkp_var = tk.StringVar(value=paths_config.get_default_rnx2rtkp_path())

        self.provider_var = tk.StringVar(value="COD")
        self.series_var = tk.StringVar(value="FIN")
        self.project_var = tk.StringVar(value="MGX")
        self.product_mode_var = tk.StringVar(value="precise")
        self.download_missing_var = tk.StringVar(value="y")
        self.downloader_script_var = tk.StringVar(value=paths_config.get_default_downloader_script_path())
        self.downloader_python_var = tk.StringVar(value=paths_config.get_default_downloader_python_path())

        self.use_ionex_var = tk.BooleanVar(value=False)
        self.use_antex_var = tk.BooleanVar(value=True)
        self.use_blq_var = tk.BooleanVar(value=False)
        self.use_bia_osb_var = tk.BooleanVar(value=False)

        self.processing_mode_var = tk.StringVar(value="static")
        self.min_overlap_var = tk.StringVar(value="45")
        self.matching_strategy_var = tk.StringVar(value="best_overlap_per_rover")
        self.overwrite_var = tk.StringVar(value="n")

        self.frequency_var = tk.StringVar(value="L1+L2")
        self.el_mask_var = tk.StringVar(value="15")
        self.solution_type_var = tk.StringVar(value="forward")
        self.ar_mode_var = tk.StringVar(value="continuous")
        self.ar_threshold_var = tk.StringVar(value="3.0")
        self.output_format_var = tk.StringVar(value="ECEF XYZ")
        self.gps_var = tk.BooleanVar(value=True)
        self.glo_var = tk.BooleanVar(value=False)
        self.gal_var = tk.BooleanVar(value=True)
        self.bds_var = tk.BooleanVar(value=True)
        self.qzs_var = tk.BooleanVar(value=False)
        self.sbs_var = tk.BooleanVar(value=False)

        self.final_window_var = tk.StringVar(value="30")
        self.q1_only_var = tk.BooleanVar(value=True)
        self.min_fixed_percent_var = tk.StringVar(value="80")
        self.min_ratio_var = tk.StringVar(value="3.0")
        self.generate_plots_var = tk.BooleanVar(value=True)

        self.execution_mode_var = tk.StringVar(value="run")
        self.generate_report_var = tk.BooleanVar(value=True)
        self.report_filename_var = tk.StringVar(value=paths_config.REPORT_FILENAME)
        self.trace_level_var = tk.StringVar(value="0")

        self.cleanup_runs_var = tk.BooleanVar(value=True)
        self.cleanup_logs_var = tk.BooleanVar(value=True)
        self.cleanup_tables_var = tk.BooleanVar(value=True)
        self.cleanup_report_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()
        self._poll_output_queue()

    def _browse_file(self, var, title):
        selected = filedialog.askopenfilename(title=title)
        if selected:
            var.set(selected)

    def _browse_dir(self, var, title):
        selected = filedialog.askdirectory(title=title)
        if selected:
            var.set(selected)

    def _build_layout(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        self._build_project_tab(nb)
        self._build_products_tab(nb)
        self._build_matching_tab(nb)
        self._build_options_tab(nb)
        self._build_qc_tab(nb)
        self._build_execution_tab(nb)
        self._build_cleanup_tab(nb)
        self._build_output_tab(nb)

    def _row_entry(self, parent, row, label, var, browse=None):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        if browse == "file":
            ttk.Button(parent, text="Browse...", command=lambda: self._browse_file(var, label)).grid(row=row, column=2, padx=(8,0), pady=4)
        elif browse == "dir":
            ttk.Button(parent, text="Browse...", command=lambda: self._browse_dir(var, label)).grid(row=row, column=2, padx=(8,0), pady=4)

    def _build_project_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Project paths")
        self._row_entry(tab, 0, "Project name", self.project_name_var)
        self._row_entry(tab, 1, "CORS GINAN report HTML", self.cors_report_var, "file")
        self._row_entry(tab, 2, "Rover GNSSBM RINEX folder", self.rover_root_var, "dir")
        self._row_entry(tab, 3, "CORS/base RINEX folder", self.base_root_var, "dir")
        self._row_entry(tab, 4, "Products / models folder", self.products_root_var, "dir")
        self._row_entry(tab, 5, "Output folder (blank = rover/RTK_process)", self.output_root_var, "dir")
        self._row_entry(tab, 6, "rnx2rtkp executable", self.rnx2rtkp_var, "file")

    def _build_products_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Products")

        rows = [
            ("Product provider", self.provider_var, paths_config.PRODUCT_PROVIDERS),
            ("Product series", self.series_var, paths_config.PRODUCT_SERIES),
            ("Product project", self.project_var, paths_config.PRODUCT_PROJECTS),
            ("Product mode", self.product_mode_var, paths_config.PRODUCT_MODES),
            ("Download missing products", self.download_missing_var, YES_NO),
        ]
        for i, (label, var, values) in enumerate(rows):
            ttk.Label(tab, text=label).grid(row=i, column=0, sticky="w", padx=(0,8), pady=4)
            ttk.Combobox(tab, textvariable=var, values=values, state="readonly", width=18).grid(row=i, column=1, sticky="w", pady=4)

        self._row_entry(tab, 5, "Downloader script", self.downloader_script_var, "file")
        self._row_entry(tab, 6, "Downloader Python", self.downloader_python_var, "file")

        ttk.Checkbutton(tab, text="Use IONEX if available", variable=self.use_ionex_var).grid(row=7, column=0, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Use ANTEX model", variable=self.use_antex_var).grid(row=8, column=0, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Use BLQ / ocean loading model", variable=self.use_blq_var).grid(row=9, column=0, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Resolve BIA/OSB products", variable=self.use_bia_osb_var).grid(row=10, column=0, sticky="w", pady=4)

    def _build_matching_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Matching")
        ttk.Label(tab, text="Processing mode").grid(row=0, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Combobox(tab, textvariable=self.processing_mode_var, values=paths_config.PROCESSING_MODES, state="readonly", width=18).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="Minimum rover/base overlap (min)").grid(row=1, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.min_overlap_var, width=18).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="Matching strategy").grid(row=2, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Combobox(tab, textvariable=self.matching_strategy_var, values=paths_config.MATCHING_STRATEGIES, state="readonly", width=26).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="Overwrite existing outputs").grid(row=3, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Combobox(tab, textvariable=self.overwrite_var, values=YES_NO, state="readonly", width=18).grid(row=3, column=1, sticky="w", pady=4)

    def _build_options_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="RTKLIB options")

        rows = [
            ("Frequencies", self.frequency_var, paths_config.FREQUENCY_MODES),
            ("Solution type", self.solution_type_var, paths_config.SOLUTION_TYPES),
            ("Ambiguity resolution mode", self.ar_mode_var, paths_config.AR_MODES),
            ("Output coordinate format", self.output_format_var, paths_config.OUTPUT_FORMATS),
        ]
        for i, (label, var, values) in enumerate(rows):
            ttk.Label(tab, text=label).grid(row=i, column=0, sticky="w", padx=(0,8), pady=4)
            ttk.Combobox(tab, textvariable=var, values=values, state="readonly", width=20).grid(row=i, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="Elevation mask (deg)").grid(row=4, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.el_mask_var, width=20).grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(tab, text="AR ratio threshold").grid(row=5, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.ar_threshold_var, width=20).grid(row=5, column=1, sticky="w", pady=4)

        sys_frame = ttk.LabelFrame(tab, text="GNSS systems", padding=8)
        sys_frame.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10,4))
        for i, (label, var) in enumerate([("GPS", self.gps_var), ("GLO", self.glo_var), ("GAL", self.gal_var), ("BDS", self.bds_var), ("QZS", self.qzs_var), ("SBS", self.sbs_var)]):
            ttk.Checkbutton(sys_frame, text=label, variable=var).grid(row=0, column=i, sticky="w", padx=(0,14))

    def _build_qc_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="QC / final solution")
        ttk.Label(tab, text="Final solution window (min)").grid(row=0, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.final_window_var, width=18).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Recommended minimum final solution window: 15 min").grid(row=1, column=1, sticky="w", pady=(0,8))

        ttk.Checkbutton(tab, text="Use Q=1 fixed only for final solution", variable=self.q1_only_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(tab, text="Minimum fixed percentage (%)").grid(row=3, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.min_fixed_percent_var, width=18).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Minimum ratio for fixed epochs").grid(row=4, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.min_ratio_var, width=18).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Generate plots", variable=self.generate_plots_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=4)

    def _build_execution_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Execution / report")

        ttk.Label(tab, text="Execution mode").grid(row=0, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Combobox(tab, textvariable=self.execution_mode_var, values=paths_config.EXECUTION_MODES, state="readonly", width=18).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Generate HTML report", variable=self.generate_report_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        self._row_entry(tab, 2, "Report filename", self.report_filename_var)
        ttk.Label(tab, text="RTKLIB trace level").grid(row=3, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(tab, textvariable=self.trace_level_var, width=18).grid(row=3, column=1, sticky="w", pady=4)

        ttk.Button(tab, text="Run processing", command=self._start_batch).grid(row=4, column=0, sticky="w", pady=(16,4))
        ttk.Button(tab, text="Stop process", command=self._stop_process).grid(row=4, column=1, sticky="w", pady=(16,4))
        ttk.Label(tab, textvariable=self.status_var).grid(row=5, column=0, columnspan=3, sticky="w", pady=4)

    def _build_cleanup_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Cleanup generated files")
        ttk.Checkbutton(tab, text="Remove runs/", variable=self.cleanup_runs_var).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Remove logs/", variable=self.cleanup_logs_var).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Remove CSV/JSON tables", variable=self.cleanup_tables_var).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Checkbutton(tab, text="Remove HTML report", variable=self.cleanup_report_var).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Button(tab, text="Preview cleanup", command=lambda: self._run_cleanup(False)).grid(row=4, column=0, sticky="w", pady=(16,4))
        ttk.Button(tab, text="Run cleanup after confirmation", command=lambda: self._run_cleanup(True)).grid(row=4, column=1, sticky="w", pady=(16,4))

    def _build_output_tab(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Output log")
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        ttk.Button(tab, text="Clear output", command=lambda: self.output_text.delete("1.0", "end")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.output_text = tk.Text(tab, wrap="word", height=30)
        self.output_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.output_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scrollbar.set)

    def _nav_systems(self):
        out = []
        if self.gps_var.get(): out.append("G")
        if self.glo_var.get(): out.append("R")
        if self.gal_var.get(): out.append("E")
        if self.bds_var.get(): out.append("C")
        if self.qzs_var.get(): out.append("J")
        if self.sbs_var.get(): out.append("S")
        return ",".join(out)

    def _build_command(self):
        cmd = [
            str(PYTHON_EXE), str(BATCH_MAIN),
            "--project-name", self.project_name_var.get().strip(),
            "--cors-report", self.cors_report_var.get().strip(),
            "--rover-root", self.rover_root_var.get().strip(),
            "--base-root", self.base_root_var.get().strip(),
            "--products-root", self.products_root_var.get().strip(),
            "--rnx2rtkp", self.rnx2rtkp_var.get().strip(),
            "--provider", self.provider_var.get(),
            "--series", self.series_var.get(),
            "--project", self.project_var.get(),
            "--product-mode", self.product_mode_var.get(),
            "--download-missing", self.download_missing_var.get(),
            "--downloader-script", self.downloader_script_var.get().strip(),
            "--downloader-python", self.downloader_python_var.get().strip(),
            "--use-ionex", "y" if self.use_ionex_var.get() else "n",
            "--use-antex", "y" if self.use_antex_var.get() else "n",
            "--use-blq", "y" if self.use_blq_var.get() else "n",
            "--use-bia-osb", "y" if self.use_bia_osb_var.get() else "n",
            "--processing-mode", self.processing_mode_var.get(),
            "--min-overlap-min", self.min_overlap_var.get().strip(),
            "--matching-strategy", self.matching_strategy_var.get(),
            "--overwrite", self.overwrite_var.get(),
            "--frequencies", self.frequency_var.get(),
            "--el-mask", self.el_mask_var.get().strip(),
            "--solution-type", self.solution_type_var.get(),
            "--ar-mode", self.ar_mode_var.get(),
            "--ar-threshold", self.ar_threshold_var.get().strip(),
            "--nav-systems", self._nav_systems(),
            "--output-format", self.output_format_var.get(),
            "--final-window-min", self.final_window_var.get().strip(),
            "--q1-only-final", "y" if self.q1_only_var.get() else "n",
            "--min-fixed-percent", self.min_fixed_percent_var.get().strip(),
            "--min-ratio", self.min_ratio_var.get().strip(),
            "--generate-plots", "y" if self.generate_plots_var.get() else "n",
            "--execution-mode", self.execution_mode_var.get(),
            "--generate-report", "y" if self.generate_report_var.get() else "n",
            "--report-filename", self.report_filename_var.get().strip(),
            "--trace-level", self.trace_level_var.get().strip(),
        ]
        if self.output_root_var.get().strip():
            cmd.extend(["--output-root", self.output_root_var.get().strip()])
        return cmd

    def _append_output(self, text):
        self.output_text.insert("end", text)
        self.output_text.see("end")

    def _start_batch(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Process already running", "A process is already running.")
            return
        cmd = self._build_command()
        self._append_output("\\n=== Starting RTKLIB Baseline Orchestrator ===\\n")
        self._append_output(" ".join(cmd) + "\\n\\n")
        self.status_var.set("Running...")
        thread = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        thread.start()

    def _run_subprocess(self, cmd):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line)
            code = self.process.wait()
            self.output_queue.put(f"\\n=== Finished with return code {code} ===\\n")
            self.output_queue.put("__PROCESS_FINISHED__")
        except Exception as exc:
            self.output_queue.put(f"ERROR: {exc}\\n")
            self.output_queue.put("__PROCESS_FINISHED__")

    def _stop_process(self):
        proc = self.process
        if proc is not None and proc.poll() is None:
            self._append_output("\\n=== Termination requested ===\\n")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()

    def _run_cleanup(self, execute: bool):
        output_root = self.output_root_var.get().strip()
        if not output_root:
            rover_root = self.rover_root_var.get().strip()
            if not rover_root:
                messagebox.showerror("Missing path", "Set rover root or output folder first.")
                return
            output_root = str(paths_config.output_root_from_rover_root(rover_root))

        if execute:
            answer = messagebox.askyesno("Confirm cleanup", f"Clean generated files under:\\n{output_root}\\n\\nProceed?")
            if not answer:
                return

        try:
            import cleanup_service
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = cleanup_service.clean_generated_files(
                    rtk_process_root=output_root,
                    execute=execute,
                    remove_runs=self.cleanup_runs_var.get(),
                    remove_logs=self.cleanup_logs_var.get(),
                    remove_tables=self.cleanup_tables_var.get(),
                    remove_report=self.cleanup_report_var.get(),
                )
            self._append_output("\\n=== Cleanup ===\\n")
            for target in result["targets"]:
                self._append_output(target + "\\n")
            self._append_output(result["message"] + "\\n")
        except Exception as exc:
            self._append_output(f"ERROR during cleanup: {exc}\\n")

    def _poll_output_queue(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item == "__PROCESS_FINISHED__":
                    self.status_var.set("Ready.")
                    self.process = None
                else:
                    self._append_output(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_output_queue)


def main():
    app = RTKLIBBaselineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
