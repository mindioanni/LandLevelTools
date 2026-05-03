from __future__ import annotations

from pathlib import Path
import contextlib
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


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = Path(sys.executable)
BATCH_MAIN = PROJECT_DIR / "batch_main.py"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_RAW_ROOT = str(Path.home() / "data" / "RINEX" / "RAW")
DEFAULT_STATIC_PRODUCTS_ROOT = str(Path.home() / "opt" / "ginan" / "ginan-gui-linux-x64" / "_internal" / "scripts" / "GinanUI" / "app" / "resources" / "inputData" / "products")
DEFAULT_TEMPLATE_YAML_PATH = str(Path.home() / "opt" / "ginan" / "ginan-gui-linux-x64" / "_internal" / "scripts" / "GinanUI" / "app" / "resources" / "ppp_TG_GEIN_NOA_template.yaml")

PPP_PROVIDERS = ["EMR", "COD", "WUM", "IGS", "GFZ", "GRG"]
PPP_SERIES = ["FIN", "RAP"]
PPP_PROJECTS = ["MGX", "OPS"]
EXECUTION_MODES = ["run", "build_only"]
REPORT_PLOT_COLUMNS_HELP = "valid: X,Y,Z,lon,lat,h,E,N,U,all"
YES_NO_OPTIONS = ["y", "n"]


class PPPBatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("GINAN/pea PPP Batch Processor")
        self.geometry("1040x900")
        self.minsize(940, 760)

        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.raw_root_var = tk.StringVar(value=DEFAULT_RAW_ROOT)
        self.static_products_root_var = tk.StringVar(value=DEFAULT_STATIC_PRODUCTS_ROOT)
        self.template_yaml_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_YAML_PATH)
        self.sample_rate_var = tk.StringVar(value="15")
        self.provider_var = tk.StringVar(value="COD")
        self.series_var = tk.StringVar(value="FIN")
        self.project_var = tk.StringVar(value="MGX")
        self.execution_mode_var = tk.StringVar(value="run")
        self.overwrite_var = tk.StringVar(value="n")
        self.dataset_limit_var = tk.StringVar(value="all")
        self.generate_timeseries_report_var = tk.StringVar(value="y")

        self.plot_x_var = tk.BooleanVar(value=True)
        self.plot_y_var = tk.BooleanVar(value=True)
        self.plot_z_var = tk.BooleanVar(value=True)
        self.plot_lon_var = tk.BooleanVar(value=False)
        self.plot_lat_var = tk.BooleanVar(value=False)
        self.plot_h_var = tk.BooleanVar(value=True)
        self.plot_e_var = tk.BooleanVar(value=False)
        self.plot_n_var = tk.BooleanVar(value=False)
        self.plot_u_var = tk.BooleanVar(value=False)

        self.cleanup_resampled_var = tk.BooleanVar(value=True)
        self.cleanup_yaml_var = tk.BooleanVar(value=True)
        self.cleanup_pea_outputs_var = tk.BooleanVar(value=True)
        self.cleanup_downloaded_products_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()
        self._poll_output_queue()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(root, text="Batch input parameters", padding=10)
        input_frame.pack(fill="x")

        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="RAW RINEX root directory").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        raw_entry = ttk.Entry(input_frame, textvariable=self.raw_root_var)
        raw_entry.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(input_frame, text="Browse...", command=self._browse_raw_root).grid(
            row=0, column=2, sticky="e", padx=(8, 0), pady=4
        )
        ttk.Label(
            input_frame,
            text="Root directory containing the raw daily/per-file RINEX dataset folders. Each subdirectory is treated as one processing dataset.",
            wraplength=760,
        ).grid(row=1, column=1, sticky="w", pady=(0, 4))

        ttk.Label(input_frame, text="Static products directory").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        static_entry = ttk.Entry(input_frame, textvariable=self.static_products_root_var)
        static_entry.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(input_frame, text="Browse...", command=self._browse_static_products_root).grid(
            row=2, column=2, sticky="e", padx=(8, 0), pady=4
        )
        ttk.Label(
            input_frame,
            text="Directory containing static auxiliary products used by PEA during processing, such as grids, loading models, antenna models and tables.",
        ).grid(row=3, column=1, sticky="w", pady=(0, 4))

        ttk.Label(input_frame, text="YAML template path").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=4
        )
        template_entry = ttk.Entry(input_frame, textvariable=self.template_yaml_path_var)
        template_entry.grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Button(input_frame, text="Browse...", command=self._browse_template_yaml_path).grid(
            row=4, column=2, sticky="e", padx=(8, 0), pady=4
        )
        ttk.Label(
            input_frame,
            text="YAML template used to generate the dedicated PEA config for each RINEX dataset.",
        ).grid(row=5, column=1, sticky="w", pady=(0, 4))

        ttk.Label(input_frame, text="Sampling interval (s)").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(input_frame, textvariable=self.sample_rate_var, width=16).grid(
            row=6, column=1, sticky="w", pady=4
        )

        ttk.Label(input_frame, text="PPP provider").grid(
            row=7, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            input_frame,
            textvariable=self.provider_var,
            values=PPP_PROVIDERS,
            state="readonly",
            width=14,
        ).grid(row=7, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="PPP series").grid(
            row=8, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            input_frame,
            textvariable=self.series_var,
            values=PPP_SERIES,
            state="readonly",
            width=14,
        ).grid(row=8, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="PPP project").grid(
            row=9, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            input_frame,
            textvariable=self.project_var,
            values=PPP_PROJECTS,
            state="readonly",
            width=14,
        ).grid(row=9, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="Execution mode").grid(
            row=10, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            input_frame,
            textvariable=self.execution_mode_var,
            values=EXECUTION_MODES,
            state="readonly",
            width=14,
        ).grid(row=10, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="Overwrite existing resampled files / outputs").grid(
            row=11, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            input_frame,
            textvariable=self.overwrite_var,
            values=YES_NO_OPTIONS,
            state="readonly",
            width=14,
        ).grid(row=11, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="Dataset limit").grid(
            row=12, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(input_frame, textvariable=self.dataset_limit_var, width=16).grid(
            row=12, column=1, sticky="w", pady=4
        )
        ttk.Label(input_frame, text="Use 'all' or a positive integer.").grid(
            row=12, column=1, sticky="w", padx=(140, 0), pady=4
        )

        ttk.Label(input_frame, text="Generate timeseries/report").grid(
            row=13, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            input_frame,
            textvariable=self.generate_timeseries_report_var,
            values=YES_NO_OPTIONS,
            state="readonly",
            width=14,
        ).grid(row=13, column=1, sticky="w", pady=4)

        ttk.Label(input_frame, text="Report plots").grid(
            row=14, column=0, sticky="nw", padx=(0, 8), pady=4
        )

        plot_frame = ttk.Frame(input_frame)
        plot_frame.grid(row=14, column=1, sticky="w", pady=4)

        ttk.Checkbutton(plot_frame, text="X", variable=self.plot_x_var).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(plot_frame, text="Y", variable=self.plot_y_var).grid(row=0, column=1, sticky="w", padx=(0, 16))
        ttk.Checkbutton(plot_frame, text="Z", variable=self.plot_z_var).grid(row=0, column=2, sticky="w", padx=(0, 16))
        ttk.Checkbutton(plot_frame, text="lon", variable=self.plot_lon_var).grid(row=0, column=3, sticky="w", padx=(0, 16))
        ttk.Checkbutton(plot_frame, text="lat", variable=self.plot_lat_var).grid(row=0, column=4, sticky="w", padx=(0, 16))
        ttk.Checkbutton(plot_frame, text="h", variable=self.plot_h_var).grid(row=0, column=5, sticky="w", padx=(0, 16))

        ttk.Checkbutton(plot_frame, text="E", variable=self.plot_e_var).grid(
            row=0, column=6, sticky="w", padx=(8, 0)
        )
        ttk.Checkbutton(plot_frame, text="N", variable=self.plot_n_var).grid(
            row=0, column=7, sticky="w", padx=(8, 0)
        )
        ttk.Checkbutton(plot_frame, text="U", variable=self.plot_u_var).grid(
            row=0, column=8, sticky="w", padx=(8, 0)
        )
        cleanup_frame = ttk.LabelFrame(root, text="Cleanup generated files", padding=10)
        cleanup_frame.pack(fill="x", pady=(10, 0))

        ttk.Checkbutton(
            cleanup_frame,
            text="Remove resampled RINEX",
            variable=self.cleanup_resampled_var,
        ).grid(row=0, column=0, sticky="w", padx=(0, 18), pady=3)

        ttk.Checkbutton(
            cleanup_frame,
            text="Remove generated YAML",
            variable=self.cleanup_yaml_var,
        ).grid(row=0, column=1, sticky="w", padx=(0, 18), pady=3)

        ttk.Checkbutton(
            cleanup_frame,
            text="Remove PEA run outputs",
            variable=self.cleanup_pea_outputs_var,
        ).grid(row=0, column=2, sticky="w", padx=(0, 18), pady=3)

        ttk.Checkbutton(
            cleanup_frame,
            text="Remove downloaded PPP products",
            variable=self.cleanup_downloaded_products_var,
        ).grid(row=0, column=3, sticky="w", padx=(0, 18), pady=3)

        cleanup_button_frame = ttk.Frame(cleanup_frame)
        cleanup_button_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.preview_cleanup_button = ttk.Button(
            cleanup_button_frame,
            text="Preview cleanup",
            command=self._preview_cleanup,
        )
        self.preview_cleanup_button.pack(side="left")

        self.run_cleanup_button = ttk.Button(
            cleanup_button_frame,
            text="Run cleanup after confirmation",
            command=self._run_cleanup_after_confirmation,
        )
        self.run_cleanup_button.pack(side="left", padx=(8, 0))

        button_frame = ttk.Frame(root)
        button_frame.pack(fill="x", pady=(10, 6))

        self.run_button = ttk.Button(
            button_frame,
            text="Run batch processing",
            command=self._start_batch,
        )
        self.run_button.pack(side="left")

        self.clear_button = ttk.Button(
            button_frame,
            text="Clear output",
            command=self._clear_output,
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop process",
            command=self._stop_process,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        ttk.Label(root, textvariable=self.status_var).pack(fill="x", pady=(0, 6))

        output_frame = ttk.LabelFrame(root, text="Output", padding=8)
        output_frame.pack(fill="both", expand=True)

        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text = tk.Text(output_frame, wrap="word", height=24)
        self.output_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scrollbar.set)

    def _browse_raw_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Select RAW RINEX root directory",
            initialdir=self.raw_root_var.get() or str(Path.home()),
        )

        if selected:
            self.raw_root_var.set(selected)

    def _selected_report_plot_columns(self) -> list[str]:
        pairs = [
            ("X", self.plot_x_var),
            ("Y", self.plot_y_var),
            ("Z", self.plot_z_var),
            ("lon", self.plot_lon_var),
            ("lat", self.plot_lat_var),
            ("h", self.plot_h_var),
            ("E", self.plot_e_var),
            ("N", self.plot_n_var),
            ("U", self.plot_u_var),
        ]

        selected = []
        for label, var in pairs:
            if var.get():
                selected.append(label)

        return selected
    def _validate_raw_root(self) -> tuple[bool, str]:
        raw_root = Path(self.raw_root_var.get()).expanduser()

        if not raw_root.exists() or not raw_root.is_dir():
            return False, f"RAW RINEX root directory does not exist or is not a directory:\n{raw_root}"

        return True, ""

    def _validate_inputs(self) -> tuple[bool, str]:
        ok, message = self._validate_raw_root()
        if not ok:
            return ok, message

        try:
            sample_rate = int(self.sample_rate_var.get().strip())
        except ValueError:
            return False, "Sampling interval must be a positive integer."

        if sample_rate <= 0:
            return False, "Sampling interval must be a positive integer."

        dataset_limit = self.dataset_limit_var.get().strip()

        if dataset_limit == "":
            return False, "Dataset limit must be 'all' or a positive integer."

        if dataset_limit.lower() != "all":
            try:
                n = int(dataset_limit)
            except ValueError:
                return False, "Dataset limit must be 'all' or a positive integer."

            if n <= 0:
                return False, "Dataset limit must be 'all' or a positive integer."

        if len(self._selected_report_plot_columns()) == 0:
            return False, "At least one report plot must be selected."


        static_products_root = self.static_products_root_var.get().strip()

        if not static_products_root:
            return False, "Static Ginan products directory must not be empty."

        static_products_path = Path(static_products_root).expanduser()

        if not static_products_path.exists() or not static_products_path.is_dir():
            return False, f"Static Ginan products directory does not exist:\n{static_products_path}"

        template_yaml_path = self.template_yaml_path_var.get().strip()

        if not template_yaml_path:
            return False, "YAML template path must not be empty."

        template_yaml = Path(template_yaml_path).expanduser()

        if not template_yaml.exists() or not template_yaml.is_file():
            return False, f"YAML template file does not exist:\n{template_yaml}"

        if not PYTHON_EXE.exists() or not PYTHON_EXE.is_file():
            return False, f"Python executable not found:\n{PYTHON_EXE}"

        if not BATCH_MAIN.exists() or not BATCH_MAIN.is_file():
            return False, f"batch_main.py not found:\n{BATCH_MAIN}"

        return True, ""


    def _browse_static_products_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Select static Ginan products directory",
            initialdir=self.static_products_root_var.get().strip() or str(Path.home()),
        )
        if selected:
            self.static_products_root_var.set(selected)

    def _browse_template_yaml_path(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select YAML template file",
            initialdir=str(Path(self.template_yaml_path_var.get()).expanduser().parent)
            if self.template_yaml_path_var.get().strip()
            else str(Path.home()),
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.template_yaml_path_var.set(selected)

    def _build_stdin_payload(self) -> str:
        dataset_limit = self.dataset_limit_var.get().strip()

        if dataset_limit.lower() == "all":
            batch_limit_input = "all"
        else:
            batch_limit_input = dataset_limit

        report_plot_columns = ",".join(self._selected_report_plot_columns())

        lines = [
            self.raw_root_var.get().strip(),
            self.static_products_root_var.get().strip(),
            self.template_yaml_path_var.get().strip(),
            self.sample_rate_var.get().strip(),
            self.provider_var.get().strip(),
            self.series_var.get().strip(),
            self.project_var.get().strip(),
            self.execution_mode_var.get().strip(),
            self.overwrite_var.get().strip(),
            batch_limit_input,
            self.generate_timeseries_report_var.get().strip(),
            report_plot_columns,
        ]

        return chr(10).join(lines) + chr(10)

    def _start_batch(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Process already running", "A batch process is already running.")
            return

        ok, message = self._validate_inputs()
        if not ok:
            messagebox.showerror("Invalid input", message)
            return

        stdin_payload = self._build_stdin_payload()

        self._append_output("\n=== Starting batch_main.py ===\n")
        self._append_output(f"Working directory: {PROJECT_DIR}\n")
        self._append_output(f"Command: {PYTHON_EXE} {BATCH_MAIN}\n")
        self._append_output(f"Report plots: {', '.join(self._selected_report_plot_columns())}\n\n")

        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Running batch processing...")

        thread = threading.Thread(
            target=self._run_subprocess,
            args=(stdin_payload,),
            daemon=True,
        )
        thread.start()

    def _run_subprocess(self, stdin_payload: str) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.process = subprocess.Popen(
                [str(PYTHON_EXE), str(BATCH_MAIN)],
                cwd=str(PROJECT_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )

            assert self.process.stdin is not None
            self.process.stdin.write(stdin_payload)
            self.process.stdin.close()

            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line)

            return_code = self.process.wait()
            self.output_queue.put(f"\n=== batch_main.py finished with return code {return_code} ===\n")
            self.output_queue.put("__PROCESS_FINISHED__")

        except Exception as exc:
            self.output_queue.put(f"\nERROR: {exc}\n")
            self.output_queue.put("__PROCESS_FINISHED__")

    def _terminate_process_group(self, proc: subprocess.Popen, sig: int) -> None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception as exc:
            self._append_output(f"\nWARNING: Could not signal process group: {exc}\n")
            try:
                if sig == signal.SIGTERM:
                    proc.terminate()
                elif sig == signal.SIGKILL:
                    proc.kill()
            except Exception as fallback_exc:
                self._append_output(f"\nWARNING: Fallback process signal failed: {fallback_exc}\n")

    def _kill_process_group_if_still_running(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            self._append_output("\n=== Process group still active; sending SIGKILL ===\n")
            self._terminate_process_group(proc, signal.SIGKILL)

    def _stop_process(self) -> None:
        proc = self.process

        if proc is not None and proc.poll() is None:
            self.status_var.set("Termination requested.")
            self._append_output("\n=== Termination requested by user ===\n")
            self._append_output("Sending SIGTERM to batch_main.py process group, including child pea processes.\n")

            self._terminate_process_group(proc, signal.SIGTERM)

            self.stop_button.configure(state="disabled")

            self.after(5000, lambda p=proc: self._kill_process_group_if_still_running(p))

    def _cleanup_kwargs(self) -> dict:
        return {
            "raw_root": Path(self.raw_root_var.get()).expanduser(),
            "remove_resampled_rinex": self.cleanup_resampled_var.get(),
            "remove_generated_yaml": self.cleanup_yaml_var.get(),
            "remove_pea_run_outputs": self.cleanup_pea_outputs_var.get(),
            "remove_downloaded_products": self.cleanup_downloaded_products_var.get(),
        }

    def _preview_cleanup(self) -> None:
        ok, message = self._validate_raw_root()
        if not ok:
            messagebox.showerror("Invalid input", message)
            return

        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Process running", "Cleanup preview is disabled while batch processing is running.")
            return

        self.status_var.set("Building cleanup preview...")
        self._append_output("\n=== Cleanup preview ===\n")

        thread = threading.Thread(
            target=self._run_cleanup_worker,
            args=(False,),
            daemon=True,
        )
        thread.start()

    def _run_cleanup_after_confirmation(self) -> None:
        ok, message = self._validate_raw_root()
        if not ok:
            messagebox.showerror("Invalid input", message)
            return

        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Process running", "Cleanup is disabled while batch processing is running.")
            return

        answer = messagebox.askyesno(
            "Confirm cleanup",
            "This will delete selected reproducible generated files.\n\n"
            "Preserved files include RAW RINEX inputs, source modules, templates/static resources, "
            "timeseries.out, timeseries.report, timeseries.report.html, and timeseries.html.\n\n"
            "Proceed with cleanup?",
        )

        if not answer:
            self._append_output("\n=== Cleanup cancelled by user ===\n")
            return

        self.status_var.set("Running cleanup...")
        self._append_output("\n=== Cleanup execution requested ===\n")

        thread = threading.Thread(
            target=self._run_cleanup_worker,
            args=(True,),
            daemon=True,
        )
        thread.start()

    def _run_cleanup_worker(self, execute: bool) -> None:
        try:
            importlib.invalidate_caches()

            if "cleanup_service" in sys.modules:
                cleanup_service = importlib.reload(sys.modules["cleanup_service"])
            else:
                cleanup_service = importlib.import_module("cleanup_service")

            buffer = io.StringIO()

            with contextlib.redirect_stdout(buffer):
                result = cleanup_service.clean_generated_files(
                    execute=execute,
                    **self._cleanup_kwargs(),
                )

            self.output_queue.put(buffer.getvalue())

            if execute:
                self.output_queue.put(f"\n=== Cleanup finished: {result['message']} ===\n")
            else:
                self.output_queue.put("\n=== Cleanup preview finished. No files were deleted. ===\n")

            self.output_queue.put("__CLEANUP_FINISHED__")

        except Exception as exc:
            self.output_queue.put(f"\nERROR during cleanup: {exc}\n")
            self.output_queue.put("__CLEANUP_FINISHED__")

    def _poll_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()

                if item == "__PROCESS_FINISHED__":
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set("Finished.")
                    self.process = None
                elif item == "__CLEANUP_FINISHED__":
                    self.status_var.set("Ready.")
                else:
                    self._append_output(item)

        except queue.Empty:
            pass

        self.after(100, self._poll_output_queue)

    def _append_output(self, text: str) -> None:
        self.output_text.insert("end", text)
        self.output_text.see("end")

    def _clear_output(self) -> None:
        self.output_text.delete("1.0", "end")


def main() -> None:
    app = PPPBatchGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
