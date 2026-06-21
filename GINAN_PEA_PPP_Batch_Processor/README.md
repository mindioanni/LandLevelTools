# GINAN/pea PPP Batch Processor

**User-facing program name:** GINAN/pea PPP Batch Processor
**Internal project/module name:** `ppp_batch_orchestrator`
**Version:** MVP v1.4-RINEX2/Product-Staging
**Baseline:** C baseline — allGNSS / fallback / flexible-tree reference version with SD/VCD report diagnostics; RINEX 2 / Compact RINEX / Hatanaka compatibility and product staging stabilization
**Date:** 2026-06-21

## Authors and AI-assisted development

### Primary author

**Ioannis Mintourakis**
Postdoctoral Researcher
Hellenic National Tsunami Warning Centre
Institute of Geodynamics
National Observatory of Athens, Greece
i.mintourakis@noa.gr
sealevelresearch@gmail.com

### AI-assisted development

**R**

“R” is the working name used by the author for OpenAI’s ChatGPT, inspired by Isaac Asimov’s Robot series.

In this project, OpenAI's model GPT-5.5 Thinking was used as an AI assistant for code development, debugging, documentation drafting and workflow design.

## Overview

GINAN/pea PPP Batch Processor is a Python-based modular workflow for automated batch processing of daily or per-file GNSS RINEX observation datasets using Ginan/PEA Precise Point Positioning.

The program was developed to support tide-gauge-related GNSS benchmark and CORS processing in a structured, reproducible and auditable way. It reads archived raw RINEX datasets, extracts the required metadata from the RINEX header, optionally resamples the observations with GFZRNX, resolves or reuses the required precise and broadcast GNSS products, generates deterministic Ginan/PEA YAML configuration files from a selected template, executes PEA, validates the generated output products, and builds a station coordinate time series from the successful daily/per-file PPP solutions.

The current implementation supports both command-line execution and a Tkinter-based graphical interface, including full processing, build-only, report-only and report-from-timeseries workflows.

The current reference state of the code is the **C baseline**. In this project, C baseline means the updated version of the processor that supports allGNSS processing, fallback product handling and more flexible input tree layouts.

Older diagnostic or historical states, such as `oldGitHub`, `V3_fully working`, A-static tests, MIX tests and temporary forensic runtime archives, are not the active reference version.

## Main Capabilities

The current MVP v1.3-SD/VCD implementation supports the following processing and reporting chain:

1. System preflight checks.
2. Discovery of raw daily/per-file RINEX datasets.
3. Flexible discovery of RINEX files in different folder-tree layouts.
4. RINEX observation file identification.
5. RINEX header parsing.
6. Extraction of station metadata, receiver/antenna metadata, approximate XYZ coordinates, antenna H/E/N offsets and observation time span.
7. Optional RINEX resampling to a user-defined sampling interval using GFZRNX.
8. Effective time-window determination from the resampled RINEX file.
9. Derivation of covered dates and day-of-year product identifiers.
10. Identification, download, reuse and validation of required GNSS products.
11. Fallback handling for legacy precise product availability.
12. Deterministic YAML generation for each dataset from a user-selected YAML template.
13. Execution of Ginan/PEA PPP processing.
14. Validation of generated `.POS`, `.GPX` and `.TRACE` outputs.
15. POS-based convergence analysis.
16. Extraction of one daily/per-file station coordinate solution from each successful run.
17. Aggregation of successful daily/per-file solutions into `timeseries.out`.
18. Generation of an HTML processing and diagnostic report, `timeseries.report.html`.
19. Warning-aware reporting when one or more datasets fail but at least one successful solution is available.
20. `report_only` regeneration from existing POS outputs.
21. `report_from_timeseries` regeneration from an existing `timeseries.out`.
22. Shift Detector (SD) diagnostics for position-change detection and shift clustering.
23. Velocity Change Detector (VCD) diagnostics for persistent horizontal velocity-change detection.
24. Automatic pre-event / event-incidence / post-event transient-window reporting.
25. ENU composite full-span plot with stable linear fits and quadratic transient fits.


## Processing Model

The processor follows a daily/per-file single-station PPP strategy.

Each raw RINEX dataset is treated as one independent processing unit. For each dataset, the software generates a dedicated YAML configuration file and runs Ginan/PEA independently. The resulting POS time series is then reduced to one representative coordinate solution after convergence detection. The successful daily/per-file solutions are combined into a station coordinate time series.

The full processing model is:

```text
raw daily/per-file RINEX dataset
-> optional GFZRNX resampling
-> effective time-window determination
-> PPP product resolution / reuse / download
-> fallback product handling where applicable
-> deterministic YAML generation
-> Ginan/PEA execution
-> PEA output validation
-> POS-based convergence analysis
-> daily/per-file solution extraction
-> timeseries.out
-> timeseries.report.html
```

Additional report regeneration modes are available:

```text
existing POS outputs -> report_only -> timeseries.out -> timeseries.report.html
existing timeseries.out -> report_from_timeseries -> timeseries.report.html
```

The `report_only` mode requires existing per-dataset POS outputs. The `report_from_timeseries` mode requires only the top-level `GINAN_process/timeseries.out` file.

## User Inputs

The GUI and command-line workflow request the following main inputs.

### RAW RINEX root directory

Root directory containing the raw daily/per-file RINEX datasets.

The C baseline supports more flexible input layouts than the original MVP v1.1 version. RINEX files may be located in dataset subfolders or, where supported by the discovery logic, in alternative station-specific folder trees.

A simple example is:

```text
RAW/
├── 2025_141/
│   └── observation_file.rnx
├── 2025_142/
│   └── observation_file.rnx
└── 2025_143/
    └── observation_file.rnx
```

Other station archives may follow different layouts. The current C baseline includes updated discovery logic intended to support such cases.

### Products directory

Directory containing or receiving the precise and auxiliary products required by PEA.

Depending on the station layout and selected processing setup, this may include files or subdirectories such as:

```text
IGS_PRECISE/
igs20.atx
finals.data.iau2000.txt
igs_satellite_metadata.snx
tables/
```

The generated YAML uses the resolved products root and the product-loading structure expected by the selected Ginan/PEA template.

### YAML template path

Path to the Ginan/PEA YAML template used to generate a dedicated YAML configuration file for each RINEX dataset.

The selected template is not processed directly by PEA. Instead, `yaml_builder.py` reads the template and modifies only the dataset-specific fields, such as:

- observation input root,
- RINEX input filename,
- output directory,
- start and end epoch,
- epoch interval,
- receiver and antenna metadata,
- approximate station coordinates,
- antenna eccentricity,
- products root,
- dynamic GNSS products.

### Sampling interval

Requested output sampling interval in seconds.

If the raw RINEX sampling interval differs from the requested value, the workflow can resample the observation file using GFZRNX.

The effective first and last epochs of the resampled RINEX file are used for YAML generation.

### PPP provider, series and project

The user selects the GNSS product family to be used, for example:

```text
Provider: COD
Series:   FIN
Project:  MGX or OPS
```

These selections are passed to the product download/resolution workflow.

The C baseline includes fallback handling for cases where the preferred precise products are not available in the expected form.

### Execution mode

Four execution modes are supported:

```text
run
build_only
report_only
report_from_timeseries
```

- `run`: executes the full workflow from RINEX discovery to PEA execution, `timeseries.out` generation and `timeseries.report.html` generation.
- `build_only`: builds products/YAML/run folders but skips PEA execution.
- `report_only`: rebuilds `timeseries.out` from existing POS outputs and then regenerates `timeseries.report.html`.
- `report_from_timeseries`: regenerates only `timeseries.report.html` from an existing `GINAN_process/timeseries.out`.

For `report_from_timeseries`, the required preserved file is:

```text
GINAN_process/timeseries.out
```

For `report_only`, the per-dataset POS outputs must still exist.

### Overwrite option

Controls whether existing generated files may be overwritten when possible.

### Dataset limit

Allows the user to process all datasets or only the first `N` datasets. This is useful for smoke tests.

### Timeseries/report generation

If enabled, the processor builds:

```text
timeseries.out
timeseries.report.html
```

after processing.

If all datasets succeed, the report is generated normally.

If one or more datasets fail but successful solutions are available, the report is still generated and includes warnings about failed datasets.

The HTML report can also be regenerated later through:

```text
report_only
report_from_timeseries
```

`report_only` rebuilds `timeseries.out` from preserved POS outputs. `report_from_timeseries` rebuilds only the HTML report from an already preserved `timeseries.out`.

### Report plot columns

The user may request plots for:

```text
X, Y, Z, lon, lat, h
```

or all available supported coordinate components, including ENU components where available.

## Repository Contents

The reference project folder is:

```text
/home/ioannis/ppp_batch_orchestrator
```

The current workflow is organized into standalone Python modules.

### Core Python modules

- `paths_config.py`
  Defines default system paths, user-input defaults, processing options and validation rules.

- `rinex_header.py`
  Discovers raw datasets, identifies RINEX observation files, parses RINEX headers, supports compact RINEX/header reading where applicable, supports updated discovery logic for non-standard station trees, and derives covered dates/day codes.

- `resample_rinex.py`
  Plans and executes optional RINEX resampling using GFZRNX. The effective resampled observation window is used for subsequent YAML generation.

- `products_download.py`
  Defines product download plans, resolves product filenames, executes downloader calls, validates product availability, supports product reuse and fallback handling for legacy product cases.

- `yaml_builder.py`
  Builds output paths and generates deterministic YAML files from the selected Ginan/PEA template.

- `pea_runner.py`
  Builds and executes the Ginan/PEA command and checks for early-stop conditions.

- `results_check.py`
  Collects run outputs, validates generated files, and summarizes `.POS` solution coverage.

- `position_timeseries.py`
  Reads successful PEA/POS outputs, performs convergence analysis, extracts daily/per-file coordinate solutions, computes common ENU coordinates, and writes `timeseries.out`.

- `timeseries_change_detection.py`
  Implements Shift Detector (SD) utilities for identifying statistically significant position changes, representative shift epochs and shift clusters in the PPP time series.

- `timeseries_velocity_detection.py`
  Implements Velocity Change Detector (VCD) utilities for rolling/sliding velocity estimation, persistent horizontal velocity-change detection and automatic transient-window support.

- `timeseries_report.py`
  Generates the HTML `timeseries.report.html`, including processing metadata, convergence settings, QC flags, daily/per-file solution summary, final coordinate statistics, plots, Shift Detector diagnostics, Velocity Change Detector diagnostics, transient windows and warnings for failed datasets.

- `system_check.py`
  Performs preflight checks for executables, required files, directory structure and writability.

- `batch_main.py`
  Top-level orchestrator that connects all modules into a complete batch-processing workflow.

- `ppp_batch_gui.py`
  Tkinter graphical user interface for launching batch processing and configuring report diagnostics.

- `cleanup_service.py`
  Helper module for cleanup-related operations, including preservation of the top-level `timeseries.out` and `timeseries.report.html` files.

### GUI / desktop launcher resources

The GUI may be launched directly from Python or through a desktop launcher.

The user-facing application name is:

```text
GINAN/pea PPP Batch Processor
```

The launcher may use a dedicated application icon, for example:

```text
GINAN_batch_processor.png
```

### Development / testing folder

- `dev_notebooks/`
  Intended location for notebooks used for development, debugging, smoke tests and integration tests of the processor.

### Local archive folder

- `ARCHIVED_STATES/`
  Local folder used to keep development backups, diagnostic snapshots, debug reports, cleanup quarantine material and historical code states.

This folder is not required for normal execution and should normally not be committed to GitHub.

## Output Structure

For each dataset, the processor creates a dedicated run directory inside the relevant `GINAN_process` folder.

A typical run directory contains files such as:

```text
<station>_default_config_HEAD_<YYYY><DDD><HH>.POS
<station>_default_config_HEAD_<YYYY><DDD><HH>.GPX
default_config_HEAD_Network_<YYYY><DDD><HH>.TRACE
stdout_<run_label>.txt
```

The station processing root may also contain generated folders such as:

```text
RESAMPLED/
yaml/
IGS_PRECISE/
GINAN_process/
```

The top-level `GINAN_process` folder contains the report/time-series products:

```text
timeseries.out
timeseries.report.html
```

where:

- `timeseries.out` contains the combined coordinate solution time series.
- `timeseries.report.html` is the HTML processing and diagnostic report.

Depending on the selected station tree, generated folders such as `RESAMPLED/`, `yaml/` and `GINAN_process/` may be created under the station RAW root or under a station-specific processing root.

The top-level report/time-series files are deliberately kept outside the per-dataset PEA run directories so that `report_from_timeseries` can be executed without retaining all large per-dataset PEA run folders.

## Run Labels

Run labels are generated from the dataset name and effective sampling interval.

The current convention is:

```text
<dataset_name>_<sampling_interval>s
```

For example:

```text
2025_141_30s
ast1017w00_30s
sant0020_30s
```

## YAML Generation Logic

The YAML generation module does not create a configuration file from scratch.

Instead, it reads the user-selected YAML template and modifies only the fields required for the current dataset and processing setup.

Typical updated fields include:

```yaml
inputs:
    inputs_root: <resolved_products_root>

    snx_files:
        - igs_satellite_metadata.snx
        - tables/sat_yaw_bias_rate.snx
        - tables/qzss_yaw_modes.snx
        - tables/bds_yaw_modes.snx
        - IGS0OPSSNX_<YYYYDDD>0000_01D_01D_CRD.SNX

    satellite_data:
        nav_files:
            - BRDC*
        clk_files:
            - '*.CLK'
        bsx_files:
            - '*.BIA'
        sp3_files:
            - '*.SP3'

    gnss_observations:
        gnss_observations_root: <resampled_dataset_dir>
        rnx_inputs:
            - <resampled_rinex_filename>
```

The generated YAML follows the product-loading logic compatible with Ginan-UI style product discovery, while using the batch processor’s own resolved, reused or downloaded products directory.

## Products and Fallback Handling

The C baseline includes improved handling of precise products and legacy product cases.

The processor may:

- reuse an existing local products folder,
- check whether required products already exist,
- call the configured downloader when products are missing,
- apply fallback handling where the preferred precise product family is not available,
- validate that the expected product files exist before PEA execution,
- preserve product-related diagnostics in the run context.

The intended goal is to make processing more robust across station archives and historical periods where product naming conventions, project identifiers or available product families may differ.

## Static Models and ANTEX Version

The PPP solution depends on the precise products and static model files used by Ginan/PEA. These may include:

- precise orbit files,
- precise clock files,
- bias files,
- SINEX files,
- broadcast navigation files,
- `igs20.atx`,
- `finals.data.iau2000.txt`,
- satellite metadata,
- yaw tables,
- loading models,
- atmospheric model grids,
- other Ginan tables.

For reproducible scientific processing, the exact product set should be preserved or documented.

At minimum, important static/model files should be recorded with:

- file name,
- full path or product root,
- file size,
- modification date,
- SHA256 checksum.

The ANTEX file `igs20.atx` is especially important. Controlled tests showed that changing `igs20.atx` can shift the ellipsoidal height solution at the millimetre level. Therefore, the `igs20.atx` version used in a production solution should be frozen and documented.

## Timeseries Generation

`position_timeseries.py` reads the successful run directories and generates `timeseries.out`.

The primary daily/per-file solution is based on the smoothed POS solution after convergence. Convergence detection uses the unsmoothed POS time series and a robust internal reference derived from the final part of the solution.

The resulting `timeseries.out` includes, among other fields:

```text
run_label
dataset_name
station_id
time_mean_all_epochs_utc
convergence_epoch_utc
convergence_delay_sec
X_m
Y_m
Z_m
lon_deg
lat_deg
h_m
E_m
N_m
U_m
qc_flags
```

The `timeseries.out` file is the minimum required input for `report_from_timeseries`.

## Report Generation

`timeseries_report.py` generates `timeseries.report.html` in HTML format.

The report includes:

1. Processing strategy.
2. PPP setup and paths.
3. Metadata sources.
4. Convergence detection method.
5. Smoothed versus unsmoothed POS usage.
6. Daily/per-file position solution definition.
7. ENU series reference.
8. QC flags.
9. Column definitions of `timeseries.out`.
10. Summary of daily/per-file solutions.
11. Final GNSS station coordinate solution from daily/per-file PPP solutions.
12. User-selected plots.
13. Shift Detector (SD) diagnostics.
14. Velocity Change Detector (VCD) diagnostics.
15. Automatic pre-event / event-incidence / post-event transient windows.
16. ENU composite full-span plot with shift intervals, representative epochs, stable linear fits and transient quadratic fits.
17. Zoomed diagnostic plots around detected shift/meta-cluster intervals.

The final coordinate solution section includes:

- final mean value,
- standard deviation,
- minimum value,
- maximum value,
- range,
- linear trend,
- equivalent velocity in mm/year,
- equivalent velocity uncertainty,
- number of daily/per-file solutions.

The plot section uses line plots without point markers and displays the time axis as decimal years.

### Shift Detector (SD)

The Shift Detector identifies position changes in the ENU PPP time series by comparing representative positions before and after candidate epochs. Candidate shifts are filtered using:

- minimum absolute jump,
- minimum jump significance,
- robust MAD-based noise floor,
- shift clustering window,
- report-grade jump threshold.

The resulting `shift cluster interval` is a geodetic interval detected in the PPP time series. It should not be interpreted directly as rupture duration. The `representative shift epoch` is a representative epoch of the detected shift cluster and should not be interpreted directly as earthquake origin time.

### Velocity Change Detector (VCD)

The Velocity Change Detector estimates rolling/sliding velocities and detects persistent changes in the horizontal velocity field. The main report-grade interpretation is based on coherent horizontal E/N velocity behaviour, not on isolated single-component changes.

The VCD supports automatic identification of:

- pre-event transient windows,
- event/incidence intervals,
- post-event transient windows,
- shift-related report-grade horizontal velocity-change clusters,
- diagnostic vertical-only changes.

The event/incidence interval is not treated as a stable velocity interval. It is reported through net displacement and deformation rate during incidence.

### ENU composite transient/stable overlays

The ENU composite full-span plot uses:

- green solid lines for stable linear fits outside transient/event intervals,
- thin orange dashed lines for quadratic transient fits inside pre-event and post-event transient windows,
- no fitted velocity model inside the event/incidence interval.

For the horizontal components, the transient interpretation is based on joint E/N quadratic fitting over common transient windows. The U component is treated as diagnostic-only.

Velocity labels indicate:

- `v`: stable linear velocity,
- `v_mid`: instantaneous velocity at the middle of a quadratic transient interval.

If one or more datasets failed, the report includes a non-successful dataset summary and marks the corresponding epochs in the plots when the timing information is available.

## Failure Handling

The processor distinguishes between:

```text
SUCCESS
BUILT
FAILED
```

If all datasets succeed, `timeseries.out` and `timeseries.report.html` are generated normally.

If one or more datasets fail but successful run outputs are available, `timeseries.out` and `timeseries.report.html` are still generated from the successful solutions. In that case, the batch summary reports that timeseries/report generation completed with warnings.

This behavior avoids losing a valid coordinate time series because of a small number of failed product downloads or failed daily runs.

For report-only workflows:

- `report_only` requires existing POS outputs.
- `report_from_timeseries` requires only `GINAN_process/timeseries.out`.

Trace files are mainly useful for debugging and audit of the PEA execution. They are not required for `report_from_timeseries`, and are not normally required for `report_only` unless future QC extensions explicitly parse them.

## External Dependencies

The processor assumes that the following external tools are installed and accessible:

- Ginan/PEA
- GFZRNX
- `auto_download_PPP.py`

The orchestrator and the downloader do not necessarily need to use the same Python interpreter.

In the reference setup, the main processor may run from an Anaconda Python environment, while `auto_download_PPP.py` may run from a dedicated downloader virtual environment.

The downloader environment must include the required Python packages, including:

- `click`
- `numpy`
- `requests`
- `hatanaka`
- `gnssanalysis`
- `matplotlib`

A dedicated downloader environment can be prepared once on the target system. For example:

```bash
cd /path/to/ginan_batch_PPP
python3 -m venv ginanenv
source ginanenv/bin/activate
pip install --upgrade pip
pip install click numpy requests Hatanaka gnssanalysis matplotlib
```

## Downloader Notes

The downloader script used by the reference setup is:

```text
auto_download_PPP.py
```

Important implementation note: the downloader must pass the selected `project_type` also to BIA product downloads.

This is necessary so that, for example, a COD / FIN / MGX processing setup resolves BIA files as:

```text
COD0MGXFIN_YYYYDDD0000_01D_01D_OSB.BIA
```

and not incorrectly as:

```text
COD0OPSFIN_YYYYDDD0000_01D_01D_OSB.BIA
```

The processor explicitly passes the selected provider, series and project to product download commands where applicable.

When the downloader is called from a notebook-driven workflow, the subprocess environment forces:

```text
MPLBACKEND=Agg
```

This avoids failures caused by notebook-specific matplotlib backends when `gnssanalysis` imports matplotlib inside the downloader environment.

## Repository Hygiene

The Git repository should contain source code, documentation, launcher resources and lightweight reference files only.

The following should normally not be committed to GitHub:

- `ARCHIVED_STATES/`
- `RESAMPLED/`
- `GINAN_process/`
- `IGS_PRECISE/`
- generated `yaml/` folders,
- downloaded precise products,
- runtime archives,
- temporary PEA outputs,
- large `.POS`, `.TRACE`, `.GPX` products,
- Python cache folders,
- notebook checkpoints,
- local backup files,
- local forensic test folders.

A `.gitignore` file should be used before pushing this project to GitHub.

For preserving report reproducibility with minimal storage, keep:

```text
GINAN_process/timeseries.out
GINAN_process/timeseries.report.html
```

This is sufficient for `report_from_timeseries`.

To rebuild `timeseries.out` from original per-dataset solutions, also keep the per-dataset POS outputs. This is required for `report_only`.

The cleanup workflow is designed to preserve top-level report/time-series files while removing large generated intermediate products when requested.

## Limitations

The current MVP remains a controlled research-processing tool.

Known limitations:

- It assumes a valid Ginan/PEA installation.
- It assumes a compatible YAML template structure.
- It assumes that the selected downloader script can retrieve the required products.
- It currently processes datasets sequentially.
- The final coordinate time series is built from successful daily/per-file runs only.
- Failed datasets are reported but not automatically repaired.
- Long-term geodetic velocities inferred from short time spans should be treated only as diagnostic trends.
- Full reproducibility requires preservation or documentation of the exact precise product set and static model files, including `igs20.atx`.
- SD/VCD results are geodetic diagnostics of the PPP time series; they are not a seismic source model.
- A shift cluster interval is not a rupture duration.
- A representative shift epoch is not necessarily an earthquake origin time.
- Pre-event and post-event transient windows should not be interpreted as definitive preseismic or postseismic deformation without comparison to independent GNSS, InSAR, seismic and geological evidence.
- Quadratic transient fits are empirical curvature diagnostics and do not estimate a physical relaxation time.

## Intended Use

The primary intended use is reproducible PPP processing of GNSS benchmark observations associated with tide-gauge datum control, sea-level monitoring and related geodetic applications.

The workflow is designed to support:

- controlled processing of archived RINEX datasets,
- repeatable Ginan/PEA configuration generation,
- traceable product selection,
- fallback handling for legacy precise product availability,
- daily/per-file PPP solution validation,
- coordinate time-series generation,
- technical reporting for GNSS station/tide-gauge documentation.

## Notes for Porting to Another System

If the workflow is moved to another system, review and update:

- `paths_config.py`,
- Ginan/PEA executable path,
- GFZRNX executable path,
- downloader Python interpreter,
- downloader script path,
- default RAW root path,
- default products directory,
- default YAML template path,
- desktop launcher paths, if used.

The GUI allows the user to override the main processing paths at runtime, but the default values are still defined in the project configuration and GUI source.

## Current Operational Status

Current operational baseline:

```text
C baseline = allGNSS / fallback / flexible-tree PPP batch orchestrator with SD/VCD report diagnostics
```

The current synchronized state includes:

- full `run` and `build_only` processing modes,
- `report_only` mode from preserved POS outputs,
- `report_from_timeseries` mode from preserved `timeseries.out`,
- Shift Detector (SD) report diagnostics,
- Velocity Change Detector (VCD) report diagnostics,
- automatic pre-event / event-incidence / post-event transient windows,
- ENU composite plot overlays with green stable linear fits and thin orange dashed quadratic transient fits.



## Version v1.4-RINEX2/Product-Staging stabilization notes

This version keeps the v1.3 SD/VCD reporting baseline and adds the stabilization changes required for RINEX 2 / Compact RINEX processing with Ginan/PEA:

- Flexible RINEX discovery for dataset folders and file-based station trees.
- Compact RINEX / Hatanaka header handling for RINEX2 `.??d` files.
- Fallback day-code derivation when daily Compact RINEX files omit `TIME OF LAST OBS`; this fallback is used only for product/date coverage, not as a claim that the final observation epoch is known from the header.
- Deterministic RINEX3 observation-label normalization after GFZRNX conversion where generic `L1/L2/D1/D2/S1/S2` labels must be mapped to explicit PEA-compatible signal labels.
- Dataset-specific `PRODUCT_STAGING/<dataset>` folders for isolated precise/static product inputs used by each generated YAML.
- Required static-product staging now includes `tables/fes2014b_Cnm-Snm.dat` when the template requests the FES2014b ocean tide potential file.
- The experimental user-provided ANTEX receiver-block logic is not part of this stabilized release; `igs20.atx` is treated as a frozen static model input.

