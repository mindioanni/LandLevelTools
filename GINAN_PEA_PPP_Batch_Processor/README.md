# GINAN/pea PPP Batch Processor

**User-facing program name:** GINAN/pea PPP Batch Processor  
**Internal project/module name:** `ppp_batch_orchestrator`  
**Version:** MVP v1.1  
**Date:** 2026-05-01  

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

The program was developed to support tide-gauge-related GNSS benchmark processing in a structured, reproducible and auditable way. It reads archived raw RINEX datasets, extracts the required metadata from the RINEX header, optionally resamples the observations with GFZRNX, resolves and downloads the required precise and broadcast GNSS products, generates deterministic Ginan/PEA YAML configuration files from a selected template, executes PEA, validates the generated output products, and builds a station coordinate time series from the successful daily/per-file PPP solutions.

The current implementation supports both command-line execution and a Tkinter-based graphical interface.

## Main Capabilities

The current MVP implementation supports the following processing chain:

1. System preflight checks.
2. Discovery of raw daily/per-file RINEX dataset folders.
3. RINEX observation file identification.
4. RINEX header parsing.
5. Extraction of station metadata, receiver/antenna metadata, approximate XYZ coordinates, antenna H/E/N offsets and observation time span.
6. Optional RINEX resampling to a user-defined sampling interval using GFZRNX.
7. Derivation of covered dates and day-of-year product identifiers.
8. Identification, download and validation of required GNSS products.
9. Deterministic YAML generation for each dataset from a user-selected YAML template.
10. Execution of Ginan/PEA PPP processing.
11. Validation of generated `.POS`, `.GPX` and `.TRACE` outputs.
12. POS-based convergence analysis.
13. Extraction of one daily/per-file station coordinate solution from each successful run.
14. Aggregation of successful daily/per-file solutions into `timeseries.out`.
15. Generation of an HTML processing report, `timeseries.report`.
16. Warning-aware reporting when one or more datasets fail but at least one successful solution is available.

## Processing Model

The processor follows a daily/per-file single-station PPP strategy.

Each raw RINEX dataset is treated as one independent processing unit. For each dataset, the software generates a dedicated YAML configuration file and runs Ginan/PEA independently. The resulting POS time series is then reduced to one representative coordinate solution after convergence detection. The successful daily/per-file solutions are combined into a station coordinate time series.

The processing model is:

```text
raw daily/per-file RINEX dataset
-> optional GFZRNX resampling
-> effective time-window determination
-> PPP product resolution/download
-> deterministic YAML generation
-> Ginan/PEA execution
-> POS-based convergence analysis
-> daily/per-file solution extraction
-> timeseries.out
-> timeseries.report
```

## User Inputs

The GUI and command-line workflow request the following main inputs.

### RAW RINEX root directory

Root directory containing the raw daily/per-file RINEX dataset folders.

Each subdirectory is treated as one processing dataset. For example:

```text
RAW/
├── 2025_141/
│   └── observation_file.rnx
├── 2025_142/
│   └── observation_file.rnx
└── 2025_143/
    └── observation_file.rnx
```

The exact dataset folder naming convention is flexible, provided each dataset folder contains one identifiable RINEX observation file.

### Static products directory

Directory containing static auxiliary products used by PEA during processing, such as grids, loading models, antenna models and tables.

Typical contents may include files or subdirectories such as:

```text
igs20.atx
finals.data.iau2000.txt
tables/
```

This path is passed into the generated YAML as the relevant products/static inputs root, depending on the selected template structure.

### YAML template path

Path to the Ginan/PEA YAML template used to generate a dedicated YAML configuration file for each RINEX dataset.

The selected template is not processed directly. Instead, `yaml_builder.py` reads the template and modifies only the dataset-specific fields, such as:

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

### PPP provider, series and project

The user selects the GNSS product family to be used, for example:

```text
Provider: COD
Series:   FIN
Project:  MGX or OPS
```

These selections are passed to the product download/resolution workflow.

### Execution mode

Two execution modes are supported:

```text
run
build_only
```

- `run`: builds YAML files and executes PEA.
- `build_only`: builds YAML files but skips PEA execution.

### Overwrite option

Controls whether existing generated files may be overwritten when possible.

### Dataset limit

Allows the user to process all datasets or only the first `N` datasets. This is useful for smoke tests.

### Timeseries/report generation

If enabled, the processor builds:

```text
timeseries.out
timeseries.report
```

after processing.

If all datasets succeed, the report is generated normally.

If one or more datasets fail but successful solutions are available, the report is still generated and includes warnings about failed datasets.

### Report plot columns

The user may request plots for:

```text
X, Y, Z, lon, lat, h
```

or all available supported coordinate components.

## Repository Contents

The reference project folder is:

```text
~/GINAN_PEA_PPP_Batch_Processor
```

The current workflow is organized into standalone Python modules.

### Core Python modules

- `paths_config.py`  
  Defines default system paths, user-input defaults, processing options and validation rules.

- `rinex_header.py`  
  Discovers raw datasets, identifies RINEX observation files, parses RINEX headers, supports compact RINEX/header reading where applicable, and derives covered dates/day codes.

- `resample_rinex.py`  
  Plans and executes optional RINEX resampling using GFZRNX.

- `products_download.py`  
  Defines product download plans, resolves product filenames, executes downloader calls, and validates product availability.

- `yaml_builder.py`  
  Builds output paths and generates deterministic YAML files from the selected Ginan/PEA template.

- `pea_runner.py`  
  Builds and executes the Ginan/PEA command and checks for early-stop conditions.

- `results_check.py`  
  Collects run outputs, validates generated files, and summarizes `.POS` solution coverage.

- `position_timeseries.py`  
  Reads successful PEA/POS outputs, performs convergence analysis, extracts daily/per-file coordinate solutions, computes common ENU coordinates, and writes `timeseries.out`.

- `timeseries_report.py`  
  Generates the HTML `timeseries.report`, including processing metadata, convergence settings, QC flags, daily/per-file solution summary, final coordinate statistics, plots and warnings for failed datasets.

- `system_check.py`  
  Performs preflight checks for executables, required files, directory structure and writability.

- `batch_main.py`  
  Top-level orchestrator that connects all modules into a complete batch-processing workflow.

- `ppp_batch_gui.py`  
  Tkinter graphical user interface for launching batch processing.

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

## Output Structure

For each dataset, the processor creates a dedicated run directory inside the relevant `GINAN_process` folder.

A typical run directory contains files such as:

```text
<station>_default_config_HEAD_<YYYY><DDD><HH>.POS
<station>_default_config_HEAD_<YYYY><DDD><HH>.GPX
default_config_HEAD_Network_<YYYY><DDD><HH>.TRACE
stdout_<run_label>.txt
```

The top-level `GINAN_process` folder contains:

```text
yaml/
timeseries.out
timeseries.report
```

where:

- `yaml/` contains generated per-dataset YAML files.
- `timeseries.out` contains the combined coordinate solution time series.
- `timeseries.report` is the HTML processing report.

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

The generated YAML follows the product-loading logic compatible with Ginan-UI style product discovery, while using the batch processor’s own resolved/downloaded products directory.

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

## Report Generation

`timeseries_report.py` generates `timeseries.report` in HTML format.

The report includes:

1. Processing strategy.
2. PPP setup and paths.
3. Convergence detection method.
4. Smoothed versus unsmoothed POS usage.
5. Daily/per-file position solution definition.
6. ENU series reference.
7. QC flags.
8. Column definitions of `timeseries.out`.
9. Summary of daily/per-file solutions.
10. Final GNSS station coordinate solution from daily/per-file PPP solutions.
11. Plots requested by user.

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

If one or more datasets failed, the report includes a non-successful dataset summary and marks the corresponding epochs in the plots when the timing information is available.

## Failure Handling

The processor distinguishes between:

```text
SUCCESS
BUILT
FAILED
```

If all datasets succeed, `timeseries.out` and `timeseries.report` are generated normally.

If one or more datasets fail but successful run outputs are available, `timeseries.out` and `timeseries.report` are still generated from the successful solutions. In that case, the batch summary reports that timeseries/report generation completed with warnings.

This behavior avoids losing a valid coordinate time series because of a small number of failed product downloads or failed daily runs.

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

## Intended Use

The primary intended use is reproducible PPP processing of GNSS benchmark observations associated with tide-gauge datum control, sea-level monitoring and related geodetic applications.

The workflow is designed to support:

- controlled processing of archived RINEX datasets,
- repeatable Ginan/PEA configuration generation,
- traceable product selection,
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
- default static products directory,
- default YAML template path,
- desktop launcher paths, if used.

The GUI allows the user to override the main processing paths at runtime, but the default values are still defined in the project configuration and GUI source.
