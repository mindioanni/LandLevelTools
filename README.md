# LandLevelTools

LandLevelTools is a collection of notebooks and Python workflows for land-level, GNSS, tide-gauge datum, vertical-reference, deformation, and geodetic time-series research.

The repository includes notebooks for tide-gauge datum network adjustment and geodetic documentation, GNSS RINEX download and processing, PPP and RTKLIB processing workflows, GNSS antenna-model analysis, EGMS / deformation time-series analysis, seismicity analysis, and auxiliary visualization and error-analysis workflows.

The repository currently contains both active workflows and older experimental or legacy notebooks. The recommended current workflows are listed first.

---

## Recommended current workflows

### `Adjust_Tide_Gauge_Datum_Network_V1.0.ipynb`

Current Jupyter notebook for tide-gauge datum network adjustment, visualization, and geodetic documentation.

The notebook supports RAW TPS and processed workbook workflows, normalizes observation tables, computes vertical components and height differences, performs network coverage and reciprocal-misclosure checks, applies weighted least-squares height adjustment, and generates automated geodetic datum documentation. It also includes optional 2D horizontal network adjustment, georeferencing of local adjusted coordinates, and plotting of selected points on map or imagery backgrounds.

This is the current maintained replacement for the obsolete Colab notebook `Adjust_Tide_Gauge_Datum_Network_V8_20260511.ipynb`, which has been removed from the repository.

### `GNSS_GEIN_NOA.ipynb`

Notebook for downloading and organizing GNSS RINEX data from the NOA GEIN GPSData archive service.

The workflow supports user-selected download paths, calendar date ranges, automatic decompression of downloaded `.gz` RINEX files, organization of unzipped RINEX files into daily dataset folders, and construction of a station coverage inventory from the NOA GEIN archive.

### `GINAN_PEA_PPP_Batch_Processor/`

Python-based modular workflow for automated Ginan/PEA PPP batch processing of daily or per-file GNSS RINEX datasets.

The processor performs system checks, RINEX discovery and header parsing, optional GFZRNX resampling, GNSS product resolution/download, deterministic YAML generation, PEA execution, output validation, POS-based convergence analysis, station coordinate time-series extraction, and HTML report generation. It supports both command-line and Tkinter GUI execution.

See:

```text
GINAN_PEA_PPP_Batch_Processor/README.md
```

### `RTKLIB_Baseline_Orchestrator/`

Python-based modular workflow for reproducible RTKLIB/rnx2rtkp short-baseline processing.

The workflow is designed for static GNSS baseline processing between a CORS/base station and one or more GNSS benchmarks near a tide gauge. It parses CORS PPP reports, discovers rover and base RINEX files, checks time overlap, resolves GNSS products, generates deterministic RTKLIB configuration files, executes rnx2rtkp, parses baseline solutions, performs QC, and builds processing reports. It supports both command-line and Tkinter GUI execution.

See:

```text
RTKLIB_Baseline_Orchestrator/README.md
```

---

## GNSS and geodetic-support notebooks

### `ANTEX_Comparisons_V1_20260219.ipynb`

Notebook for GNSS ANTEX antenna PCO/PCV analysis and visualization.

It parses ANTEX files, extracts antenna and frequency information, produces summary tables, plots PCO bar charts, plots NOAZI PCV curves versus zenith angle, creates 2D PCV polar heatmaps, and optionally compares two antennas. It also generates an HTML report and downloadable output archive.

### `GINAN_PEA_PPP_timeseries_post_processor_v1_20260203.ipynb`

Notebook for post-processing Ginan/PEA POS and TRACE outputs.

It discovers PPP sessions, attaches TRACE files, extracts embedded YAML processing strategies, performs strategy-consistency checks, parses POS files, generates statistics and plots, extracts optional TRACE-derived diagnostics, compares sessions, and finalizes an HTML report. It is useful for auditing and comparing PPP processing runs after Ginan/PEA execution.

### `Vertical_Component_error_V0_20260210.ipynb`

Notebook for vertical-component uncertainty or error-analysis workflows.

The current audit identified it as a standalone auxiliary notebook with limited embedded documentation. It is retained as a geodetic error-analysis notebook pending further review.

---

## Tide-gauge and mapping notebooks

### `NOA TG datum map V1.ipynb`

Notebook for mapping or visualizing NOA tide-gauge datum information.

The current audit found limited embedded metadata in this notebook. It is retained as an auxiliary visualization notebook until its workflow is documented or replaced.

---

## Deformation and remote-sensing notebooks

### `EGMS_shifts.ipynb`

Notebook for processing EGMS-related deformation or shift time series.

The workflow imports geospatial, outlier-detection, change-point, filtering, SQLite, and large-data tools. It is retained as an EGMS deformation-analysis notebook.

---

## Seismicity notebooks

### `Seismicity_V8_15032025.ipynb`

Newer seismicity-analysis notebook version retained as the preferred seismicity notebook.

### `Seismicity_V7_11022025.ipynb`

Older seismicity-analysis notebook version retained for reproducibility. It may be reviewed later and removed if all required functionality is confirmed to be superseded by `Seismicity_V8_15032025.ipynb`.

---

## Repository organization status

### Current active notebooks

- `Adjust_Tide_Gauge_Datum_Network_V1.0.ipynb`
- `GNSS_GEIN_NOA.ipynb`
- `ANTEX_Comparisons_V1_20260219.ipynb`
- `GINAN_PEA_PPP_timeseries_post_processor_v1_20260203.ipynb`
- `Seismicity_V8_15032025.ipynb`

### Current active Python workflow packages

- `GINAN_PEA_PPP_Batch_Processor/`
- `RTKLIB_Baseline_Orchestrator/`

### Auxiliary or legacy notebooks

- `EGMS_shifts.ipynb`
- `NOA TG datum map V1.ipynb`
- `Seismicity_V7_11022025.ipynb`
- `Vertical_Component_error_V0_20260210.ipynb`

### Removed obsolete notebook

- `Adjust_Tide_Gauge_Datum_Network_V8_20260511.ipynb`

This obsolete Colab version has been removed from the repository. The maintained current version is `Adjust_Tide_Gauge_Datum_Network_V1.0.ipynb`.

---

## Suggested cleanup plan

No additional files should be deleted automatically. A safe cleanup strategy is:

1. Keep `Adjust_Tide_Gauge_Datum_Network_V1.0.ipynb` as the current tide-gauge datum network notebook.
2. Keep `GNSS_GEIN_NOA.ipynb` as the NOA GEIN GNSS download and inventory notebook.
3. Keep `GINAN_PEA_PPP_Batch_Processor/` and `RTKLIB_Baseline_Orchestrator/` as documented operational packages.
4. Review `Seismicity_V7_11022025.ipynb` against `Seismicity_V8_15032025.ipynb`; remove V7 only if fully superseded.
5. Review `NOA TG datum map V1.ipynb` and `Vertical_Component_error_V0_20260210.ipynb` for documentation completeness.
6. Consider future folder organization:
   - `notebooks/`
   - `legacy_notebooks/`
   - `workflows/`
   - `docs/`
   - `examples/`

At the current stage, the repository remains mostly flat to preserve existing notebook links and avoid breaking current usage.
