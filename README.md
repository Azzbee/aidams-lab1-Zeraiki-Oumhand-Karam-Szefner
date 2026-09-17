# AIDAMS Lab 1: steel plants, geospatial analysis and exposure

Course: Research and Emerging Topics in Data Science: Climate Risks (AIDAMS S5, IDSI 51003).

| # | Full name | Student ID |
|---|-----------|------------|
| 1 | Adam Zeraiki | B00819260 |
| 2 | Lina Oumhand | B00820945 |
| 3 | Chris William Karam | B00825083 |
| 4 | Liam Szefner | B00822121 |

**Repository:** [https://github.com/Azzbee/aidams-lab1-Zeraiki-Oumhand-Karam-Szefner](https://github.com/Azzbee/aidams-lab1-Zeraiki-Oumhand-Karam-Szefner)

**Submitter:** Adam Zeraiki

**Streamlit Cloud app (bonus):** [https://aidams-lab1-zeraiki-oumhand-karam-szefner.streamlit.app/](https://aidams-lab1-zeraiki-oumhand-karam-szefner.streamlit.app/)

## What is in this repository

| Path | Content |
|---|---|
| `lab_1.ipynb` | The lab, fully executed. Every figure is stored as an interactive Plotly output and as a PNG copy, so maps are visible on GitHub too. |
| `app.py` | Streamlit dashboard (Part 6): filters, KPIs, plant / density / exposure / company maps, EDA charts, data table. |
| `data/` | Processed tables written by the notebook and read by the dashboard (CSV and Parquet twins) plus `metadata.json`. |
| `Plant-level_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx` | Raw GEM plant-level file used by the notebook. |
| `requirements.txt` | Dashboard dependencies (used by Streamlit Cloud). |
| `requirements-notebook.txt` | Extra packages needed to re-run the notebook. |

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app only reads the files in `data/`, so it works from a fresh clone without the raw datasets.

## Re-run the notebook

1. `pip install -r requirements-notebook.txt`
2. Put the GEM plant-level workbook next to the notebook (already in the repo). The loader finds the workbook by filename pattern and maps known column aliases to the handout names.
3. Put `LitPop_pc_300_arcsec_{CHN,IND,JPN}_v1.hdf5` from Moodle in `litpop/` or `data/`. Raw HDF5 files are excluded from Git.
4. Run all cells. Part 6 rewrites `data/` and then starts the dashboard headlessly as a self-test.

Static PNG copies of the figures need `kaleido` and Chrome. Without them the notebook still runs and keeps the interactive outputs.

## Method in short

- **Capacity.** GEM reports capacity per plant and status on a separate sheet. "Capacity" here means nominal crude steel capacity with status *operating* or *operating pre-retirement* (2,229 Mtpa). Summing every row would give 3,672 Mtpa by mixing planned, idle, retired and cancelled capacity.
- **Missing values.** GEM writes `unknown` and `>0` for gaps and `N/A` for "not applicable". The workbook is read with `keep_default_na=False` so the two are not merged.
- **Coordinates.** Parsed from the single `Coordinates` text column, range-checked, then compared with the other plants of the same country. One longitude sign error is corrected and logged (Hyundai Steel Louisiana plant).
- **Exposure.** LitPop sample for China, India and Japan (300 arc-seconds, reference year 2018, produced capital in USD). Each plant gets its nearest grid cell through a haversine ball tree, accepted within one cell diagonal (13.1 km) and only inside those three countries, plus the sum of all cells within 25 km. 612 plants are matched, 60% of world operating capacity.
- **Companies.** Aggregation by GEM's immediate `Owner`. Unknown owners are excluded; missing or inconsistent parent stakes are not inferred. Representative location: the company's largest plant (the capacity-weighted spherical centroid is computed as well).

## Main findings

1. China holds 35% of plants and 49% of operating capacity. Hebei province alone holds about 10%.
2. India's pipeline is 2.4 times its operating capacity, against 0.1 times for China.
3. Ownership is fragmented: 1,068 immediate owners, 90% with a single plant. The top 10 hold 13% of capacity.
4. Steel plants sit in cells around the 95th percentile of national asset density, but plant size is only weakly related to surrounding assets (rank correlation 0.12).
5. Most companies own one plant in one country, so their exposure summary usually reflects a single plant. Exposure alone does not measure climate risk.

## Data sources and licence

- Global Energy Monitor, *Global Iron and Steel Tracker*, June 2026 (V1) release. Creative Commons Attribution 4.0. https://globalenergymonitor.org/projects/global-iron-steel-tracker
- LitPop: Eberenz, S., Stocker, D., Röösli, T., Bresch, D. N. (2020). Asset exposure data for global physical risk assessment. *Earth System Science Data*, 12, 817-833. https://doi.org/10.5194/essd-12-817-2020. Data archive: https://doi.org/10.3929/ethz-b-000331316. Course sample provided on Moodle.
