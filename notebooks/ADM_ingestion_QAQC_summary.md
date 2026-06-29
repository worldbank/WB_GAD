# ADM Ingestion QA/QC Summary

Date: 2026-06-29
Source notebook: `notebooks/ADM_ingestion_QAQC.ipynb`
QA/QC output folder: `C:\WBG\Work\data\ADMIN\QAQC`

## Overall status

The QA/QC run produced issue logs and issue layers for duplicate IDs, duplicate names, and topology checks.
The workflow appears partially complete for publication exports (region dissolves and supplemental CSVs exist; core ADM publication files were not found in the current `FOR_PUBLICATION` folder snapshot).

## Key findings

### 1) Primary key / duplicate ID checks

- ADM1 duplicate IDs (`ADM1CD_c`): 4 records flagged
  - Artifact: `adm1_duplicates_ADM1CD_c.gpkg`
- ADM2 duplicate IDs (`ADM2CD_c`): 9 records flagged
  - Artifact: `adm2_duplicates_ADM2CD_c.gpkg`

### 2) Data completeness

From completeness logs generated at 2026-06-29 09:46:

- ADM0 (`n = 251`)
  - `WB_A3`: 2 missing (99.20% complete)
  - `CONTINENT`: 1 missing (99.60% complete)
  - `WB_REGION`: 31 missing (87.65% complete)
  - `WB_INCOME`: 32 missing (87.25% complete)
  - No missing values reported for `ISO_A3`, `ISO_A2`, `WB_STATUS`, `SOV_ISO_A3`, `SOV_NAME`, `NAM_0`, `geometry`, `ADM0CD_c`
- ADM1 (`n = 3182`)
  - `WB_A3`: 28 missing (99.12% complete)
  - `ADM1CD`: 2635 missing (17.19% complete)
  - `ADM1CD_t`: 547 missing (82.81% complete)
  - No missing values reported for `ISO_A3`, `ISO_A2`, `WB_REGION`, `WB_STATUS`, `SOV_ISO_A3`, `SOV_NAME`, `NAM_0`, `NAM_1`, `geometry`, `ADM1CD_c`
- ADM2 (`n = 40953`)
  - `NAM_2`: 213 missing (99.48% complete)
  - No missing values reported for `ISO_A3`, `WB_STATUS`, `NAM_0`, `NAM_1`, `ADM1CD_c`, `ADM2CD_c`, `WB_REGION`

Note: `adm0_missing_iso_a3.gpkg` was not found, which is consistent with zero missing `ISO_A3` in ADM0.

### 3) Duplicate name checks

- ADM1 duplicate name groups (`NAM_1` within `ISO_A3`): 2 groups
  - Countries/groups flagged: `AIA`, `GHA`
- ADM2 duplicate name groups (`NAM_2` within `ADM1CD_c`): 17 groups
  - Groups flagged include: `CHN002`, `CHN003`, `CHN027`, `GGY007`, `GGY009`, `GGY012`, `GGY013`, `IDN033`, `IDN037`, `MEX020`, `TWN001`, `USA021`, `USA026`, `USA047`, `VEN010`, `VEN018`, `ZWE006`

### 4) Topology checks

From `adm_topology_issues.gpkg`:

- Total topology issue features: 336
- By issue type:
  - `ADM2_overlaps`: 288
  - `ADM2_slivers`: 47
  - `ADM1_slivers`: 1
- Countries with highest issue counts (top 10):
  - `VNM` 45
  - `BRA` 32
  - `USA` 28
  - `MEX` 28
  - `IDN` 26
  - `AUS` 17
  - `COL` 10
  - `IND` 8
  - `PAN` 7
  - `MOZ` 6

## Output inventory

### QA/QC artifacts (`C:\WBG\Work\data\ADMIN\QAQC`)

- `adm_topology_issues.gpkg`
- `adm0_data_completeness.log`
- `adm1_data_completeness.log`
- `adm2_data_completeness.log`
- `adm1_duplicates_ADM1CD_c.gpkg`
- `adm2_duplicates_ADM2CD_c.gpkg`
- `ADM1_name_duplicates.log`
- `ADM2_name_duplicates.log`

### Publication folder snapshot (`C:\WBG\Work\data\ADMIN\NEW_WB_BOUNDS\FOR_PUBLICATION`)

Present at check time:

- `WB_GAD_adm1_additional_columns.csv`
- `WB_GAD_adm2_additional_columns.csv`
- `REGIONS/WB_GAD_CONTINENT.gpkg`
- `REGIONS/WB_GAD_WB_REGION.gpkg`

Core publication layers (for example, `WB_GAD_ADM0`, `WB_GAD_ADM1`, `WB_GAD_ADM2`, `WB_GAD_ADM0_complete`, `WB_GAD_ocean_mask`) were not observed in this folder snapshot.

## Recommended follow-up

1. Resolve duplicate ID records in ADM1 and ADM2 using the duplicate GeoPackages.
2. Review high-volume ADM2 topology issues first (`ADM2_overlaps`) for `VNM`, `BRA`, `USA`, `MEX`, and `IDN`.
3. Confirm whether ADM0 regional classification gaps (`WB_REGION`, `WB_INCOME`) are expected for disputed/non-standard entities.
4. Re-run final export step and verify all expected publication layers are written to `FOR_PUBLICATION`.
