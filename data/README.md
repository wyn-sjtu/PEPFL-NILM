# Input data

The `raw/` directory includes the prepared household CSVs used by the pipeline:

```text
raw/
  REDD/
    House_1.csv
    House_2.csv
    ...
  REFIT/
    CLEAN_House1.csv
    CLEAN_House2.csv
    ...
```

Pass `--data-root data/raw` when running commands from the repository root. No extraction or conversion step is required.

Each task reads the `Aggregate` column and its appliance target column:

| CLI appliance | Dataset | Target column |
|---|---|---|
| `Fridge` | REDD / REFIT | `Fridge` |
| `DishWasher` | REDD / REFIT | `Dish` |
| `Microwave` | REDD / REFIT | `Microwave` |
| `Washer` | REDD | `Washer` |
| `WashingMachine` | REFIT | `Washing` |
| `Kettle` | REFIT | `Kettle` |

The 26 files support nine tasks and 94 household-appliance combinations. The exact household membership, source headers, row counts, and SHA-256 hashes are specified in [configs/data_scope.json](../configs/data_scope.json). Input verification is enabled by default.

Keep source-row order and values unchanged. The loader splits each household chronologically, fits scalers on training data, and excludes invalid or boundary-crossing windows. Do not standardize the complete series, shuffle rows, remove missing rows, merge channels, or concatenate households before loading.

These are prepared inputs from the upstream research project, not the original REDD or REFIT distributions. Dataset references, attribution, and source links are provided in [DATA_SOURCES.md](../docs/DATA_SOURCES.md).
