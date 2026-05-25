# Data folder

This folder documents expected data placement but does not contain ARC--MOF raw data.

Required local files for running the pipeline:

- `clean_data.csv`
- `geo-clusters.csv`
- `mc-clusters.csv`
- `func-clusters.csv`
- `flig-clusters.csv`

Optional local files:

- `geometric_properties.csv`
- `all_topology_lists.csv`

By default, the current pipeline expects these files next to `split_strategy_pipeline_raw_arc.py`. You may either place local copies beside the script or edit `DATA_DIR` in the script to point to this `data/` folder.

Do not commit the raw or prepared ARC--MOF data files.
