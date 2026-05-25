# GitHub setup commands

Target repository:

```text
https://github.com/Asgari-research/mof-chemical-holdouts
```

## If the repository already exists and you have access

```bash
git clone https://github.com/Asgari-research/mof-chemical-holdouts.git
cd mof-chemical-holdouts

# copy the contents of this kit into the cloned folder
# then:
git status
git add README.md LICENSE CITATION.cff .gitignore requirements.txt environment.yml split_strategy_pipeline_raw_arc.py docs data source_data figures manuscript .github
git commit -m "Initialize MOF chemical holdouts reproducibility package"
git push origin main
```

## If the repository is empty and cloning does not work

```bash
mkdir mof-chemical-holdouts
cd mof-chemical-holdouts
# copy the kit contents here

git init
git branch -M main
git add .
git commit -m "Initialize MOF chemical holdouts reproducibility package"
git remote add origin https://github.com/Asgari-research/mof-chemical-holdouts.git
git push -u origin main
```

## Recommended second commit after adding source-data tables

After copying compact source data from `All_Tables_Main_SI.zip`:

```bash
git add source_data docs/SOURCE_DATA_MANIFEST.md
git commit -m "Add manuscript and SI source-data tables"
git push
```

## Do not commit database files

Before every commit, run:

```bash
git status
```

Make sure these files are not staged:

- `clean_data.csv`
- `geo-clusters.csv`
- `mc-clusters.csv`
- `func-clusters.csv`
- `flig-clusters.csv`
- `geometric_properties.csv`
- `all_topology_lists.csv`
- `paper1_split_strategy_outputs_*`
- full prediction CSVs
- model `.joblib` files
