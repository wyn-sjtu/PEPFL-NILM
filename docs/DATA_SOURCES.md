# Data sources and references

## Included data

`data/raw/` contains 26 prepared household CSVs from the `dataset/REDD` and `dataset/REFIT` directories of [Lightweight-Federated-NILM](https://github.com/hkuedl/Lightweight-Federated-NILM/tree/main/dataset). The files are included unchanged from the input snapshot used by this pipeline. Their SHA-256 hashes, headers, row counts, and task membership are recorded in [configs/data_scope.json](../configs/data_scope.json).

These files are the upstream project's prepared inputs, not the original REDD or REFIT distributions. The complete original-to-CSV conversion is not documented here; no additional assumptions are made about channel selection, resampling, gap filling, or appliance merging. This repository performs chronological splitting, training-only normalization, and window construction from the supplied rows.

## REDD

**Reference:** J. Zico Kolter and Matthew J. Johnson, "REDD: A Public Data Set for Energy Disaggregation Research," *SustKDD Workshop on Data Mining Applications in Sustainability*, 2011.

- [Original project](http://redd.csail.mit.edu/)
- [Author-hosted paper](https://zicokolter.com/publications/kolter2011redd.pdf)
- [Prepared CSV source](https://github.com/hkuedl/Lightweight-Federated-NILM/tree/main/dataset/REDD)

The included prepared REDD files retain their dataset attribution. An explicit redistribution license has not been identified in the available source records.

## REFIT

**Reference:** David Murray, Lina Stankovic, and Vladimir Stankovic, "An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study," *Scientific Data*, vol. 4, article 160122, 2017. [doi:10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122).

| Release | Official record | DOI |
|---|---|---|
| Raw measurements | [REFIT: Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/) | [10.15129/31da3ece-f902-4e95-a093-e0a9536983c4](https://doi.org/10.15129/31da3ece-f902-4e95-a093-e0a9536983c4) |
| Cleaned measurements | [REFIT: Electrical Load Measurements (Cleaned)](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) | [10.15129/9ab14b0e-19ac-4279-938f-27f643078cec](https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec) |

The official raw and cleaned releases are licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Credit belongs to David Murray, Lina Stankovic, and Vladimir Stankovic. The included CSVs are derived inputs distributed by the [upstream project](https://github.com/hkuedl/Lightweight-Federated-NILM/tree/main/dataset/REFIT); they should not be described as byte-identical copies of the official raw or cleaned archives.

## Prepared-data source

Yehui Li, Ruiyang Yao, Dalin Qin, and Yi Wang, "Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring," *IEEE Transactions on Smart Grid*, vol. 16, no. 2, pp. 1950-1961, 2025. [doi:10.1109/TSG.2024.3482363](https://doi.org/10.1109/TSG.2024.3482363).

[Author repository](https://github.com/hkuedl/Lightweight-Federated-NILM)

The dataset papers describe the measurements; the upstream paper and repository establish the prepared-input and experimental context. This repository implements a separate fixed-CNN, private-FiLM, adaptive-Huber pipeline.

## BibTeX

```bibtex
@inproceedings{kolter2011redd,
  author = {Kolter, J. Zico and Johnson, Matthew J.},
  title = {{REDD}: A Public Data Set for Energy Disaggregation Research},
  booktitle = {Proceedings of the SustKDD Workshop on Data Mining Applications in Sustainability},
  year = {2011},
  url = {https://zicokolter.com/publications/kolter2011redd.pdf}
}

@article{murray2017refit,
  author = {Murray, David and Stankovic, Lina and Stankovic, Vladimir},
  title = {An Electrical Load Measurements Dataset of {United Kingdom} Households from a Two-Year Longitudinal Study},
  journal = {Scientific Data},
  volume = {4},
  note = {Article 160122},
  year = {2017},
  doi = {10.1038/sdata.2016.122}
}

@article{li2025lightweight,
  author = {Li, Yehui and Yao, Ruiyang and Qin, Dalin and Wang, Yi},
  title = {Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring},
  journal = {IEEE Transactions on Smart Grid},
  volume = {16},
  number = {2},
  pages = {1950--1961},
  year = {2025},
  doi = {10.1109/TSG.2024.3482363}
}
```
