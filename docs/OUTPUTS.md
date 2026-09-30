# Commands and output formats

## Code map

| File | Purpose |
|---|---|
| `src/nilm_f/data.py` | CSV loading, chronological splits, training-only scaling, and window construction |
| `src/nilm_f/model.py` | Shared 1D CNN and private FiLM layers |
| `src/nilm_f/loss.py` | Adaptive Huber loss and local threshold state |
| `src/nilm_f/training.py` | Federated averaging, local adaptation, and checkpoint resumption |
| `src/nilm_f/evaluation.py` | Evaluation and optional prediction export |
| `src/nilm_f/memory.py` | Analytical memory estimates |
| `src/nilm_f/cli.py` | Command-line interface |
| `configs/` | Training configuration and data manifest |
| `src/nilm_f/resources/` | Configuration copies bundled with the installed package |

## Preparation

`prepare` writes `preprocessing.json`, containing input hashes, split boundaries, scaler statistics, retained window support, and training-only loss statistics. It does not train a model or save a duplicate normalized dataset. `train` performs the same preparation automatically.

## Training

Each training directory contains:

| File | Contents |
|---|---|
| `latest.pt` | Checkpoint after a complete federated round or local adaptation pass, including model, Adam, AMP, threshold, and random-generator states |
| `models.pt` | Final household models and training-only scalers |
| `training_config.json` | Run configuration |
| `data_manifest.json` | Input and preprocessing identity |
| `history.json` | Per-client losses, update counts, communication, timing, and GPU memory measurements |
| `training_result.json` | Model size, stage costs, and completion summary |
| `status.json` | Run progress |

Both checkpoint formats are accepted by `evaluate`. The `--resume` option continues a matching run from `latest.pt`; an interrupted partial round or pass is restarted from the last complete checkpoint. Dataset, configuration, and implementation identity must remain unchanged. Load PyTorch checkpoints only from trusted sources.

## Evaluation

`evaluate` writes `metrics.json`. Adding `--save-predictions` exports one NPZ file per household with these arrays:

- `prediction_w`: nonnegative predictions in watts.
- `target_w`: appliance targets in watts.
- `aggregate_w`: aggregate input power in watts.
- `raw_window_start`: starting CSV row of each window.

Power arrays have shape `[number_of_windows, 576]`. Row indices are not timestamps, and retained windows may be separated by missing-data gaps.

`run-all` writes `summary.json` and `summary.csv` with separate rows for development and formal evaluation. Development uses the internal validation portion; formal evaluation uses the outer test portion.

## Time and memory

The method performs no architecture search. Training time includes serial household processing, transfers, averaging, and broadcasts. Data preparation, evaluation, and checkpoint I/O are recorded separately. Add development and formal training times when reporting the complete computation budget.

CUDA allocated and reserved peaks are recorded separately for federated training and FiLM-only adaptation. The analytical memory estimate counts model weights, two Adam moments, and logical FP16 forward outputs. It excludes gradients, CUDA workspaces, and allocator effects; measured GPU peaks are reported separately.

Run directories, checkpoints, metrics, and prediction files are generated locally and are excluded from version control.
