# Method

## Shared CNN and private FiLM

Each household-appliance pair uses one sequence-to-sequence predictor with 576-point input and output windows:

```text
Aggregate window
  -> [Conv1D(kernel 7, padding 3, 32 channels) -> ReLU -> FiLM] x 5
  -> adaptive average pooling to 72 points
  -> flatten -> dense(128) -> ReLU -> FiLM
  -> linear(576)
```

FiLM applies a channel-wise affine transformation `h = gamma * z + beta`. Gains start at one and offsets at zero. Each household learns its own coefficients, which remain private. The shared parameter set contains the convolutional and dense layers; aggregation excludes every FiLM parameter.

The model predicts normalized appliance power. Training reconstructs predictions and targets in watts using the household's training-only target mean and scale. Inference additionally clips negative predictions to zero.

## Data preparation

Let `N` be the number of rows in a household CSV and `L = 576`.

1. The outer training boundary is `outer = floor(0.8 * N)`. The remaining rows form the test portion.
2. Development uses `inner = floor(0.9 * floor((outer - 1) / L)) * L` as the training-prefix boundary. The following training suffix is reserved for validation.
3. Aggregate and target standardizers fit the current raw training prefix: `[0, inner)` for development and `[0, outer)` for formal training. Validation and test values do not contribute to their statistics.
4. Source-row order is preserved. Windows containing nonfinite aggregate or target values are excluded without deleting rows or compressing time. Within each outer segment, valid window end positions are enumerated and every 576th valid end is retained. The final possible end is omitted to retain the original window convention.
5. Development assigns retained training windows to fit or validation according to the internal boundary and excludes crossing windows. Formal training uses all retained outer-training windows. No window crosses the train-test boundary.

Formal training starts from a fresh seed-1 initialization. Development model weights and optimizer states are not transferred. File membership and household order follow `configs/data_scope.json`.

## Adaptive Huber objective

For watt-domain residual `e = prediction_w - target_w`, the objective is

```text
H_delta(e) = 0.5 * e^2                     if |e| <= delta
             delta * (|e| - 0.5 * delta)  otherwise

loss = mean(H_delta(e)) / target_scale^2
```

The threshold range is obtained from the 75th percentile of training-window target powers strictly above 50 W. A 50 W fallback is used when no such values exist. This statistic uses at most 65,536 deterministically spaced active-power ranks:

```text
lower = max(1 W, 0.1 * q75_on)
upper = 10 * lower
delta_initial = upper
```

For each minibatch, the detached absolute residuals provide a new 75th-percentile estimate, using at most 2,048 deterministically spaced entries. After a successful optimizer update:

```text
delta_next = clip(0.95 * delta + 0.05 * q75_residual, lower, upper)
```

The updated threshold is used by the next minibatch. An optimizer update skipped by the AMP gradient scaler does not advance the threshold. Loss and threshold computations use FP32. Threshold state remains local throughout federated training and final adaptation.

## Training schedule

Training uses seed 1, Adam with learning rate `3e-4`, and batches of 64 windows.

For each of 50 federated rounds, every participating household receives the shared CNN weights and makes one pass over its training windows, updating both CNN and FiLM parameters. The server then averages CNN weights with equal weight for each household. Private parameters and optimizer states persist across broadcasts.

After the final broadcast, the CNN is frozen and each household makes 30 more local passes updating only FiLM. The Huber threshold continues to adapt. These passes require no further communication. Adam moments for frozen shared parameters are removed; private optimizer state is retained.

CUDA training uses mixed precision and activation checkpointing. Model weights remain FP32. FiLM-only adaptation still propagates gradients through the CNN to earlier adaptation layers, so adapter storage alone does not describe peak training memory.

## Evaluation

Predictions are converted to watts and clipped at zero. The pipeline does not fit an output calibrator. MAE and activation F1 are calculated per household and then macro-averaged within each dataset-appliance task. Both predictions and targets use the same strict activation threshold, `power > 50 W`; F1 is zero when its denominator is zero.

Development evaluates only the internal validation portion. Formal evaluation uses the held-out outer test portion. The setting evaluates future data from participating households, rather than unseen households.

## References

- McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html), AISTATS, 2017.
- Perez et al., [FiLM: Visual Reasoning with a General Conditioning Layer](https://ojs.aaai.org/index.php/AAAI/article/view/11671), AAAI, 2018.
- Huber, [Robust Estimation of a Location Parameter](https://doi.org/10.1214/aoms/1177703732), 1964.
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), ICLR, 2018.
- Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), 2016.

Dataset and prepared-input references are listed in [DATA_SOURCES.md](DATA_SOURCES.md).
