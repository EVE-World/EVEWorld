# Training

EVEWorld fine-tunes GigaWorld-0 on DreamGen GR1 and AgiBot with the joint objective below.

## Recipes

| Setting | GigaWorld-0 (DreamGenBench / AgiBot) |
|---|---|
| Initialization | Video-Pretrain-2B checkpoint. |
| Trainable | Full transformer. |
| Optimizer | CAME-8bit. |
| Learning rate | `4.32e-5` (`2^-14.5`). |
| Weight decay | `0.01`. |
| Micro batch | 8. |
| Gradient accumulation | 8. |
| Effective batch | 64. |
| Steps | 250 on DreamGenBench, 50 on AgiBot. |
| Resolution | `480 x 768` / `480 x 640`. |
| Clip | 93 frames at 16 FPS. |
| TIA block | 23. |
| `lambda_TIA` | 0.5, warmed up over 20 steps. |
| Noise gate | `sigma in [0.2, 0.5]`. |

The released configurations put these values under the same keys; see
the `eveworld.yaml` files under `configs/paper/`
([`gigaworld/dreamgen`](../configs/paper/gigaworld/dreamgen/eveworld.yaml) and
[`gigaworld/agibot`](../configs/paper/gigaworld/agibot/eveworld.yaml)).
The Standard SFT baselines use the same budget with a uniform loss: 250 steps
at effective batch 64 on DreamGenBench and 50 steps on AgiBot.

`method.kind` selects SFT, IGR only, TIA only or EVEWorld. SFT and IGR-only
runs do not attach a TIA adapter; TIA-only and EVEWorld attach it in both
training and inference.

## Splits

Every configuration names its manifest under `data.split`. The GigaWorld-0 DreamGen arms read
[`data/splits/dreamgen/train.txt`](../data/splits/dreamgen/train.txt), the 92
post-training clips; the 12 clips of
[`data/splits/dreamgen/val.txt`](../data/splits/dreamgen/val.txt) form the development set; the 126 prompts of
[`data/splits/dreamgen/test.txt`](../data/splits/dreamgen/test.txt) are generated
but never trained on. The AgiBot arm reads the 777 clips of
`data/splits/agibot/train.txt`, which the preparation script has already filtered
against the 21 EWMBench test episodes.

## Objective

The joint objective adds the TIA consistency term to the IGR restoration term:

```
L = L_IGR + lambda_TIA * L_TIA
```

For a sample that carries an IGR corruption, `L_IGR` restores the clean latent
from the count-edited input; for a clean sample it reduces to the base
reconstruction objective. `L_TIA` is computed on the post-transport features,
with the weight and the optional noise gate set per backbone as in the table
above.

### IGR term

```
L_IGR = E[ lambda(sigma) * (1 / (C*T*H*W)) *
           || W_hat^(1/2) ⊙ (D_theta(z_tilde + sigma*eps, c, sigma) - z) ||^2_2 ]
```

`D_theta` is the denoising transformer, `z` the clean VAE latent of the
demonstration and `z_tilde` the latent of the count-edited video produced by
pasting the target patch into a sampled region. `W_hat` is the spatial weight
map normalized to unit mean, `sigma` is the sampled noise level and
`lambda(sigma)` is the EDM weighting `(sigma^2 + sigma_data^2) / (sigma *
sigma_data)^2` with `sigma_data = 0.5`. The weight map assigns 3.0 to the
disturbed interaction region and 1.0 elsewhere; it is built per clip as
described in [`data_preparation.md`](data_preparation.md) and loaded from the
precomputed cache when one exists.

A sample is count-edited with probability `method.p_dup` (0.5 in the released
configurations). Clips whose target could not be tracked have the gate
disabled, in which case the sample keeps the clean video and a unit weight map.
The paste itself is sampled by the corruption builder, whose defaults are:

| Setting | Default |
|---|---|
| Paste-zone bias, B / A / background | `(0.4, 0.2, 0.4)`. |
| Retry budget per patch | `60` attempts. |
| Minimum duration of the paste window | `3` latent frames. |
| Patch scale | `U(0.8, 1.2)`. |
| Patch alpha | `U(0.4, 1.0)`. |
| Feather / blur the patch | probability `0.3`. |

### TIA term

TIA attaches at a single transformer block and refines the target features of
each frame with those transported from the previous frame:

```
H'_t = H_t + gamma * tanh(W_out(H_bar_t - W_in H_t))
```

`H_bar_t` is the transported feature field, `W_in` a shared down-projection,
`W_out` a bias-free output projection initialized to zero, and `gamma` the
residual scale (0.1). The first frame is left unchanged. Locally, each cell of
frame `t-1` queries the `7 x 7` window centered on the same coordinate in frame
`t`, and the cosine similarities become softmax transport weights at
temperature `tau = 0.07`, with a projection dimension of 64.

The alignment loss pulls the transported features towards the true target
location and away from the other candidates in the window:

```
L_TIA = -(1 / |V|) * sum_t log( exp(s_{t,u_t} / tau) /
                                sum_{v in C_t} exp(s_{t,v} / tau) )

s_{t,v} = f_bar_{t-1}(u_{t-1})^T f_bar_t(v)
```

`V` contains the adjacent-frame pairs whose target is visible in both frames
with detector confidence above the threshold, `u_t` is the target location,
`C_t` the candidate window and `f_bar` the normalized feature after transport.
The target locations and the GroundingDINO detections are training-time
supervision only: at inference the matcher and the transport run on the model's
own hidden features.

### Weight schedule

On GigaWorld-0 the TIA term is not applied at full weight from the first step:
`lambda_TIA` warms up linearly from 0 to 0.5 over the first 20 steps, and the
noise gate restricts the term to sampled noise levels `sigma in [0.2, 0.5]`,
where the transport is informative but the input is not yet clean. The values are
`method.warmup_steps`, `method.sigma_low`, `method.sigma_high` and
`method.gated`; the files
[`ablations/tia/warmup.yaml`](../configs/ablations/tia/warmup.yaml) and
[`ablations/tia/gate.yaml`](../configs/ablations/tia/gate.yaml) expose
the two axes.

## Layer selection

The TIA block is not tuned with gradients: an offline probe scans the candidate
blocks and keeps the one whose local matches land closest to the annotated
target. For every candidate block, frame `t-1`'s target feature retrieves its
best local match inside the `7 x 7` window of frame `t`, and the probe averages
the distance to the annotated target location over the frame pairs in `V`:

```
u_hat_t = arg max_{v in C_t} f_{t-1}(u_{t-1})^T f_t(v)
l_star  = arg min_l (1 / |V|) sum_t || u_hat_t - u_t ||_2
```

Blocks 8 to 26 are swept; the probe selects block 23 for GigaWorld-0 and
AgiBot. The sweep is configured by
[`ablations/tia/layer_sweep.yaml`](../configs/ablations/tia/layer_sweep.yaml),
which reads the same annotations the corruption builder uses. The probe
artifacts are written under `experiments/analysis/tia_layer_probe/`.

TIA itself is implemented in
[`../src/eveworld/methods/tia`](../src/eveworld/methods/tia), the corruption
builder and the restoration loss in
[`../src/eveworld/methods/igr`](../src/eveworld/methods/igr).

## Commands

The entry point reads one YAML config and accepts overrides. Run it
from the repository root:

```bash
python scripts/train/train_gigaworld.py \
    --config configs/paper/gigaworld/dreamgen/eveworld.yaml

python scripts/train/train_gigaworld.py \
    --config configs/paper/gigaworld/agibot/eveworld.yaml
```

The controls and the ablation arms use the same entry point with a different
config:

```bash
python scripts/train/train_gigaworld.py --config configs/paper/gigaworld/dreamgen/sft.yaml
python scripts/train/train_gigaworld.py --config configs/paper/gigaworld/dreamgen/igr.yaml
python scripts/train/train_gigaworld.py --config configs/paper/gigaworld/dreamgen/tia.yaml
python scripts/train/train_gigaworld.py --config configs/ablations/igr/wmap_binary3x.yaml
```

| Flag | Meaning |
|---|---|
| `--config` | YAML file under `configs/`; required. |
| `--steps` | Override `train.max_steps`. |
| `--seed` | Override the run seed. |
| `--output-dir` | Override `output_dir`; defaults to the value in the config. |
| `--resume` | Continue from the last checkpoint in the output directory. |
| `--dry-run` | Run a tiny synthetic batch on CPU and exit; no data or GPU needed. |

Always check a config with the dry run before starting a real job, since it
validates the method wiring, the layer index and the loss composition without
reading the data roots:

```bash
python scripts/train/train_gigaworld.py \
    --config configs/paper/gigaworld/dreamgen/eveworld.yaml --dry-run
```

A run writes `<output-dir>/checkpoint-<step:06d>.pt` every
`train.save_every` steps (50 by default) together
with `latest.json` pointing to the newest checkpoint. The run directory is
set by `output_dir`; the naming and the per-row checkpoint
selection are described in [`checkpoints.md`](checkpoints.md). Resuming
with `--resume` restarts from the newest checkpoint in the run directory.

Sampling and generation settings are deliberately not part of training; they
live in the `inference` block of the same config and are used by
[`inference.md`](inference.md).
