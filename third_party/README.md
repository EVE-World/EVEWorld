# Third-Party Backbones

Clone GigaWorld-0 into the gitignored `third_party/giga-world-0/` directory:

```bash
bash scripts/setup/clone_gigaworld.sh
```

The script accepts a destination argument and a `--ref` override. Use `--no-deps`
to skip installing the upstream training and dataset packages.

## GigaWorld-0

- Upstream: <https://github.com/open-gigaai/giga-world-0>
- Role: the main backbone. DreamGenBench, WorldArena 1.0, EWMBench (AgiBot) and
  PBench all fine-tune and evaluate GigaWorld-0 Video-Pretrain-2B.
- Training also imports the two upstream framework packages, which are not on
  PyPI:

  ```bash
  pip install git+https://github.com/open-gigaai/giga-train.git
  pip install git+https://github.com/open-gigaai/giga-datasets.git
  ```

- The backbone is Apache-2.0; see the `LICENSE` inside the checkout.

## Other external tools

| Tool | Used for |
|---|---|
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | target detection for IGR annotation and for MLR counting |
| [SAM2](https://github.com/facebookresearch/sam2) | instance tracking and the robot-occluder masks behind the MLR occlusion check |
| [WorldArena 1.0](https://github.com/WorldArena-Official/WorldArena) | the official WorldArena metric harness |
| [DreamGen](https://github.com/NVIDIA/DreamGen) | the DreamGen GR1 split and its judging protocol |
| [EWMBench](https://github.com/AgibotTech/EWMBench) | the AgiBot metric suite |

Their weights and config paths are read from the environment; copy
[`.env.example`](../.env.example) to `.env` and fill in what you have.
`scripts/setup/check_environment.py` reports which of them it can find.
