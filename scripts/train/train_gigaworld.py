#!/usr/bin/env python3
"""Train the EVEWorld objective on GigaWorld-0 (DreamGenBench or AgiBot).

The script resolves the run described by ``--config``, prints it, and hands it to
``eveworld.integrations.gigaworld.trainer.GigaWorldTrainer``::

    python scripts/train/train_gigaworld.py \
        --config configs/paper/gigaworld/dreamgen/eveworld.yaml
    python scripts/train/train_gigaworld.py \
        --config configs/paper/gigaworld/agibot/eveworld.yaml --steps 50 --device cuda:0

``--steps`` and ``--seed`` override ``train.max_steps`` and ``seed`` of the configuration
before it is loaded, and ``--output-dir`` replaces the ``output_dir`` the config derives from
its name, so the effective configuration of a run is the one the script writes to
``<output-dir>/config.yaml``. ``--resume`` continues a run from the checkpoint it names.

``--dry-run`` prints the plan and exercises the training stack on a tiny synthetic batch on
the CPU, then stops before creating the output directory. The released 2B backbone weights
are not needed for that, so the smoke check also runs on a machine without a GPU or a
checkpoint download.

Files written under ``--output-dir`` (the ``output_dir`` of the config by default)::

    config.yaml    the effective configuration of the run, command-line overrides included
    run.json       command line, device, step budget and dataset size of the run

The trainer writes its checkpoints next to them, one every ``train.save_every`` steps.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

RUN_CONFIG_NAME = "config.yaml"
RUN_RECORD_NAME = "run.json"

LAYOUT = """\
files written under --output-dir:

    config.yaml    the effective configuration of the run, command-line overrides included
    run.json       command line, device, step budget and dataset size of the run

the trainer writes its checkpoints next to them, one every ``train.save_every`` steps.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the training script."""
    parser = argparse.ArgumentParser(
        description="Train EVEWorld on GigaWorld-0 (DreamGenBench or AgiBot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help=("training configuration, e.g. configs/paper/gigaworld/dreamgen/eveworld.yaml " "(default: $EVEWORLD_CONFIG)"),
    )
    run.add_argument("--steps", type=int, help="override train.max_steps of the configuration")
    run.add_argument("--seed", type=int, help="override seed of the configuration")
    run.add_argument("--output-dir", help="directory of the run, overriding output_dir")
    run.add_argument("--resume", help="checkpoint to continue the run from")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", help="torch device of the run, e.g. cuda:0 or cpu")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the synthetic smoke step, without training or writing",
    )
    return parser.parse_args(argv)


def repo_root() -> Path:
    """Repository root: ``eveworld.utils.io.repo_root`` when importable, else the pyproject walk."""
    try:
        from eveworld.utils.io import repo_root as _library_root
    except ImportError:
        pass
    else:
        return Path(_library_root())
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@dataclass
class RunPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration, the stem of the default output directory.
        output_dir: Directory the run writes its config, record and checkpoints to.
        split: Split file listing the clips of the run.
        metadata: Directory holding the per-clip metadata of the split.
        clips: Number of clips the split file lists.
        steps: Optimisation steps of the run, ``train.max_steps`` after the overrides.
        seed: Seed of the run.
        device: Device requested on the command line, or ``None`` for the trainer default.
        resume: Checkpoint to continue from, or ``None`` for a run from the released weights.
        command: Command line of the run, quoted, as recorded in ``run.json``.
        dry_run: Whether the run stops after the plan and the smoke step.
    """

    config_path: Path
    config: Any
    name: str
    output_dir: Path
    split: Path
    metadata: Path
    clips: int
    steps: int
    seed: int
    device: str | None
    resume: Path | None
    command: str
    dry_run: bool


def load_config(config_path: Path, overrides: Sequence[str]) -> Any:
    """Parse ``config_path`` with ``overrides`` applied on top of it."""
    from eveworld.utils.config import load_config as _load_config

    return _load_config(config_path, overrides)


def cfg_value(config: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` out of a parsed configuration without raising on a missing key."""
    from eveworld.utils.config import cfg_get

    return cfg_get(config, key, default)


def split_rows(path: Path) -> list[str]:
    """Rows of a split file, i.e. its lines that are neither comments nor blank."""
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append(stripped)
    return rows


def build_plan(args: argparse.Namespace, config_path: Path, argv: Sequence[str]) -> RunPlan:
    """Resolve the configuration, the split and the run settings into a :class:`RunPlan`."""
    root = repo_root()
    overrides: list[str] = []
    if args.steps is not None:
        overrides.append(f"train.max_steps={int(args.steps)}")
    if args.seed is not None:
        overrides.append(f"seed={int(args.seed)}")
    if args.output_dir is not None:
        overrides.append(f"output_dir={Path(args.output_dir).expanduser()}")
    config = load_config(config_path, overrides)

    from eveworld.utils.config import resolve_path

    output_dir = resolve_path(cfg_value(config, "output_dir", "outputs/run"), root)
    split = resolve_path(cfg_value(config, "data.split", ""), root)
    metadata = resolve_path(cfg_value(config, "data.metadata", ""), root)
    if not split.is_file():
        raise FileNotFoundError(f"split file {split} does not exist")
    resume = None if not args.resume else resolve_path(args.resume, root)
    command = " ".join(shlex.quote(part) for part in ("python scripts/train/train_gigaworld.py", *argv))
    return RunPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        output_dir=output_dir,
        split=split,
        metadata=metadata,
        clips=len(split_rows(split)),
        steps=int(cfg_value(config, "train.max_steps", 0)),
        seed=int(cfg_value(config, "seed", 0)),
        device=args.device,
        resume=resume,
        command=command,
        dry_run=bool(args.dry_run),
    )


def print_plan(plan: RunPlan) -> None:
    """Print the run, one aligned field per line."""
    config = plan.config
    lr = cfg_value(config, "train.lr")
    weight_decay = cfg_value(config, "train.weight_decay")
    batch = int(cfg_value(config, "train.batch_size", 0))
    accum = int(cfg_value(config, "train.grad_accum", 1))
    save_every = int(cfg_value(config, "train.save_every", 0))
    resolution = cfg_value(config, "model.resolution", (0, 0))
    checkpoint = cfg_value(config, "model.checkpoint", "-")
    gated = "gated" if bool(cfg_value(config, "method.gated", False)) else "ungated"
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(f"output dir:   {plan.output_dir}")
    print(
        f"method:       {cfg_value(config, 'method.kind', 'eveworld')}, "
        f"layer {int(cfg_value(config, 'method.layer', 0))}, "
        f"p_dup {cfg_value(config, 'method.p_dup')}, "
        f"lambda_tia {cfg_value(config, 'method.lambda_tia')}, "
        f"warmup {int(cfg_value(config, 'method.warmup_steps', 0))}, "
        f"sigma {cfg_value(config, 'method.sigma_low')}-{cfg_value(config, 'method.sigma_high')}, "
        f"{gated}"
    )
    print(f"optimizer:    {cfg_value(config, 'train.optimizer', '-')}, lr {lr:g}, " f"weight decay {weight_decay:g}")
    print(f"schedule:     {plan.steps} steps, batch {batch} x {accum} accumulation = " f"{batch * accum} clips, checkpoint every {save_every}")
    print(
        f"model:        {cfg_value(config, 'model.backbone', '-')} / {checkpoint}, "
        f"{int(resolution[0])}x{int(resolution[1])}, "
        f"{int(cfg_value(config, 'model.num_frames', 0))} frames at "
        f"{cfg_value(config, 'model.fps')} fps, block {int(cfg_value(config, 'model.block_index', 0))}"
    )
    print(f"data:         {plan.clips} clips from {plan.split}")
    found = "" if plan.metadata.is_dir() else " (not found)"
    print(f"metadata:     {plan.metadata}{found}")
    print(f"seed:         {plan.seed}")
    print(f"device:       {plan.device or 'auto'}")
    print(f"resume:       {plan.resume or 'none'}")


def instantiate(cls: Any, **kwargs: Any) -> Any:
    """Instantiate ``cls`` with the keyword arguments it declares.

    Classes with ``**kwargs`` receive every argument; otherwise, undeclared keys are omitted.
    """
    parameters = inspect.signature(cls).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return cls(**kwargs)
    return cls(**{key: value for key, value in kwargs.items() if key in parameters})


def write_run_files(plan: RunPlan) -> None:
    """Write the effective configuration and the run record into the output directory."""
    from eveworld.utils.config import save_config
    from eveworld.utils.io import write_json

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(plan.config, plan.output_dir / RUN_CONFIG_NAME)
    record = {
        "name": plan.name,
        "command": plan.command,
        "config": str(plan.config_path),
        "split": str(plan.split),
        "metadata": str(plan.metadata),
        "clips": plan.clips,
        "steps": plan.steps,
        "seed": plan.seed,
        "device": plan.device or "auto",
        "resume": None if plan.resume is None else str(plan.resume),
    }
    write_json(record, plan.output_dir / RUN_RECORD_NAME)


def smoke_step(plan: RunPlan) -> None:
    """Step the configured trainer on a small synthetic batch without writing a run."""
    from eveworld.integrations.gigaworld import GigaWorldTrainer

    trainer = GigaWorldTrainer(
        config=plan.config,
        output_dir=plan.output_dir,
        device="cpu",
        seed=plan.seed,
        max_steps=plan.steps,
        batch_size=1,
        grad_accum=1,
        optimizer_kind="adamw",
    )
    summary = trainer.dry_run(steps=1)
    print(
        f"smoke:        {trainer.method_kind}, transport={trainer.tia_enabled}, "
        f"igr {summary['igr']:.4f}, tia {summary['tia']:.4f}, total {summary['total']:.4f}"
    )
    report_geometry(plan)


def report_geometry(plan: RunPlan) -> None:
    """Print the geometry the integration resolves for the configuration, when importable."""
    try:
        from eveworld.integrations.gigaworld.model import GigaWorldConfig, latent_grid
    except (ImportError, AttributeError) as error:
        print(f"smoke:        skipped: eveworld.integrations.gigaworld is not importable ({error})")
        return
    config = GigaWorldConfig.from_config(plan.config)
    frames, height, width = latent_grid(config)
    print(
        f"smoke:        geometry {config.variant}, {config.width}x{config.height}, "
        f"{config.num_frames} frames, {frames}x{height}x{width} latents, block {config.block_index}"
    )


def train(plan: RunPlan) -> int:
    """Run the training described by ``plan``."""
    import eveworld.integrations.gigaworld as integration
    from eveworld.utils.config import resolve_path
    from eveworld.utils.seed import set_seed

    if not plan.metadata.is_dir():
        raise FileNotFoundError(f"metadata directory {plan.metadata} does not exist")
    set_seed(plan.seed)
    write_run_files(plan)
    model_config = integration.GigaWorldConfig.from_config(plan.config)
    dataset_cls = integration.AgiBotDataset if model_config.variant == "agibot" else integration.DreamGenDataset
    dataset = instantiate(
        dataset_cls,
        config=plan.config,
        root=resolve_path(cfg_value(plan.config, "data.root", "."), repo_root()),
        split=plan.split,
        metadata_dir=plan.metadata,
        num_frames=model_config.num_frames,
        resolution=(model_config.width, model_config.height),
        fps=model_config.fps,
        seed=plan.seed,
    )
    trainer = instantiate(
        integration.GigaWorldTrainer,
        config=plan.config,
        dataset=dataset,
        output_dir=plan.output_dir,
        device=plan.device,
        resume=plan.resume,
        max_steps=plan.steps,
        seed=plan.seed,
        optimizer_kind=cfg_value(plan.config, "train.optimizer", "came8bit"),
        lr=float(cfg_value(plan.config, "train.lr", 4.32e-5)),
        weight_decay=float(cfg_value(plan.config, "train.weight_decay", 0.01)),
        batch_size=int(cfg_value(plan.config, "train.batch_size", 8)),
        grad_accum=int(cfg_value(plan.config, "train.grad_accum", 8)),
        save_every=int(cfg_value(plan.config, "train.save_every", 50)),
    )
    trainer.train()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: resolve the run, then train it or dry-run it."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(arguments)
    if not args.config:
        print("error: --config is required (or export EVEWORLD_CONFIG)", file=sys.stderr)
        return 2
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        print(f"error: config file {config_path} does not exist", file=sys.stderr)
        return 2
    if args.steps is not None and args.steps < 1:
        print("error: --steps must be positive", file=sys.stderr)
        return 2
    try:
        plan = build_plan(args, config_path, arguments)
        print_plan(plan)
        if plan.dry_run:
            smoke_step(plan)
            print("dry run: no run directory written")
            return 0
        return train(plan)
    except ImportError as error:
        print(f"error: the eveworld package is not importable: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, NotImplementedError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
