"""Method isolation and checkpoint compatibility on the CPU stand-in backbone."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from eveworld.integrations.gigaworld.model import GigaWorldConfig, GigaWorldModel
from eveworld.integrations.gigaworld.trainer import GigaWorldTrainer, _constant_stream, _StandInPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trainer_factory(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def make(kind):
        config = yaml.safe_load((ROOT / f"configs/paper/gigaworld/dreamgen/{kind}.yaml").read_text())
        config["model"].update(resolution=[32, 32], num_frames=9, block_index=0)
        config["method"].update(layer=0, window=3)
        model = GigaWorldModel(GigaWorldConfig.from_config(config), pipeline=_StandInPipeline())
        return GigaWorldTrainer(
            config,
            model=model,
            output_dir=tmp_path / kind,
            device="cpu",
            optimizer_kind="adamw",
            batch_size=1,
            grad_accum=1,
            max_steps=2,
        )

    yield make
    torch.set_num_threads(threads)


@pytest.mark.parametrize("kind", ["sft", "igr", "tia", "eveworld"])
def test_method_training_and_checkpoint_round_trip(kind, trainer_factory):
    trainer = trainer_factory(kind)
    enabled = kind in {"tia", "eveworld"}
    trainer._prepare()
    before = {key: value.clone() for key, value in trainer.model.denoiser.state_dict().items()}
    batch = trainer._synthetic_batch(trainer.config)
    trainer._step(_constant_stream(batch))
    assert any(not torch.equal(before[key], value) for key, value in trainer.model.denoiser.state_dict().items())
    assert (trainer.adapter is not None) == enabled
    assert (trainer.hook is not None) == enabled
    params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    if enabled:
        assert trainer.hook.calls > 0
        assert {id(p) for p in trainer.adapter.parameters()} <= params
    else:
        assert params == {id(p) for p in trainer.model.parameters() if p.requires_grad}
    checkpoint = trainer.save_checkpoint()
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["version"] == 2
    assert payload["method_kind"] == kind
    assert payload["tia_enabled"] == enabled

    restored = trainer_factory(kind)
    assert restored.load_checkpoint(checkpoint) == 1
    for key, value in trainer.model.denoiser.state_dict().items():
        assert torch.equal(value, restored.model.denoiser.state_dict()[key])
    if enabled:
        for key, value in trainer.adapter.state_dict().items():
            assert torch.equal(value, restored.adapter.state_dict()[key])
        assert restored.hook.targets is None
        restored.model.eval().forward(batch, generator=restored.generator)
        assert restored.hook.calls > 0
    else:
        assert restored.adapter is None
        assert restored.hook is None
    restored._step(_constant_stream(batch))
    assert restored.step == 2


@pytest.mark.parametrize("kind", ["tia", "eveworld"])
def test_legacy_transport_checkpoint_loads(kind, trainer_factory, tmp_path):
    trainer = trainer_factory(kind)
    payload = torch.load(trainer.save_checkpoint(), weights_only=False)
    payload["version"] = 1
    del payload["method_kind"], payload["tia_enabled"]
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    restored = trainer_factory(kind)
    assert restored.load_checkpoint(path) == 0
    assert restored.hook is not None


@pytest.mark.parametrize("kind", ["sft", "igr"])
def test_legacy_adapter_is_not_silently_discarded(kind, trainer_factory, tmp_path):
    payload = torch.load(trainer_factory("eveworld").save_checkpoint(), weights_only=False)
    payload["version"] = 1
    del payload["method_kind"], payload["tia_enabled"]
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="contains a TIA adapter"):
        trainer_factory(kind).load_checkpoint(path)


def test_checkpoint_method_must_match_even_without_transport(trainer_factory):
    path = trainer_factory("sft").save_checkpoint()
    with pytest.raises(ValueError, match="does not match"):
        trainer_factory("igr").load_checkpoint(path)


@pytest.mark.parametrize("weight_map", ["legacy_multilevel", "path_loc_2x", "path_loc_3x", "interaction_2x"])
def test_unsupported_weight_map_fails_before_loading_a_model(weight_map):
    with pytest.raises(NotImplementedError, match="method.weight_map"):
        GigaWorldTrainer({"method": {"weight_map": weight_map}}, device="cpu")


def test_training_entry_point_passes_method_and_run_settings(monkeypatch, tmp_path):
    import eveworld.integrations.gigaworld as integration

    spec = importlib.util.spec_from_file_location("train_entry", ROOT / "scripts/train/train_gigaworld.py")
    entry = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, entry)
    spec.loader.exec_module(entry)
    config = yaml.safe_load((ROOT / "configs/paper/gigaworld/dreamgen/igr.yaml").read_text())
    config["train"].update(lr=2e-4, batch_size=2, grad_accum=3, save_every=7)
    received = {}

    class Dataset:
        def __init__(self, **kwargs):
            pass

    class Trainer:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def train(self):
            received["trained"] = True

    monkeypatch.setattr(integration, "DreamGenDataset", Dataset)
    monkeypatch.setattr(integration, "GigaWorldTrainer", Trainer)
    monkeypatch.setattr(entry, "write_run_files", lambda plan: None)
    plan = SimpleNamespace(
        config=config, metadata=tmp_path, split=tmp_path / "split.txt", output_dir=tmp_path, device="cpu", resume=None, seed=17, steps=3
    )
    entry.train(plan)
    assert received["config"] is config
    assert received["config"]["method"]["kind"] == "igr"
    assert received["seed"] == 17
    assert received["max_steps"] == 3
    assert received["lr"] == 2e-4
    assert received["batch_size"] == 2
    assert received["grad_accum"] == 3
    assert received["save_every"] == 7
    assert received["trained"]
