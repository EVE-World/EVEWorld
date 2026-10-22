"""Clip-level MLR aggregation and the timestamp diagnostic."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eveworld.evaluation.mlr import adjust_counts, aggregate, clip_mlr, detect_events, timestamp_violation_rate


def test_clip_outcomes_and_timestamp_rates_are_distinct():
    rows = [{"events": [False, False, True, False]}, {"events": [False] * 20}]
    result = aggregate(rows)
    assert result["mlr"] == 50.0
    assert result["mean_timestamp_violation_rate"] == 12.5
    assert result["events"] == 1
    assert clip_mlr(rows[0]["events"]) == timestamp_violation_rate(rows[0]["events"]) == 25.0


def test_errors_and_ineligible_clips_do_not_enter_the_mlr_denominator():
    result = aggregate([{"event": True}, {"eligible": False}, {"error": "unreadable"}, {"eligible": False, "event": True}])
    assert result["mlr"] == 100.0
    assert result["coverage"] == 25.0
    assert result["num_eligible"] == 1
    assert result["num_clips"] == 4
    assert result["errors"] == ["unreadable"]


def test_binary_rows_need_no_timestamp_diagnostic():
    result = aggregate([{"event": True}, {"event": False}])
    assert result["mlr"] == 50.0
    assert result["mean_timestamp_violation_rate"] is None


def test_legacy_timestamp_rows_still_establish_binary_outcomes():
    result = aggregate([{"mlr": 0.5}, {"mlr": 0.0}])
    assert result["mlr"] == 50.0
    assert result["mean_timestamp_violation_rate"] == 0.25


@pytest.mark.parametrize("row", [{}, {"events": []}, {"mlr": None}, {"mlr": float("nan")}, {"mlr": -1}, {"mlr": 101}])
def test_missing_or_invalid_outcomes_are_not_counted_as_success(row):
    with pytest.raises(ValueError):
        aggregate([row])


def test_conflicting_event_fields_are_rejected():
    with pytest.raises(ValueError, match="disagrees"):
        aggregate([{"events": [False, True], "event": False}])


def test_empty_run_has_no_eligible_clips():
    result = aggregate([])
    assert result["mlr"] == result["coverage"] == result["num_eligible"] == 0


@pytest.mark.parametrize(
    "counts,occluded,flagged",
    [
        ([1, 0, 0, 0], False, True),
        ([1, 0, 0, 1], False, True),
        ([1, 2, 2, 2], False, True),
        ([1, 2, 1, 1], False, False),
        ([1, 0, 0, 0], True, False),
    ],
)
def test_persistence_and_occlusion_feed_clip_outcomes(counts, occluded, flagged):
    counts = np.asarray(counts)
    exemptions = np.full((len(counts), 1), occluded)
    adjusted = adjust_counts(counts, exemptions, expected=1)
    events = detect_events(adjusted != 1, exemptions=adjusted != counts, k=2)
    result = aggregate([{"events": events}])
    assert result["mlr"] == (100.0 if flagged else 0.0)


def test_cli_and_library_share_aggregation_and_run_coverage(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_mlr_entry", root / "scripts/evaluate/eval_mlr.py")
    entry = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, entry)
    spec.loader.exec_module(entry)
    rows = [{"events": [False, False, True, False]}, {"events": [False] * 4}]
    plan = SimpleNamespace(
        config_path=Path("protocol.yaml"),
        name="test",
        model="test",
        protocol={"protocol": "test"},
        command="test",
        pred_dir=Path("clips"),
        clips=[1, 2],
        benchmark={"coverage": 19.22},
    )
    result = entry.summarize(plan, rows)
    assert result["mlr"] == aggregate(rows)["mlr"] == 50.0
    assert result["coverage"] == result["run_coverage"] == 100.0
    assert result["reference_coverage"] == 19.22
    assert result["mean_timestamp_violation_rate"] == result["mean_clip_mlr"] == 12.5
