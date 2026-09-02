from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solarwm.backends.wan22.runtime import data as raw_data
from solarwm.data.archive import RawSample
from solarwm.data.sampling import SamplePlan


def _plan() -> SamplePlan:
    return SamplePlan(
        sample_id="dataset/bad-camera",
        key="bad-camera",
        shard="raw/shard.tar",
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=0,
        source_frame_indices=tuple(range(81)),
        reader_rank=7,
        worker_id=0,
    )


class _Reader:
    def __init__(self, result: RawSample | Exception) -> None:
        self.result = result

    def materialize(self, _plan: SamplePlan) -> RawSample:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _raw_sample(*, finite: bool) -> RawSample:
    plan = _plan()
    return RawSample(
        plan=plan,
        index_values={"fps": 16.0},
        caption="caption",
        scene="scene",
        manifest={
            "video": {"fps": 16.0},
            "camera": {
                "array_key": "c2w",
                "convention": raw_data.CAMERA_CONVENTION,
                "dtype": "float32",
                "finite": finite,
                "magnitude_audit_seconds": 10.0,
                "max_camera_abs": 20.0,
                "max_rel_translation": 20.0,
                "shape": [81, 4, 4],
            },
        },
        members={"video_member": b"video", "camera_member": b"camera"},
    )


def _config() -> dict[str, Any]:
    return {
        "data": {
            "height": 480,
            "width": 832,
            "fps": 16.0,
            "max_rel_translation": 20.0,
            "max_camera_abs": 20.0,
            "camera_array_key": "c2w",
        },
        "model": {"frame_sequence_length": 1560},
    }


def _batch_config(*, micro_batch_size: int = 1) -> dict[str, Any]:
    config = _config()
    config["data"].update(
        train_index="train.jsonl",
        pixel_frames=81,
        random_start=True,
        seed=42,
        shuffle_buffer=1,
        partition_mode="global_occurrence",
        transport={"root": "/read-only-data"},
    )
    config["train"] = {"micro_batch_size": micro_batch_size}
    return config


class _ShardContext:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _ShardContext:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass


class _OnePlanPerEpochSampler:
    def __init__(self, plan: SamplePlan) -> None:
        self.plan = plan

    def iter_epoch(self, _epoch: int) -> Any:
        yield self.plan


def _patch_raw_iterator(
    monkeypatch: pytest.MonkeyPatch,
    materialize: Any,
) -> None:
    plan = _plan()
    monkeypatch.setattr(raw_data, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(raw_data, "read_index", lambda *_args: (object(),))
    monkeypatch.setattr(raw_data, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "TarShardReader", _ShardContext)
    monkeypatch.setattr(raw_data, "RawSampleReader", lambda *_args: object())
    monkeypatch.setattr(
        raw_data,
        "CanonicalSampler",
        lambda *_args: _OnePlanPerEpochSampler(plan),
    )
    monkeypatch.setattr(raw_data, "_materialize_raw_sample", materialize)


def _topology() -> SimpleNamespace:
    return SimpleNamespace(
        dp_rank=0,
        dp_world_size=1,
        node_id=0,
        node_count=1,
        local_dp_rank=0,
        local_dp_world_size=1,
    )


def test_materialize_raw_sample_skips_nonfinite_camera_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        raw_data,
        "decode_video",
        lambda *_args, **_kwargs: torch.zeros((81, 3, 1, 1)),
    )

    result = raw_data._materialize_raw_sample(
        _Reader(_raw_sample(finite=False)),
        _plan(),
        _config(),
    )

    assert result is None
    assert (
        "[wds][rank7] skip bad-camera: DataContractError: "
        "raw Wan manifest must attest camera finite=true"
    ) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failure_site", "error"),
    [
        ("materialize", OSError("broken tar member")),
        ("decode", ValueError("broken decoded sample")),
    ],
)
def test_materialize_raw_sample_skips_any_sample_local_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_site: str,
    error: Exception,
) -> None:
    sample = _raw_sample(finite=True)
    reader = _Reader(error if failure_site == "materialize" else sample)
    if failure_site == "decode":

        def fail_decode(*_args: Any) -> raw_data.DecodedWanSample:
            raise error

        monkeypatch.setattr(raw_data, "decode_raw_sample", fail_decode)

    assert raw_data._materialize_raw_sample(reader, _plan(), _config()) is None
    output = capsys.readouterr().out
    assert "[wds][rank7] skip bad-camera:" in output
    assert f"{type(error).__name__}: {error}" in output


def test_materialize_raw_sample_skips_a_prefetch_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_prepare(_plan: SamplePlan) -> None:
        raise OSError("temporary object download failure")

    assert (
        raw_data._materialize_raw_sample(
            _Reader(_raw_sample(finite=True)),
            _plan(),
            _config(),
            prepare=fail_prepare,
        )
        is None
    )
    assert (
        "[wds][rank7] skip bad-camera: OSError: temporary object download failure"
        in capsys.readouterr().out
    )


def test_raw_iterator_fails_only_after_a_complete_epoch_has_no_healthy_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_raw_iterator(monkeypatch, lambda *_args, **_kwargs: None)
    batches = raw_data.iter_raw_batches(_batch_config(), _topology())

    with pytest.raises(
        RuntimeError,
        match=r"reader rank=0 worker=0 emitted no samples in epoch 0",
    ):
        next(batches)


def test_raw_iterator_does_not_treat_an_incomplete_batch_as_an_empty_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = object()
    _patch_raw_iterator(monkeypatch, lambda *_args, **_kwargs: healthy)
    monkeypatch.setattr(raw_data, "collate_raw_samples", lambda samples: tuple(samples))
    batches = raw_data.iter_raw_batches(
        _batch_config(micro_batch_size=2),
        _topology(),
    )

    assert next(batches) == (healthy, healthy)
