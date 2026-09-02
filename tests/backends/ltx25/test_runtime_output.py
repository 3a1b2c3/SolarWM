from __future__ import annotations

from types import SimpleNamespace

from solarwm.backends.ltx25.runtime import VerifiedTrainingRuntime


class _RelativeCheckpointRuntime:
    def save_checkpoint(self, _step: int) -> str:
        return "checkpoint-00000020"


def test_verified_runtime_preserves_relative_checkpoint_compatibility(
    tmp_path, monkeypatch
) -> None:
    observed = []

    def verify(path):
        observed.append(path)
        return SimpleNamespace(step=20, manifest_digest="a" * 64)

    monkeypatch.setattr("solarwm.backends.ltx25.runtime.verify_checkpoint", verify)
    monkeypatch.setattr(
        "solarwm.backends.ltx25.runtime.validate_ltx_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    runtime = VerifiedTrainingRuntime(
        _RelativeCheckpointRuntime(),  # type: ignore[arg-type]
        config={},
        model_receipt=SimpleNamespace(as_dict=lambda: {}),  # type: ignore[arg-type]
        output_dir=tmp_path,
    )

    assert runtime.save_checkpoint(20) == "a" * 64
    assert observed == [tmp_path / "checkpoints/checkpoint-00000020"]
