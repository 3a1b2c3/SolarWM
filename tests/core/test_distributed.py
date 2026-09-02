from __future__ import annotations

import pytest

from solarwm.errors import BackendContractError
from solarwm.runtime.distributed import (
    assert_peer_fingerprints,
    gather_and_assert_sp_identity,
    identity_fingerprint,
)


def test_identity_hash_is_mapping_order_independent() -> None:
    assert identity_fingerprint({"sample": "a", "noise": 7}) == identity_fingerprint(
        {"noise": 7, "sample": "a"}
    )


def test_peer_mismatch_fails_closed() -> None:
    assert assert_peer_fingerprints(["same", "same"]) == "same"
    with pytest.raises(BackendContractError, match="different"):
        assert_peer_fingerprints(["left", "right"])


def test_sp1_does_not_require_torch_distributed() -> None:
    assert gather_and_assert_sp_identity({"sample": "a"}, sp_size=1)
