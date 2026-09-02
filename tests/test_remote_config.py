from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from makermodslab.remote_teleop.config import (
    OperatorRoleConfig,
    RemoteConfigError,
    RemoteRoleConfigStore,
    RobotRoleConfig,
)


def make_certificate(tmp_path: Path) -> tuple[Path, Path]:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for TLS config validation")
    certificate = tmp_path / "certificate.pem"
    private_key = tmp_path / "private-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=100.64.0.10",
        ],
        check=True,
        capture_output=True,
    )
    private_key.chmod(0o600)
    return certificate, private_key


def test_robot_config_is_private_path_redacted_and_runtime_disabled(tmp_path: Path) -> None:
    certificate, private_key = make_certificate(tmp_path)
    config = RobotRoleConfig(
        node_id="robot-1",
        robot_name="SO101 robot",
        bind_address="100.64.0.10",
        control_port=7443,
        udp_port=7444,
        tls_certificate_path=str(certificate),
        tls_private_key_path=str(private_key),
        leader_calibration_id="leader-1",
        leader_calibration_digest="a" * 64,
    )
    store = RemoteRoleConfigStore(tmp_path / "state")
    assert store.public()["configured"] is False
    store.save_robot(config)

    public = store.public()
    assert public["role"] == "robot"
    assert public["runtime_enabled"] is False
    assert public["config"]["tls_certificate_configured"] is True
    assert "path" not in " ".join(public["config"])
    assert str(tmp_path) not in str(public)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.load() == ("robot", config)


def test_config_rejects_wildcard_public_and_permission_unsafe_key(tmp_path: Path) -> None:
    certificate, private_key = make_certificate(tmp_path)
    private_key.chmod(0o644)
    with pytest.raises(RemoteConfigError, match="owner-only"):
        RobotRoleConfig(
            node_id="robot-1",
            robot_name="SO101 robot",
            bind_address="100.64.0.10",
            control_port=7443,
            udp_port=7444,
            tls_certificate_path=str(certificate),
            tls_private_key_path=str(private_key),
            leader_calibration_id="leader-1",
            leader_calibration_digest="a" * 64,
        )
    with pytest.raises(RemoteConfigError, match="private IP"):
        OperatorRoleConfig(
            node_id="operator-1",
            robot_id="robot-1",
            leader_robot_name="SO101 leader",
            control_uri="wss://8.8.8.8:7443",
            certificate_fingerprint="b" * 64,
        )
