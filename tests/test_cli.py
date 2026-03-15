"""
CLI integration tests for NV Maser Digital Twin.

Run with:
    pytest tests/test_cli.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
SCRIPTS = REPO / "scripts"


def _run(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a command in the repo root and return the result."""
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO),
    )


# ---------------------------------------------------------------------------
# 1. python -m nv_maser --help
# ---------------------------------------------------------------------------
def test_main_help():
    result = _run(sys.executable, "-m", "nv_maser", "--help")
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 2. python scripts/train_rl.py --help
# ---------------------------------------------------------------------------
def test_train_rl_help():
    result = _run(sys.executable, str(SCRIPTS / "train_rl.py"), "--help")
    assert result.returncode == 0, result.stderr
    # Verify --config flag is advertised
    assert "--config" in result.stdout, (
        "--config flag not found in train_rl.py --help output"
    )


# ---------------------------------------------------------------------------
# 3. python -m nv_maser dataset --help
# ---------------------------------------------------------------------------
def test_dataset_subcommand_help():
    result = _run(sys.executable, "-m", "nv_maser", "dataset", "--help")
    assert result.returncode == 0, result.stderr
    assert "--num-samples" in result.stdout, (
        "--num-samples not found in dataset --help output"
    )


# ---------------------------------------------------------------------------
# 4. Build a tiny dataset (10 samples) and verify cache dir is created
# ---------------------------------------------------------------------------
def test_dataset_build_small(tmp_path):
    cache_dir = tmp_path / "cache"
    result = _run(
        sys.executable, "-m", "nv_maser",
        "dataset",
        "--num-samples", "10",
        "--cache-dir", str(cache_dir),
    )
    assert result.returncode == 0, result.stderr
    assert cache_dir.exists(), "Cache directory was not created"
    # Expect at least one .npz file inside
    npz_files = list(cache_dir.glob("*.npz"))
    assert len(npz_files) >= 1, f"No .npz files found in {cache_dir}"


# ---------------------------------------------------------------------------
# 5. train_rl.py honours --config YAML (2 episodes, 5 steps → quick)
# ---------------------------------------------------------------------------
def test_train_rl_config_flag(tmp_path):
    yaml_path = tmp_path / "override.yaml"
    # Write a minimal YAML override — just override a training field so we
    # verify the flag is parsed without errors; actual training params come
    # from the CLI args below.
    yaml_path.write_text("training:\n  epochs: 1\n")

    result = _run(
        sys.executable,
        str(SCRIPTS / "train_rl.py"),
        "--config", str(yaml_path),
        "--episodes", "2",
        "--steps", "5",
    )
    assert result.returncode == 0, result.stderr
