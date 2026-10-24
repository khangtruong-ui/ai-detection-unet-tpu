"""Tests for detachment mode, signal shielding, and sid-kill process termination."""

import os
import signal
import subprocess
import sys
import time
import pytest

from sid_unet.kill import (
    find_unet_processes,
    get_ancestor_pids,
    is_unet_process,
    kill_all_background_tasks,
    cli_main as kill_cli_main,
)
from sid_unet.utils.detach import should_detach, is_detached_worker
from sid_unet.utils.signals import shield_process_signals


@pytest.mark.timeout(60)
def test_should_detach_defaults(monkeypatch):
    monkeypatch.delenv("_SID_DETACHED", raising=False)
    monkeypatch.delenv("SID_FOREGROUND", raising=False)

    # By default, should detach
    assert should_detach([]) is True

    # With foreground flag, should NOT detach
    assert should_detach(["--foreground"]) is False
    assert should_detach(["--no-detach"]) is False
    assert should_detach(["--config", "test.yaml", "--foreground"]) is False


@pytest.mark.timeout(60)
def test_ancestor_pids_contains_current():
    ancestors = get_ancestor_pids()
    assert os.getpid() in ancestors


@pytest.mark.timeout(60)
def test_is_unet_process():
    assert is_unet_process(1234, "sid-train --config default.yaml", "", "") is True
    assert is_unet_process(1234, "sid-cache --max-samples 10", "", "") is True
    assert is_unet_process(1234, "python -m sid_unet.train", "", "") is True
    assert is_unet_process(1234, "bash", "/bin/bash", "/workspace") is False
    assert is_unet_process(1234, "pytest", "/usr/bin/pytest", "") is False


@pytest.mark.timeout(60)
def test_kill_dummy_background_task(tmp_path):
    log_file = tmp_path / ".logs.txt"

    # Spawn dummy background process mimicking a sid_unet task
    dummy = subprocess.Popen(
        [sys.executable, "-c", "import time\n# sid_unet dummy background\ntime.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)

    try:
        assert dummy.poll() is None

        # Verify discovery
        procs = find_unet_processes()
        matching = [p["pid"] for p in procs if p["pid"] == dummy.pid]
        assert len(matching) == 1

        # Test dry-run
        dry_count = kill_all_background_tasks(dry_run=True, quiet=True, log_file=str(log_file))
        assert dry_count >= 1
        assert dummy.poll() is None  # Still alive

        # Test actual kill
        killed = kill_all_background_tasks(force=True, quiet=True, log_file=str(log_file))
        assert killed >= 1
        dummy.wait(timeout=2.0)
        assert dummy.poll() is not None

        # Verify log entry in .logs.txt
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "[sid-kill]" in content
    finally:
        if dummy.poll() is None:
            dummy.kill()


@pytest.mark.timeout(60)
def test_shield_process_signals():
    code = (
        "import signal, sys\n"
        "from sid_unet.utils.signals import shield_process_signals\n"
        "ignored = shield_process_signals(detach_terminal=False, protect_streams=False)\n"
        "if hasattr(signal, 'SIGHUP'):\n"
        "    assert signal.SIGHUP in ignored\n"
        "sys.exit(0)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0
