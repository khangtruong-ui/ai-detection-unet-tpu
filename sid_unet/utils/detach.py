"""Process detachment utilities for CLI entrypoints (sid-train, sid-cache, sid-kill).

Defaults CLI commands to detach from terminal mode into background sessions,
redirecting stdout and stderr into '.logs.txt'. Supports '--foreground' / '--no-detach'
to execute synchronously in the active terminal session.
"""

from __future__ import annotations

from datetime import datetime
import os
import subprocess
import sys
from typing import List, Optional

DEFAULT_LOG_PATH = ".logs.txt"


def is_detached_worker() -> bool:
    """Check if the current process is already running as a detached child worker."""
    return os.environ.get("_SID_DETACHED") == "1"


def should_detach(argv: Optional[List[str]] = None) -> bool:
    """Determine if execution should detach from terminal mode.

    Returns True by default unless '--foreground' or '--no-detach' is passed,
    or the SID_FOREGROUND environment variable is set, or already detached.
    """
    if is_detached_worker():
        return False

    if os.environ.get("SID_FOREGROUND") in ("1", "true", "True"):
        return False

    args = sys.argv[1:] if argv is None else argv
    for flag in ("--foreground", "--no-detach"):
        if flag in args:
            return False

    return True


def strip_foreground_flags(argv: List[str]) -> List[str]:
    """Remove foreground override flags before launching internal worker."""
    return [a for a in argv if a not in ("--foreground", "--no-detach")]


def maybe_detach_cli(
    cli_name: str,
    module_name: str,
    argv: Optional[List[str]] = None,
    log_file_path: str = DEFAULT_LOG_PATH,
) -> bool:
    """If detachment is required, spawn background process and exit parent.

    Args:
        cli_name: Name of the CLI utility (e.g. 'sid-train', 'sid-cache', 'sid-kill').
        module_name: Full python module path to invoke (e.g. 'sid_unet.train').
        argv: Optional command line arguments list (defaults to sys.argv[1:]).
        log_file_path: Output log file path (defaults to '.logs.txt').

    Returns:
        False if continuing in foreground; does not return in parent if detached (exits 0).
    """
    raw_args = sys.argv[1:] if argv is None else argv

    if not should_detach(raw_args):
        return False

    clean_args = strip_foreground_flags(raw_args)

    # Ensure log directory exists
    log_abs = os.path.abspath(log_file_path)
    log_dir = os.path.dirname(log_abs)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = (
        f"\n{'='*70}\n"
        f"[{timestamp}] 🚀 [{cli_name}] Launching detached background session\n"
        f"Command: {sys.executable} -m {module_name} {' '.join(clean_args)}\n"
        f"Log file: {log_abs}\n"
        f"{'='*70}\n"
    )

    with open(log_abs, "a", encoding="utf-8") as f:
        f.write(banner)
        f.flush()

    child_env = os.environ.copy()
    child_env["_SID_DETACHED"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-u", "-m", module_name, *clean_args, "--foreground"]

    with open(log_abs, "a", encoding="utf-8") as out_f:
        child_proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # setsid to detach controlling terminal
            env=child_env,
        )

    launch_msg = (
        f"🚀 [{cli_name}] Process successfully detached into background (PID: {child_proc.pid})\n"
        f"📄 Activity and training logs are streamed to: {log_file_path}\n"
        f"💡 Use 'sid-kill' to stop running background tasks, or run with '--foreground' for interactive mode."
    )
    print(launch_msg)

    with open(log_abs, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] 📌 [{cli_name}] Child process started with PID {child_proc.pid}\n")

    # Exit the interactive terminal session cleanly
    sys.exit(0)
