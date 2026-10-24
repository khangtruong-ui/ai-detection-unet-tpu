"""Process cleanup and termination utility for background SID-UNet tasks (sid-kill).

Discovers running or background sid_unet/sid-train/sid-cache processes, terminates them cleanly,
and logs actions to '.logs.txt'.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("sid_unet.kill")

UNET_CLI_NAMES = {
    "sid-train",
    "sid-cache",
    "sid-dataset-cache",
    "sid-eval",
    "sid-predict",
    "sid-kill",
}

UNET_CMDLINE_SUBSTRINGS = [
    "sid_unet",
    "sid_unet_tpu",
    "ai-detection-unet",
    "sid-train",
    "sid-cache",
]

PROTECTED_PROCESS_NAMES = {
    "systemd",
    "init",
    "sshd",
    "bash",
    "sh",
    "zsh",
    "tmux",
    "screen",
    "agy",
    "pytest",
    "cron",
}


def get_ancestor_pids() -> Set[int]:
    """Collect all ancestor PIDs of the current process up to root."""
    ancestors: Set[int] = set()
    current = os.getpid()
    ancestors.add(current)
    try:
        import psutil
        p = psutil.Process(current)
        for parent in p.parents():
            ancestors.add(parent.pid)
    except Exception:
        pid = current
        while pid > 1:
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat_parts = f.read().split()
                    ppid = int(stat_parts[3])
                    if ppid <= 1 or ppid in ancestors:
                        break
                    ancestors.add(ppid)
                    pid = ppid
            except Exception:
                break
    return ancestors


def is_unet_process(pid: int, cmdline_str: str, exe_str: str, cwd_str: str) -> bool:
    """Determine whether a process is related to sid_unet background tasks."""
    cmd_lower = cmdline_str.lower()
    exe_name = os.path.basename(exe_str).lower() if exe_str else ""

    for cli_name in UNET_CLI_NAMES:
        if cli_name in cmd_lower or exe_name == cli_name:
            if "sid-kill" not in cmd_lower:
                return True

    for sub in UNET_CMDLINE_SUBSTRINGS:
        if sub in cmd_lower:
            if "sid-kill" not in cmd_lower and "pytest" not in cmd_lower:
                return True

    return False


def find_unet_processes(include_children: bool = True) -> List[Dict[str, Any]]:
    """Scan the system for background or active sid_unet tasks."""
    ancestors = get_ancestor_pids()
    target_procs: Dict[int, Dict[str, Any]] = {}

    try:
        import psutil

        for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "exe", "cwd"]):
            try:
                pid = proc.info["pid"]
                if pid in ancestors or pid <= 1:
                    continue

                name = proc.info.get("name") or ""
                if name.lower() in PROTECTED_PROCESS_NAMES:
                    continue

                cmd_list = proc.info.get("cmdline") or []
                cmdline = " ".join(cmd_list)
                exe = proc.info.get("exe") or ""
                cwd = proc.info.get("cwd") or ""

                if is_unet_process(pid, cmdline, exe, cwd):
                    target_procs[pid] = {
                        "pid": pid,
                        "name": name,
                        "cmdline": cmdline,
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.debug(f"Process scan error: {e}")

    return list(target_procs.values())


def kill_all_background_tasks(
    force: bool = False,
    timeout: float = 2.0,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    log_file: str = ".logs.txt",
) -> int:
    """Discover and cleanly terminate all background sid_unet tasks."""
    procs = find_unet_processes()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"[{timestamp}] 🛑 [sid-kill] Discovered {len(procs)} candidate background task(s).\n"
    if not quiet:
        print(f"🔍 Discovered {len(procs)} candidate sid_unet background process(es).")

    if not procs:
        if not quiet:
            print("✨ No background tasks found.")
        _append_log(log_file, log_entry + f"[{timestamp}] ✨ No tasks terminated.\n")
        return 0

    if dry_run:
        for p in procs:
            msg = f"   • PID {p['pid']}: {p['cmdline'][:80]}"
            if not quiet:
                print(msg)
            log_entry += msg + "\n"
        _append_log(log_file, log_entry)
        return len(procs)

    pids = [p["pid"] for p in procs]

    # Send SIGINT first to trigger graceful checkpoint saving if trainer is active
    initial_signal = signal.SIGKILL if force else signal.SIGINT
    for pid in pids:
        try:
            os.kill(pid, initial_signal)
        except (ProcessLookupError, PermissionError):
            pass

    if not force:
        # Wait up to timeout for processes to finish graceful cleanup
        t0 = time.time()
        while time.time() - t0 < timeout:
            alive = [p for p in pids if _is_alive(p)]
            if not alive:
                break
            time.sleep(0.1)

        # Escalate remaining to SIGKILL
        still_alive = [p for p in pids if _is_alive(p)]
        for pid in still_alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    time.sleep(0.15)
    final_alive = [p for p in pids if _is_alive(p)]
    killed_count = len(pids) - len(final_alive)

    summary = f"[{timestamp}] ✅ [sid-kill] Successfully killed {killed_count} process(es)."
    if final_alive:
        summary += f" ({len(final_alive)} still alive: {final_alive})"
    _append_log(log_file, summary + "\n")

    if not quiet:
        print(f"✅ Successfully terminated {killed_count} background process(es).")

    return killed_count


def _is_alive(pid: int) -> bool:
    try:
        import psutil
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        pass

    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line
        return True
    except (ProcessLookupError, FileNotFoundError):
        return False
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _append_log(log_file: str, content: str):
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for sid-kill."""
    parser = argparse.ArgumentParser(
        prog="sid-kill",
        description="Terminate all background or orphaned tasks related to sid_unet.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Immediately terminate processes with SIGKILL without waiting for graceful shutdown.",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Display matching processes without terminating them.",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=2.0,
        help="Grace period in seconds before SIGKILL escalation (default: 2.0).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    kill_all_background_tasks(
        force=args.force,
        timeout=args.timeout,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
