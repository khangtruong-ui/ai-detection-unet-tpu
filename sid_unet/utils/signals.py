"""Signal handling utilities for shielding long-running background tasks from SSH disconnects."""

from __future__ import annotations

import errno
import logging
import os
import signal
import sys
from typing import Any, Iterable, Optional, Set

logger = logging.getLogger("sid_unet.signals")

SSH_DISCONNECT_SIGNAL_NAMES = (
    "SIGHUP",    # 1: Controlling terminal closed / SSH dropped
    "SIGPIPE",   # 13: Broken pipe (closed stdout/stderr)
    "SIGQUIT",   # 3: Terminal quit
    "SIGTSTP",   # 20: Terminal stop
    "SIGTTIN",   # 21: Background tty read
    "SIGTTOU",   # 22: Background tty write
    "SIGWINCH",  # 28: Terminal window resize
    "SIGIO",     # 29: Asynchronous I/O signal
)

SSH_DISCONNECT_SIGNALS: tuple[int, ...] = tuple(
    getattr(signal, name)
    for name in SSH_DISCONNECT_SIGNAL_NAMES
    if hasattr(signal, name)
)


class SafeStreamWrapper:
    """Wraps text streams to silently suppress BrokenPipeError and EIO on terminal detachment."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, s: str) -> int:
        try:
            return self._stream.write(s)
        except (BrokenPipeError, OSError) as e:
            if isinstance(e, OSError) and e.errno not in (errno.EPIPE, errno.EIO, errno.EBADF):
                raise
            return len(s) if hasattr(s, "__len__") else 0

    def flush(self) -> None:
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError) as e:
            if isinstance(e, OSError) and e.errno not in (errno.EPIPE, errno.EIO, errno.EBADF):
                raise

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def detach_controlling_terminal() -> bool:
    """Detach the process from its controlling terminal session to prevent kernel hangups."""
    detached = False
    try:
        os.setsid()
        detached = True
    except OSError:
        try:
            os.setpgrp()
            detached = True
        except OSError:
            pass
    return detached


def protect_standard_streams() -> None:
    """Protect sys.stdout and sys.stderr against broken pipes on terminal disconnect."""
    if not isinstance(sys.stdout, SafeStreamWrapper):
        sys.stdout = SafeStreamWrapper(sys.stdout)
    if not isinstance(sys.stderr, SafeStreamWrapper):
        sys.stderr = SafeStreamWrapper(sys.stderr)


def shield_process_signals(
    signals_to_ignore: Optional[Iterable[int]] = None,
    log_info: bool = False,
    detach_terminal: bool = True,
    protect_streams: bool = True,
) -> Set[int]:
    """Shield the process from SSH disconnects while preserving clean terminability."""
    if signals_to_ignore is None:
        signals_to_ignore = SSH_DISCONNECT_SIGNALS

    if detach_terminal:
        detach_controlling_terminal()

    if protect_streams:
        protect_standard_streams()

    ignored: Set[int] = set()
    for sig in signals_to_ignore:
        try:
            signal.signal(sig, signal.SIG_IGN)
            ignored.add(sig)
        except (ValueError, OSError) as e:
            if log_info:
                logger.debug(f"Could not ignore signal {sig}: {e}")

    return ignored
