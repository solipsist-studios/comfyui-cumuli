# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Run one of the pack's external stages as a child process and stream its output.

The subprocess primitive shared by every long stage -- ring generation, the rig
solve and training -- so its messages name the pack, not any one of the
checkouts it drives. Everything crosses the boundary as argv, files and one
merged output stream: nothing here imports the child's code.

Two things this module is careful about:

* **No pipe deadlock.** stderr is merged into stdout (``STDOUT``) and a single
  reader drains it continuously. There is never a second pipe that could fill
  its buffer while we block on the first.
* **Cancellation.** The child is started in its own process group, so pressing
  Cancel in ComfyUI kills the whole tree (``conda run`` -> ``python`` ->
  GVHMR/skeleton worker children) instead of orphaning the GPU job.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("comfyui-cumuli")

#: Bytes pulled from the child per read. Small enough that tqdm carriage
#: returns reach the UI promptly, large enough to stay cheap.
_CHUNK = 4096

#: How often the cancellation callback is consulted while the child is quiet.
_POLL_SECONDS = 0.25


class SubprocessError(RuntimeError):
    """Raised when the child exits non-zero."""

    def __init__(self, message: str, returncode: int, tail: Sequence[str]):
        super().__init__(message)
        self.returncode = returncode
        self.tail = list(tail)


class SubprocessCancelled(RuntimeError):
    """Raised after the child has been killed on the caller's request."""


@dataclass
class CommandResult:
    returncode: int
    lines: list[str]
    elapsed: float


def _iter_output(stream, should_cancel: Callable[[], bool] | None) -> Iterator[str]:
    """Yield logical lines, splitting on both ``\\n`` and ``\\r``.

    tqdm redraws its bar with a carriage return and no newline, so a plain
    ``readline`` would block until the bar finishes. Reading raw chunks and
    splitting on either terminator gives one string per bar repaint.
    """

    buffer = bytearray()
    last_check = time.monotonic()
    while True:
        chunk = stream.read1(_CHUNK) if hasattr(stream, "read1") else stream.read(_CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
        start = 0
        for index, byte in enumerate(buffer):
            if byte in (0x0A, 0x0D):
                piece = bytes(buffer[start:index]).decode("utf-8", "replace")
                start = index + 1
                if piece:
                    yield piece
        del buffer[:start]
        now = time.monotonic()
        if should_cancel is not None and now - last_check > _POLL_SECONDS:
            last_check = now
            if should_cancel():
                return
    if buffer:
        yield bytes(buffer).decode("utf-8", "replace")


def _terminate(process: subprocess.Popen, grace: float = 10.0) -> None:
    """Stop the child and everything it spawned."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        LOGGER.warning("Cumuli: subprocess %s did not exit after SIGKILL.", process.pid)


def run_streaming(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    tail_lines: int = 60,
) -> CommandResult:
    """Run ``argv`` to completion, calling ``on_line`` for every output line.

    ``should_cancel`` is polled between reads; returning ``True`` kills the
    process group and raises :class:`SubprocessCancelled`.
    """

    started = time.monotonic()
    LOGGER.info("Cumuli: launching %s (cwd=%s)", " ".join(str(a) for a in argv), cwd)
    process = subprocess.Popen(  # noqa: S603 - argv is built from validated settings
        [str(a) for a in argv],
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )

    lines: list[str] = []
    cancelled = False
    try:
        for line in _iter_output(process.stdout, should_cancel):
            lines.append(line)
            if len(lines) > tail_lines:
                del lines[: len(lines) - tail_lines]
            if on_line is not None:
                on_line(line)
        if should_cancel is not None and should_cancel():
            cancelled = True
    except BaseException:
        _terminate(process)
        raise
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass

    if cancelled:
        _terminate(process)
        raise SubprocessCancelled("The run was cancelled.")

    returncode = process.wait()
    elapsed = time.monotonic() - started
    if returncode != 0:
        detail = "\n".join(lines[-20:]) or "(no output captured)"
        raise SubprocessError(
            f"Subprocess exited with status {returncode}.\n{detail}",
            returncode=returncode,
            tail=lines,
        )
    return CommandResult(returncode=returncode, lines=lines, elapsed=elapsed)
