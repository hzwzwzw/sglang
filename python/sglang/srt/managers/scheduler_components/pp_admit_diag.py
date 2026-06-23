"""Per-rank disk logger for PP admit-time consensus divergence debugging.

When ``SGLANG_PP_ADMIT_DIAG=1`` is set, every PP scheduler writes a
structured event log to ``/tmp/sglang_pp_admit_pp{N}_tp{M}.log`` (path
overridable via ``SGLANG_PP_ADMIT_DIAG_DIR``). Events are flushed
line-by-line so a crash leaves the most recent activity on disk.

Disk logs (vs. crash_diag's in-memory ring) keep full history; the
ring evicts older events at low QPS. ``diff`` across rank files is the
fastest way to find where consensus or admit decisions started to
diverge across ranks. See ``scripts/diff_admit_diag.sh`` for the
analysis workflow.

Event kinds emitted (one event per line, ``KIND key=value ...``):

    LOCAL_TREE_MATCH      per-rid match_prefix length contribution
    PHASE_A_OUT           tree-match dict this rank is forwarding
    PHASE_B_IN            tree-match dict this rank received
    ADMIT_DECISION        per-rid admit / defer decision
    TREE_INSERT           cache_(un)finished_req insert event
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "", "false", "False", "no", "No")


PP_ADMIT_DIAG_ENABLED = _env_flag("SGLANG_PP_ADMIT_DIAG", False)


class PPAdmitDiagLogger:
    """Per-rank append-only log writer. Thread-local lock for concurrent
    forward + scheduler thread safety. Best-effort -- never raises."""

    def __init__(self, pp_rank: int, tp_rank: int):
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.enabled = PP_ADMIT_DIAG_ENABLED
        self._fh = None
        self._lock = threading.Lock()
        if not self.enabled:
            return
        diag_dir = os.environ.get("SGLANG_PP_ADMIT_DIAG_DIR", "/tmp")
        self.path = os.path.join(
            diag_dir, f"sglang_pp_admit_pp{pp_rank}_tp{tp_rank}.log"
        )
        try:
            # Truncate at startup so each run starts fresh; users who
            # want to keep prior logs should rotate manually.
            self._fh = open(self.path, "w", buffering=1)  # line-buffered
            self._fh.write(
                f"# pp_admit_diag start pp={pp_rank} tp={tp_rank} "
                f"pid={os.getpid()} t={time.time():.3f}\n"
            )
            self._fh.flush()
        except Exception as e:  # noqa: BLE001
            logger.warning("[pp_admit_diag] failed to open %s: %r", self.path, e)
            self._fh = None
            self.enabled = False

    def log(self, kind: str, **fields: Any) -> None:
        """Append one event. ``kind`` is uppercase (LOCAL_TREE_MATCH,
        ADMIT_DECISION, ...). Fields are key=value, value rendered with
        ``repr`` for strings (no spaces) or str for primitives. Never
        raises; best-effort fsync-free.
        """
        if not self.enabled or self._fh is None:
            return
        try:
            parts = [kind, f"pp={self.pp_rank}"]
            for k, v in fields.items():
                if v is None:
                    parts.append(f"{k}=None")
                elif isinstance(v, str):
                    parts.append(f"{k}={v}")
                elif isinstance(v, (list, tuple)):
                    if len(v) > 8:
                        parts.append(
                            f"{k}={','.join(str(x) for x in list(v)[:8])}+{len(v) - 8}more"
                        )
                    else:
                        parts.append(f"{k}={','.join(str(x) for x in v)}")
                else:
                    parts.append(f"{k}={v}")
            line = " ".join(parts) + "\n"
            with self._lock:
                self._fh.write(line)
        except Exception:  # noqa: BLE001 — diag must never crash scheduler
            pass

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
        self._fh = None


# Module-level singleton; populated by Scheduler.__init__ via ``init_logger``.
# Read from any tree_cache / scheduler call site via ``get_logger``.
_LOGGER: Optional[PPAdmitDiagLogger] = None


def init_logger(pp_rank: int, tp_rank: int) -> PPAdmitDiagLogger:
    """Construct and install the singleton per-rank logger."""
    global _LOGGER
    _LOGGER = PPAdmitDiagLogger(pp_rank, tp_rank)
    return _LOGGER


def get_logger() -> Optional[PPAdmitDiagLogger]:
    """Returns the installed logger, or ``None`` when diag is disabled
    or before init runs. Callers should check for ``None`` and skip.
    """
    return _LOGGER


def hash_token_ids(token_ids: Sequence[int], take: int = -1) -> str:
    """8-char hex hash of token sequence, suitable for cross-rank
    "same content?" verification at log scan time. Stable, no
    GPU sync. ``take`` truncates the sequence (-1 = full).
    """
    if not token_ids:
        return "empty"
    try:
        if take > 0 and len(token_ids) > take:
            token_ids = token_ids[:take]
        h = hashlib.md5(repr(list(token_ids)).encode()).hexdigest()[:8]
        return h
    except Exception:  # noqa: BLE001
        return "err"


def short_rid(rid: str, n: int = 12) -> str:
    """Last-N chars of rid, matching crash_diag convention so logs and
    dumps are easy to cross-reference.
    """
    if not rid:
        return "_"
    return rid[-n:] if len(rid) > n else rid
