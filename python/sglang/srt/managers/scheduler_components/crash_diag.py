"""Lightweight crash-time scheduler state dump + rolling event ring buffer.

Two-part instrument designed for triaging rare PP-sync / hicache /
scheduler races that take hours of stress to reproduce. Negligible
steady-state cost; useful data only materializes on crash.

Part A — rolling event ring (``record``):
    A bounded ``collections.deque`` of (timestamp, kind, fields)
    tuples. Each ``record(kind, **fields)`` call is a Python-level dict
    construction + append, ~100us. Default capacity is 1000 entries
    (a few seconds of mb_id-level activity at typical PP loop rates).
    Fields must be JSON-serializable primitives.

Part B — on-crash snapshot + dump (``dump_on_crash``):
    Wrap the scheduler's run_event_loop in try/except. On exception,
    call ``dump_on_crash`` to write the ring + a fresh state snapshot
    (queue lengths, dict sizes, ongoing trackers) to a JSON file.
    Best-effort: never raises, so it cannot mask the real crash.

The scheduler crash handler at ``run_scheduler_process`` already
catches and logs exceptions; we hook in front of that to capture state
before the process exits.

Disable / configure via env:
    SGLANG_CRASH_DIAG=0          disable entirely (zero overhead)
    SGLANG_CRASH_DUMP_DIR=/path  override default /tmp dump location
    SGLANG_CRASH_DIAG_RING=1000  ring buffer capacity
"""

from __future__ import annotations

import collections
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "", "false", "False", "no", "No")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


CRASH_DIAG_ENABLED = _env_flag("SGLANG_CRASH_DIAG", True)


class SchedulerCrashDiag:
    """See module docstring."""

    def __init__(self, pp_rank: int, tp_rank: int):
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        max_events = _env_int("SGLANG_CRASH_DIAG_RING", 1000)
        self.events: "collections.deque[tuple[float, str, Dict[str, Any]]]" = (
            collections.deque(maxlen=max_events)
        )
        self.start_time = time.time()
        self.dump_dir = os.environ.get("SGLANG_CRASH_DUMP_DIR", "/tmp")
        self.enabled = CRASH_DIAG_ENABLED

    def record(self, kind: str, **fields: Any) -> None:
        """Append one event to the rolling buffer. Cheap (~100us) and
        never raises. Fields should be JSON-serializable primitives;
        any TypeError on dump is silently caught.
        """
        if not self.enabled:
            return
        try:
            self.events.append((time.time(), kind, fields))
        except Exception:  # noqa: BLE001 — diag must never crash scheduler
            pass

    def snapshot_state(self, scheduler: Any) -> Dict[str, Any]:
        """Best-effort snapshot of scheduler + tree_cache key state.

        Captures sizes, not contents, of dicts that may be large. For
        consensus / handoff dicts we sample up to 20 rids so cross-rank
        diff comparisons can pinpoint a single divergent rid.
        """
        s: Dict[str, Any] = {
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
        }
        try:
            s["forward_ct"] = getattr(scheduler, "forward_ct", None)
            s["waiting_queue_len"] = len(getattr(scheduler, "waiting_queue", []))
            chunked = getattr(scheduler, "chunked_req", None)
            s["chunked_req_rid"] = chunked.rid if chunked is not None else None
            s["disagg_prefill_inflight_queue_len"] = len(
                getattr(scheduler, "disagg_prefill_inflight_queue", []) or []
            )
            running_mbs = getattr(scheduler, "running_mbs", None) or []
            s["running_mbs_sizes"] = [
                len(b.reqs) if (b is not None and hasattr(b, "reqs")) else 0
                for b in running_mbs
            ]
        except Exception as e:  # noqa: BLE001
            s["scheduler_snapshot_error"] = repr(e)

        try:
            tc = getattr(scheduler, "tree_cache", None)
            if tc is not None:
                s["tree_cache_class"] = type(tc).__name__
                for attr in (
                    "ongoing_prefetch",
                    "ongoing_load_back",
                    "ongoing_write_through",
                    "ongoing_backup",
                    "_handoff_in_flight",
                    "_local_prefetch_done_rids",
                    "_global_consensus_prefetch_done",
                    "_prefetch_device_indices_by_reqid",
                    "prefetch_loaded_tokens_by_reqid",
                ):
                    val = getattr(tc, attr, None)
                    if val is not None:
                        s[attr] = len(val)
                # Sample rids for the cross-rank diffable dicts.
                for attr in (
                    "_local_prefetch_done_rids",
                    "_global_consensus_prefetch_done",
                ):
                    val = getattr(tc, attr, None)
                    if val:
                        try:
                            keys = (
                                list(val.keys()) if isinstance(val, dict) else list(val)
                            )
                            s[f"{attr}_sample"] = sorted(keys)[:20]
                        except Exception:
                            pass
                cc = getattr(tc, "cache_controller", None)
                if cc is not None:
                    s["ack_load_queue_len"] = len(
                        getattr(cc, "ack_load_queue", []) or []
                    )
                    s["ack_write_queue_len"] = len(
                        getattr(cc, "ack_write_queue", []) or []
                    )
                    s["prefetch_tokens_occupied"] = getattr(
                        cc, "prefetch_tokens_occupied", None
                    )
        except Exception as e:  # noqa: BLE001
            s["tree_cache_snapshot_error"] = repr(e)

        return s

    def dump_on_crash(
        self,
        scheduler: Any,
        reason: str = "scheduler_exception",
        exc: Optional[BaseException] = None,
    ) -> Optional[str]:
        """Write ring + snapshot + traceback to a JSON file under
        ``SGLANG_CRASH_DUMP_DIR`` (default /tmp). Returns the path on
        success, or None on failure. Never raises.
        """
        if not self.enabled:
            return None
        ts_ms = int(time.time() * 1000)
        path = os.path.join(
            self.dump_dir,
            f"sglang_crash_pp{self.pp_rank}_tp{self.tp_rank}_{ts_ms}.json",
        )
        try:
            tb_str: Optional[str] = None
            if exc is not None:
                try:
                    tb_str = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                except Exception:
                    tb_str = repr(exc)
            payload = {
                "reason": reason,
                "exception": repr(exc) if exc is not None else None,
                "traceback": tb_str,
                "uptime_seconds": time.time() - self.start_time,
                "snapshot": self.snapshot_state(scheduler),
                # Most recent last; scan from the bottom for crash-time state.
                "recent_events": [
                    {"t": t, "kind": k, **f} for (t, k, f) in list(self.events)
                ],
                "ring_capacity": self.events.maxlen,
                "ring_size": len(self.events),
            }
            with open(path, "w") as f:
                json.dump(payload, f, default=str, indent=2)
            logger.error(
                "[crash_diag] dumped scheduler state to %s "
                "(events=%d, reason=%s, exc=%s)",
                path,
                len(self.events),
                reason,
                type(exc).__name__ if exc is not None else None,
            )
            return path
        except Exception as e:  # noqa: BLE001
            logger.error("[crash_diag] failed to dump state: %r", e)
            return None
