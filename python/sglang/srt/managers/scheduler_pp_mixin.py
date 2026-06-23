from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj
from sglang.srt.utils.common import get_device_module, is_xpu

logger = logging.getLogger(__name__)

# Optional per-step diagnostic logging for PP-rank scheduling/launch
# state. Enabled via SGLANG_PP_DESYNC_DIAG=1. Per-rank logging only,
# no collectives.
_PP_DESYNC_DIAG = os.environ.get("SGLANG_PP_DESYNC_DIAG", "0") not in (
    "0",
    "",
    "false",
    "False",
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _pp_desync_log_schedule(self: "Scheduler", mb_id: int, batch) -> None:
    """Log per-rank scheduling result so cross-rank comparison can spot
    a shape-mismatch about to happen (different extend_seq_lens for the
    same rid across ranks).
    """
    pp = self.ps.pp_rank
    tp = self.ps.tp_rank
    step = self.forward_ct
    # Snapshot tree-match consensus state so a crash pins the moment
    # a rank's view diverges.
    tc = getattr(self, "tree_cache", None)
    g_keys = []
    g_size = 0
    l_size = 0
    if tc is not None:
        gset = getattr(tc, "_agreed_tree_match_lens", None)
        if gset is not None:
            g_size = len(gset)
            try:
                g_keys = sorted(gset.keys()) if isinstance(gset, dict) else sorted(gset)
                g_keys = [k[-8:] for k in g_keys[:8]]
            except Exception:
                g_keys = []
    chunked_rid = self.chunked_req.rid[-8:] if self.chunked_req is not None else None
    if batch is None:
        logger.warning(
            "[PP_DESYNC schedule] step=%d mb=%d pp=%d tp=%d batch=None waiting=%d "
            "chunked=%s g_size=%d g_keys=%s l_size=%d",
            step,
            mb_id,
            pp,
            tp,
            len(self.waiting_queue),
            chunked_rid,
            g_size,
            g_keys,
            l_size,
        )
        return
    summary = []
    for r in batch.reqs[:6]:
        prefix = (
            len(r.prefix_indices)
            if getattr(r, "prefix_indices", None) is not None
            else 0
        )
        summary.append(
            (
                r.rid[-8:],
                prefix,
                getattr(r, "host_hit_length", 0),
                getattr(r, "storage_hit_length", 0),
                getattr(r, "extend_input_len", None),
            )
        )
    logger.warning(
        "[PP_DESYNC schedule] step=%d mb=%d pp=%d tp=%d nreq=%d ext_tok=%s "
        "chunked=%s g_size=%d g_keys=%s l_size=%d reqs=%s",
        step,
        mb_id,
        pp,
        tp,
        len(batch.reqs),
        getattr(batch, "extend_num_tokens", "?"),
        chunked_rid,
        g_size,
        g_keys,
        l_size,
        summary,
    )


def _pp_desync_log_launch(self: "Scheduler", mb_id: int, pp_proxy_tensors) -> None:
    """Log what _pp_launch_batch is about to feed into run_batch.

    The crash signature for shape-mismatch is ``cur_ext != ipc_hs[0]``
    -- local positions tensor length doesn't match the IPC
    hidden_states length received from the previous PP rank.
    """
    cur = self.cur_batch
    if cur is None:
        return
    ipc_hs_shape = None
    if pp_proxy_tensors is not None:
        try:
            ipc_hs_shape = tuple(pp_proxy_tensors["hidden_states"].shape)
        except (KeyError, AttributeError):
            ipc_hs_shape = "?"
    input_ids = getattr(cur, "input_ids", None)
    input_ids_shape = tuple(input_ids.shape) if hasattr(input_ids, "shape") else None
    logger.warning(
        "[PP_DESYNC launch] step=%d mb=%d pp=%d tp=%d cur_ext=%s nreq=%d input_ids=%s ipc_hs=%s",
        self.forward_ct,
        mb_id,
        self.ps.pp_rank,
        self.ps.tp_rank,
        getattr(cur, "extend_num_tokens", "?"),
        len(cur.reqs),
        input_ids_shape,
        ipc_hs_shape,
    )


def _crash_diag_record_schedule(self: "Scheduler", mb_id: int, batch) -> None:
    """Append a per-mb_id schedule snapshot to the crash-diag ring.

    Captures the minimum needed to retrace state divergence after a
    crash: req identities + extend lens + prefix lens, chunked_req,
    and the consensus / handoff dict sizes. Cap reqs to 8 for size.
    """
    try:
        cur = batch
        reqs_summary = []
        if cur is not None and hasattr(cur, "reqs"):
            for r in cur.reqs[:8]:
                reqs_summary.append(
                    (
                        r.rid[-12:],
                        getattr(r, "extend_input_len", None),
                        (
                            len(r.prefix_indices)
                            if getattr(r, "prefix_indices", None) is not None
                            else 0
                        ),
                        getattr(r, "storage_hit_length", 0),
                    )
                )
        tc = getattr(self, "tree_cache", None)
        cc = getattr(tc, "cache_controller", None) if tc is not None else None
        chunked = getattr(self, "chunked_req", None)
        self.crash_diag.record(
            "schedule",
            step=getattr(self, "forward_ct", None),
            mb_id=mb_id,
            nreq=(len(cur.reqs) if cur is not None and hasattr(cur, "reqs") else 0),
            ext_tok=(
                getattr(cur, "extend_num_tokens", None) if cur is not None else None
            ),
            chunked_rid=(chunked.rid[-12:] if chunked is not None else None),
            waiting=len(getattr(self, "waiting_queue", [])),
            agreed_tree=(len(getattr(tc, "_agreed_tree_match_lens", {})) if tc else 0),
            handoff_inflight=(len(getattr(tc, "_handoff_in_flight", {})) if tc else 0),
            ack_load=(len(getattr(cc, "ack_load_queue", [])) if cc else 0),
            ack_write=(len(getattr(cc, "ack_write_queue", [])) if cc else 0),
            ongoing_load_back=(len(getattr(tc, "ongoing_load_back", {})) if tc else 0),
            ongoing_write_through=(
                len(getattr(tc, "ongoing_write_through", {})) if tc else 0
            ),
            reqs=reqs_summary,
        )
    except Exception:  # noqa: BLE001 — diag must never crash scheduler
        pass


def _crash_diag_record_launch(self: "Scheduler", mb_id: int, pp_proxy_tensors) -> None:
    """Append a per-mb_id launch snapshot to the crash-diag ring.

    Records the local cur_ext and the IPC hidden_states shape so the
    cur_ext != ipc_hs[0] crash class is directly visible in the dump.
    """
    try:
        cur = self.cur_batch
        if cur is None:
            return
        ipc_hs_shape = None
        if pp_proxy_tensors is not None:
            try:
                ipc_hs_shape = tuple(pp_proxy_tensors["hidden_states"].shape)
            except (KeyError, AttributeError):
                ipc_hs_shape = None
        self.crash_diag.record(
            "launch",
            step=getattr(self, "forward_ct", None),
            mb_id=mb_id,
            cur_ext=getattr(cur, "extend_num_tokens", None),
            nreq=len(cur.reqs) if hasattr(cur, "reqs") else 0,
            ipc_hs=ipc_hs_shape,
        )
    except Exception:  # noqa: BLE001
        pass


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool


class SchedulerPPMixin:
    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.request_receiver.recv_requests()
                    self.process_input_requests(recv_reqs)
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        self.send_req_work = self._pp_send_pyobj_to_next_stage(
                            recv_reqs,
                            async_send=True,
                        )
                with torch.profiler.record_function("get_next_batch_to_run"):
                    self.mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch
                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                            )

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        self.init_pp_loop_state()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []

        # Tree-match consensus ring (per-iter, mirrors bootstrap/release
        # ring shape). All ranks must compute the same flag value or
        # the ring desyncs (mismatched send/recv counts).
        l3_consensus_ring_enabled = self.enable_hicache_storage and hasattr(
            self.tree_cache, "_agreed_tree_match_lens"
        )
        pmbs: List[Optional[Dict[str, int]]] = [None] * self.pp_loop_size
        consensus_tree_match_rids: Optional[Dict[str, int]] = None
        if l3_consensus_ring_enabled:
            logger.info(
                "[hicache] tree-match consensus ring enabled (pp_size=%d, pp_loop_size=%d). "
                "Admit-time prefix length is PP-MIN-truncated to keep extend_input_len "
                "consistent across ranks despite per-rank tree state divergence.",
                self.ps.pp_size,
                self.pp_loop_size,
            )

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                next_consensus_tree_match_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # PHASE A intersect. When the L3 ring is enabled, bootstrap
                # and prefetch-done are piggybacked onto a SINGLE P2P recv
                # (saves one round-trip per mb_id; see
                # ``_pp_pd_get_bootstrap_and_l3_done_ids``). The send
                # below at SEND TO NEXT also carries both payloads.
                # When disabled, fall back to the original single-payload
                # bootstrap path; prefetch-done state is unused.
                # prefetch_done is a dict[rid, effective_hit_tokens] -- the
                # length is part of the consensus payload because PP
                # ranks may observe different token counts under timeout
                # policy and we MIN them to a single agreed length.
                tree_match_rids: Dict[str, int] = {}
                if l3_consensus_ring_enabled:
                    bootstrapped_rids, tree_match_rids = (
                        self._pp_pd_get_bootstrap_and_l3_done_ids()
                    )
                    pmbs[mb_id] = tree_match_rids
                else:
                    bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk()
                batch = self.get_new_batch_prefill()
                if _PP_DESYNC_DIAG:
                    _pp_desync_log_schedule(self, mb_id, batch)
                # Append per-mb_id schedule snapshot to the crash-diag
                # ring (~100us/call, bounded by the deque's maxlen).
                if self.crash_diag.enabled:
                    _crash_diag_record_schedule(self, mb_id, batch)
                batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()

                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                # PHASE B forward consensus. When L3 ring is enabled, use
                # the combined helper that piggybacks the prefetch-done
                # consensus onto the bootstrap consensus's P2P send (one
                # send instead of two). When disabled, the L3 ring is
                # absent so the original separate helper is used.
                if l3_consensus_ring_enabled:
                    (
                        send_consensus_bootstrapped_work,
                        consensus_bootstrapped_rids,
                        consensus_tree_match_rids,
                    ) = self._pp_pd_send_consensus_bootstrap_and_l3_done_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                        consensus_tree_match_rids,
                        tree_match_rids,
                    )
                else:
                    send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                        self._pp_pd_send_consensus_bootstrapped_ids(
                            bmbs,
                            next_first_rank_mb_id,
                            consensus_bootstrapped_rids,
                            bootstrapped_rids,
                        )
                    )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if bmbs[next_mb_id] is not None:
                    # Combined recv when the consensus ring is enabled:
                    # payload is (bootstrapped_rids, tree_match_dict).
                    # tree_match_dict is rid -> PP-MIN page-aligned
                    # match_prefix length. Legacy 3-tuple payloads
                    # (with a removed prefetch_done dict in the middle)
                    # are accepted for rolling-deploy compatibility.
                    raw = self._pp_recv_pyobj_from_prev_stage()
                    if l3_consensus_ring_enabled:
                        if len(raw) == 3:
                            (
                                next_consensus_bootstrapped_rids,
                                _legacy_prefetch_done,
                                next_tree_match_payload,
                            ) = raw
                        else:
                            (
                                next_consensus_bootstrapped_rids,
                                next_tree_match_payload,
                            ) = raw
                        if next_tree_match_payload:
                            tc = self.tree_cache
                            try:
                                tc.apply_tree_match_consensus(next_tree_match_payload)
                            except AttributeError:
                                # Older tree_cache without the API;
                                # admit truncation skips silently.
                                pass
                            next_consensus_tree_match_rids = next_tree_match_payload
                            try:
                                from sglang.srt.managers.scheduler_components.pp_admit_diag import (
                                    get_logger as _pp_admit_get_logger,
                                )

                                _diag = _pp_admit_get_logger()
                                if _diag is not None and _diag.enabled:
                                    items = list(next_tree_match_payload.items())
                                    sample = ",".join(
                                        f"{rid[-12:]}:{v}" for rid, v in items[:8]
                                    )
                                    _diag.log(
                                        "PHASE_B_IN",
                                        step=getattr(self, "forward_ct", None),
                                        mb_id=mb_id,
                                        n_tree_match=len(next_tree_match_payload),
                                        sample=sample if sample else "empty",
                                    )
                            except Exception:
                                pass
                        else:
                            next_consensus_tree_match_rids = {}
                    else:
                        next_consensus_bootstrapped_rids = raw
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    # When L3 ring is enabled, piggyback prefetch_done into
                    # the bootstrapped send -- ONE P2P send carries both,
                    # paired with the combined recv at PHASE A intersect.
                    # When disabled, send raw bootstrapped_rids exactly
                    # like the original code path.
                    if l3_consensus_ring_enabled:
                        bootstrap_payload = (
                            bootstrapped_rids,
                            tree_match_rids,
                        )
                        try:
                            from sglang.srt.managers.scheduler_components.pp_admit_diag import (
                                get_logger as _pp_admit_get_logger,
                            )

                            _diag = _pp_admit_get_logger()
                            if _diag is not None and _diag.enabled:
                                items = list(tree_match_rids.items())
                                sample = ",".join(
                                    f"{rid[-12:]}:{v}" for rid, v in items[:8]
                                )
                                _diag.log(
                                    "PHASE_A_OUT",
                                    step=getattr(self, "forward_ct", None),
                                    mb_id=mb_id,
                                    n_tree_match=len(tree_match_rids),
                                    sample=sample if sample else "empty",
                                )
                        except Exception:
                            pass
                    else:
                        bootstrap_payload = bootstrapped_rids
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrap_payload, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids
                if l3_consensus_ring_enabled:
                    consensus_tree_match_rids = next_consensus_tree_match_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        self.init_pp_loop_state()

        # PD additional state initialization
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # reaching consensus through PP ranks
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work)

                # get batch to run and proxy tensors if needed
                batch = self.get_next_disagg_decode_batch_to_run()
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not self.cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()

                # early send output if possible
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)

                if self.cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch and not self.cur_batch.forward_mode.is_prebuilt():
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.on_idle()

    def init_pp_loop_state(self: Scheduler):
        self.pp_loop_size: int = self.ps.pp_size + self.server_args.pp_async_batch_depth
        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
        self.require_attn_tp_allgather = (
            not self.server_args.enable_nsa_prefill_context_parallel
        )
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]] = deque()

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.launch_event = None
        self._pp_tensor_dict_inbox: Dict[str, deque[Dict[str, torch.Tensor]]] = (
            defaultdict(deque)
        )

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            model_runner = self.tp_worker.model_runner
            model_config = model_runner.model_config
            input_ids_list = []
            for i in range(128):
                chunk_size = int(
                    self.chunked_prefill_size * 1.25
                    - i * (self.chunked_prefill_size * 1.25 // 128)
                )
                if chunk_size <= 0:
                    break
                input_ids = np.random.randint(
                    0, 10000, size=chunk_size, dtype=np.int64
                ).tolist()
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            # Create and profile requests
            for i, input_ids in enumerate(
                tqdm(
                    input_ids_list,
                    desc="Profiling prefill latency for dynamic chunking",
                )
            ):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.fill_ids = req.origin_input_ids
                req.logprob_start_len = -1
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = len(req.fill_ids)

                if is_dp_attention_enabled():
                    # For profiling, we only have one request on PP0
                    # Set global_num_tokens to indicate this rank has tokens, others have 0
                    dp_size = get_attention_dp_size()
                    global_num_tokens = [0] * dp_size
                    dp_rank = get_attention_dp_rank()
                    global_num_tokens[dp_rank] = current_seq_len
                    batch.global_num_tokens = global_num_tokens
                    batch.global_num_tokens_for_logprob = global_num_tokens

                hs = (
                    getattr(model_config, "hc_hidden_size", None)
                    or model_config.hidden_size
                )
                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, hs),
                        dtype=model_config.dtype,
                        device=self.device,
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device=self.device,
                    ),
                }

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with device synchronization for accurate timing
                device_module = get_device_module()
                # Synchronize before starting timing to ensure clean measurement
                device_module.synchronize()

                start = time.perf_counter()
                batch.prepare_for_extend()

                forward_batch = ForwardBatch.init_new(batch, model_runner)
                set_is_extend_in_batch(batch.forward_mode.is_extend())

                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                device_module.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV cache
                if req.req_pool_idx is not None:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(req.fill_ids)
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    self.req_to_token_pool.free(req)

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

            if self.ps.attn_tp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
                seq_lens, latencies = data_to_sync_tp

            if self.ps.attn_cp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # Quadratic model: f(l) = al^2 + bl + c
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = self.max_prefill_tokens
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    rids_to_check=good_consensus_bootstrapped_rids
                    + bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [[req.rid for req in good_reqs], [req.rid for req in failed_reqs]]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            good_bootstrapped_rids, bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_bootstrapped_rids = list(
                set(prev_good_bootstrapped_rids) & set(curr_good_bootstrapped_rids)
            )
            bad_bootstrapped_rids = list(
                set(prev_bad_bootstrapped_rids) | set(curr_bad_bootstrapped_rids)
            )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_prefill_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_pd_get_bootstrap_and_l3_done_ids(self: Scheduler):
        """Combined PHASE A intersect for bootstrap + tree-match length.

        Wire format (per non-first rank):
            recv  ([prev_good, prev_bad], prev_tree_match_dict)
            send  ([new_good,  new_bad],  new_tree_match_dict)

        tree_match is ``dict[rid, page_aligned_match_len]`` for every
        rid in the local waiting queue, intersected by PP-MIN. The
        agreed length is consumed at admit time to truncate
        ``req.prefix_indices`` so all ranks use the same
        ``extend_input_len``. Without this, per-rank tree state
        divergence (eviction asymmetry, async L3 inserts) yields
        per-rank prefix lengths -> kernel shape mismatch.
        """
        local_good, local_bad = self.get_rids(
            self.disagg_prefill_bootstrap_queue.queue,
            True,
            [KVPoll.WaitingForInput],
            [KVPoll.Failed],
        )
        local_tree_match = self._compute_local_tree_match_lens()

        if self.pp_group.is_first_rank:
            return [local_good, local_bad], local_tree_match

        payload = self._pp_recv_pyobj_from_prev_stage()
        # Backward-compat: accept a legacy 3-tuple payload (with the
        # removed prefetch_done dict in the middle) during rolling
        # deploys. Only the tree-match field is used now.
        if len(payload) == 3:
            prev_bootstrapped, _legacy_prefetch_done, prev_tree_match = payload
        else:
            prev_bootstrapped, prev_tree_match = payload
        prev_good, prev_bad = prev_bootstrapped

        good = list(set(prev_good) & set(local_good))
        bad = list(set(prev_bad) | set(local_bad))
        # PP-MIN on tree-match length: a rid is in consensus only if
        # every rank reported it; agreed length is the smallest local
        # match any rank returned.
        tree_match = {
            rid: min(prev_tree_match[rid], local_tree_match[rid])
            for rid in (set(prev_tree_match) & set(local_tree_match))
        }
        return [good, bad], tree_match

    def _pp_pd_send_consensus_bootstrap_and_l3_done_ids(
        self: Scheduler,
        bmbs: List[Optional[List[List[str]]]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: Optional[List[List[str]]],
        bootstrapped_rids: List[List[str]],
        consensus_tree_match_rids: Optional[Dict[str, int]] = None,
        tree_match_rids: Optional[Dict[str, int]] = None,
    ):
        """Combined PHASE B forward for bootstrap + tree-match length.

        Single P2P send carries both consensus payloads. Mirrors
        ``_pp_pd_send_consensus_bootstrapped_ids`` exactly for the gate
        (last_rank uses ``bmbs[next_first_rank_mb_id]``; others gate on
        whether they received a consensus tuple last iter).
        """
        if tree_match_rids is None:
            tree_match_rids = {}
        work: List[P2PWork] = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                # last_rank: my PHASE-A intersect is the global consensus
                consensus_bootstrapped_rids = bootstrapped_rids
                consensus_tree_match_rids = tree_match_rids
                payload = (
                    consensus_bootstrapped_rids,
                    consensus_tree_match_rids,
                )
                work = self._pp_send_pyobj_to_next_stage(payload, async_send=True)
        else:
            if consensus_bootstrapped_rids is not None:
                # mid rank: forward what we received last iter. Default to
                # {} rather than [] so the wire type stays a dict.
                payload = (
                    consensus_bootstrapped_rids,
                    (
                        consensus_tree_match_rids
                        if consensus_tree_match_rids is not None
                        else {}
                    ),
                )
                work = self._pp_send_pyobj_to_next_stage(payload, async_send=True)
        return (
            work,
            consensus_bootstrapped_rids,
            consensus_tree_match_rids,
        )

    def _compute_local_tree_match_lens(self: Scheduler) -> Dict[str, int]:
        """Compute per-rid local ``match_prefix`` length for every rid
        in the waiting queue.

        Used as this rank's contribution to the PP-MIN tree-match
        consensus. Each rank builds the same RadixKey for the same
        token content, but local trees may differ in eviction state,
        so match lengths diverge -- intersect MIN ensures every rank
        admits with the same prefix length.

        Contributes for EVERY rid in waiting_queue every mb_id,
        including ones whose ``prefix_indices`` was set on a previous
        deferred admit attempt; otherwise cache-hit rids stuck on
        defer never reach the consensus ring.

        Returns ``dict[rid, device_match_len]`` (page-aligned). Empty
        when the tree cache has no ``match_prefix`` API or the queue
        is empty. Host residency is intentionally excluded -- the
        admit path zeroes ``host_hit_length`` to keep the cross-rank
        load_back path deterministic.
        """
        tc = getattr(self, "tree_cache", None)
        if tc is None or not hasattr(tc, "match_prefix"):
            return {}
        wq = getattr(self, "waiting_queue", None)
        if not wq:
            return {}
        try:
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
            from sglang.srt.mem_cache.radix_cache import RadixKey
        except Exception:  # noqa: BLE001
            return {}
        result: Dict[str, int] = {}
        try:
            from sglang.srt.managers.scheduler_components.pp_admit_diag import (
                get_logger as _pp_admit_get_logger,
            )
            from sglang.srt.managers.scheduler_components.pp_admit_diag import (
                hash_token_ids as _pp_admit_hash,
            )
            from sglang.srt.managers.scheduler_components.pp_admit_diag import (
                short_rid as _pp_admit_short_rid,
            )

            _diag = _pp_admit_get_logger()
        except Exception:
            _diag = None
        for req in wq:
            try:
                fill_ids = list(req.origin_input_ids) + list(req.output_ids)
                if not fill_ids:
                    result[req.rid] = 0
                    continue
                # Cap at input_len-1 so we don't claim the final token
                # as a cache hit (matches _compute_max_prefix_len in
                # schedule_batch.py).
                token_ids_to_match = fill_ids[: max(0, len(fill_ids) - 1)]
                if not token_ids_to_match:
                    result[req.rid] = 0
                    continue
                match_result = tc.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(
                            token_ids=token_ids_to_match,
                            extra_key=getattr(req, "extra_key", None),
                        ),
                        req=req,
                    )
                )
                dev = match_result.device_indices
                try:
                    n = dev.numel()
                except AttributeError:
                    n = len(dev) if dev is not None else 0
                result[req.rid] = int(n)
                if _diag is not None and _diag.enabled:
                    _diag.log(
                        "LOCAL_TREE_MATCH",
                        step=getattr(self, "forward_ct", None),
                        rid=_pp_admit_short_rid(req.rid),
                        local_len=int(n),
                        input_len=len(token_ids_to_match),
                        token_hash=_pp_admit_hash(token_ids_to_match, take=4096),
                    )
            except Exception:  # noqa: BLE001 — never break the ring
                # On any error, omit the rid from contribution so PHASE A
                # intersect treats it as "not present" (rid won't be in
                # consensus this round; admit on this rank would defer).
                continue
        return result

    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
    ]:
        self._pp_commit_comm_work(work=self.send_output_work)
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = self.ps.attn_dp_rank * self.ps.attn_tp_size
            p2p_work = point_to_point_pyobj(
                data,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                ((self.ps.pp_rank + 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = self.ps.attn_dp_rank * self.ps.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.ps.pp_rank - 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
            )
        else:
            data = None

        if self.ps.attn_tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )

        if self.ps.attn_cp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_cp_group.rank,
                self.attn_cp_cpu_group,
                src=self.attn_cp_group.ranks[0],
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
        msg_type: str = "default",
    ):
        # Warn once if using default untyped messages
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        p2p_work = []
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                ),
                async_send=async_send,
            )
        )
        return p2p_work

    def _pp_recv_typed_dict(
        self: Scheduler,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
    ) -> Dict[str, torch.Tensor]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind.
        """
        if expected_kind in self._pp_tensor_dict_inbox:
            inbox_queue = self._pp_tensor_dict_inbox[expected_kind]
            if inbox_queue:
                return inbox_queue.popleft()

        while True:
            tensor_dict = self.pp_group.recv_tensor_dict(
                all_gather_group=all_gather_group
            )
            received_kind = tensor_dict.get("__msg_type__", "default")
            if received_kind == expected_kind:
                if received_kind == "default":
                    logger.warning_once(
                        f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                        "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
                    )
                return tensor_dict
            else:
                logger.debug(
                    f"PP recv: expected {expected_kind}, got {received_kind}, stashing"
                )
                self._pp_tensor_dict_inbox[received_kind].append(tensor_dict)

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self._pp_recv_typed_dict(
                    expected_kind="proxy",
                    all_gather_group=(
                        self.attn_tp_group if self.require_attn_tp_allgather else None
                    ),
                )
            )
        return pp_proxy_tensors

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        return self._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        batch.output_ids = pp_outputs["next_token_ids"]
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
        )
        return output_result

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque,
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            if mbs[next_first_rank_mb_id] is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if not mbs[next_first_rank_mb_id].forward_mode.is_prebuilt():
                    self.device_module.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                            msg_type="output",
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                        msg_type="output",
                    )
        return send_output_work

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # On CUDA, isend is async: it enqueues to the stream and returns,
        # so every rank can send first safely. On some backends isend is
        # effectively blocking and does not return until the peer posts a
        # matching recv; if every PP rank sends first, all ranks block
        # waiting for a receiver and the ring deadlocks. Order send/recv
        # by pp_rank parity (even: send->recv, odd: recv->send) so each
        # adjacent pair has one sender and one receiver posted at the
        # same time.

        # CUDA: send first
        # XPU: even ranks send first, odd ranks recv first.
        send_first = (not is_xpu()) or ((self.ps.pp_rank % 2) == 0)

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            if mbs[next_mb_id] is None or mbs[next_mb_id].forward_mode.is_prebuilt():
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    mbs[next_mb_id], mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        if send_first:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        if _PP_DESYNC_DIAG:
            _pp_desync_log_launch(self, mb_id, pp_proxy_tensors)
        if self.crash_diag.enabled:
            _crash_diag_record_launch(self, mb_id, pp_proxy_tensors)
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    self.cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(self.cur_batch, pp_proxy_tensors)
                set_time_batch(
                    self.cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, self.cur_batch)
                            ),
                        )
                    )
        return result, event

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                rids_to_check=good_consensus_prealloc_rids
                + bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c (or bl + c for linear)"""
            return (
                self.quadratic_coeff_a * l * l
                + self.linear_coeff_b * l
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
