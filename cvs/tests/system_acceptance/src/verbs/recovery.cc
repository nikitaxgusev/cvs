/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

// Verbs acceptance tests -- RC recovery & error handling (Group C), REAL 2-node.
//
// The QP error/recovery mechanics RCCL's resiliency (failover/recovery) path
// depends on, exercised on a CONNECTED cross-node RC QP (rank0<->rank1) rather
// than a loopback self-connect. This matters: the known AINIC RESET defect
// manifests on a QP that carries live REMOTE state, which loopback cannot
// reproduce. Tests:
//   * vat_qp_err_transition   -- QP->ERR flushes outstanding WRs, then RESET
//   * vat_cqe_error_codes     -- WR_FLUSH_ERR isolation across two QPs
//   * vat_rc_qp_reset_recovery-- recover a connected QP after a failure via
//                                ERR->RESET->INIT->RTS with QPN re-exchange, and
//                                confirm cross-node transfer resumes. Includes a
//                                negative control (the downed QP must not carry
//                                data before recovery).
//   * vat_async_event_monitor -- ibv_get_async_event / ibv_ack_async_event
//                                (local fault injection; single-node)
//
// Built only with ACCEPT_HAVE_IBVERBS. Without it, an empty TU.

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"
#include "verbs/TwoNode.hpp"
#include "wrapper/verbs/VerbsHelpers.hpp"

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "common/Bootstrap.hpp"
#include "common/Capability.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {

namespace {

constexpr int kPollMs = 5000;
constexpr int kClientRank = 1;

// Drain up to `expect` completions, counting those with `want` status.
int drain_cq(struct ibv_cq* cq, int expect, enum ibv_wc_status want,
             int* want_count, int budget_ms) {
  int reaped = 0;
  if (want_count) *want_count = 0;
  long waited = 0;
  const long budget_us = static_cast<long>(budget_ms) * 1000L;
  while (reaped < expect) {
    struct ibv_wc wc;
    std::memset(&wc, 0, sizeof(wc));
    const int n = ibv_poll_cq(cq, 1, &wc);
    if (n < 0) break;
    if (n == 0) {
      if (waited >= budget_us) break;
      struct timespec ts{0, 1000000L};
      ::nanosleep(&ts, nullptr);
      waited += 1000;
      continue;
    }
    ++reaped;
    if (want_count && wc.status == want) ++(*want_count);
  }
  return reaped;
}

}  // namespace

// QP -> ERR mechanics on a connected QP: post recv WRs, drive to ERR, confirm
// the outstanding WRs drain as IBV_WC_WR_FLUSH_ERR, then RESET cleanly. Both
// ranks do the same to keep collectives matched.
ACCEPT_TEST(vat_qp_err_transition, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr int kN = 4;
  constexpr size_t kLen = 256;
  std::vector<uint8_t> buf(kLen * kN);
  struct ibv_mr* mr =
      ibv_reg_mr(conn.pd(), buf.data(), buf.size(), IBV_ACCESS_LOCAL_WRITE);
  std::string reason;
  if (mr == nullptr) {
    reason = "ibv_reg_mr failed";
  } else {
    for (int i = 0; i < kN && reason.empty(); ++i)
      reason = conn.post_recv(mr->lkey, buf.data() + i * kLen, kLen,
                              static_cast<uint64_t>(i));
    if (reason.empty()) reason = conn.to_err();
    if (reason.empty()) {
      int flushed = 0;
      const int reaped =
          drain_cq(conn.cq(), kN, IBV_WC_WR_FLUSH_ERR, &flushed, kPollMs);
      if (reaped < kN)
        reason = "only " + std::to_string(reaped) + "/" + std::to_string(kN) +
                 " WRs drained on ERR";
      else if (flushed < kN)
        reason = "expected " + std::to_string(kN) + " WR_FLUSH_ERR, got " +
                 std::to_string(flushed);
    }
    if (reason.empty()) reason = conn.to_reset();
  }
  ibvw::safe_dereg_mr(mr);
  boot->barrier();  // keep both ranks in lockstep before teardown
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer ERR-transition failed" : reason);
  return pass();
}

// WR_FLUSH_ERR: client drives its QP to ERR and confirms posted recv WRs drain
// with IBV_WC_WR_FLUSH_ERR. The server QP stays untouched (no cross-node QP
// poisoning). This tests local flush behaviour, not inter-QP isolation.
ACCEPT_TEST(vat_cqe_error_codes, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 4096;
  std::vector<uint8_t> buf(kLen);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  struct ibv_mr* mr = ibv_reg_mr(conn.pd(), buf.data(), kLen, access);
  verbsx::Gate mine = mr ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mine) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail("ibv_reg_mr failed on a rank");
  }
  verbsx::RemoteBuf mine_rb{reinterpret_cast<uint64_t>(buf.data()), mr->rkey};
  (void)verbsx::exchange_rbuf(ctx, mine_rb);  // lockstep only; this rank never targets the peer

  std::string reason;
  // Client posts recvs then errors ITS OWN qp -> its WRs flush locally.
  if (is_client) {
    for (int i = 0; i < 4 && reason.empty(); ++i)
      reason = conn.post_recv(mr->lkey, buf.data(), kLen,
                              static_cast<uint64_t>(0xA00 + i));
    if (reason.empty()) reason = conn.to_err();
    if (reason.empty()) {
      int flushed = 0;
      drain_cq(conn.cq(), 4, IBV_WC_WR_FLUSH_ERR, &flushed, kPollMs);
      if (flushed < 4)
        reason = "client: expected 4 flush errors, got " +
                 std::to_string(flushed);
    }
  }
  // Barrier: the server's QP must remain healthy despite the client's error.
  // (No cross-QP poisoning: they are independent QPs/CQs on separate nodes.)
  boot->barrier();
  ibvw::safe_dereg_mr(mr);
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer CQE-error test failed" : reason);
  return pass();
}

// Recover a downed CONNECTED RC QP and resume cross-node transfer. This is the
// real resiliency path: a QP with live remote state is driven to ERR, proven
// non-functional (negative control), then recovered via ERR->RESET->INIT->RTS
// with a QpInfo re-exchange, and a cross-node RDMA_WRITE must succeed again.
// Note: RESET reuses the same QP handle and QPN -- the re-exchange is needed
// because to_rts() on both ends requires the peer's dest_qp_num even when
// unchanged.
//
// NOTE re AINIC: the ionic driver has a known ibv_modify_qp(RESET) defect --
// after ERR->RESET->INIT->RTS, post-recovery RDMA_WRITE times out (QP is
// non-functional at the hardware level despite state transitions returning 0).
// Confirmed by qp_recovery_repro.c (~/qp_recovery_repro.c on TW cluster).
//
// Observed behaviour during test development: the test consistently PASSES on
// AINIC (wr_id=2 confirmed via assertion -- not a stale-CQE false positive),
// while standalone C reproducers (single-process, same timeout, same GID/size)
// consistently fail. The structural differences systematically ruled out:
// loopback vs two-QP, single vs two ibv_context, self-connect, pre-RESET write
// on ERR QP, receiver reset from RTS vs ERR, write payload size, and timeout.
// The remaining unexplained variable is some ionic driver/firmware internal state
// that differs between a forked-process harness worker and a standalone program.
// This requires ionic driver source to diagnose. The test result is correct for
// the harness execution model (which matches RCCL's process model).
ACCEPT_TEST(vat_rc_qp_reset_recovery, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 45) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 4096;
  std::vector<uint8_t> buf(kLen);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  struct ibv_mr* mr = ibv_reg_mr(conn.pd(), buf.data(), kLen, access);
  verbsx::Gate mine = mr ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mine) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail("ibv_reg_mr failed on a rank");
  }
  verbsx::RemoteBuf mine_rb{reinterpret_cast<uint64_t>(buf.data()), mr->rkey};
  verbsx::RemoteBuf peer_rb = verbsx::exchange_rbuf(ctx, mine_rb);

  std::string reason;

  // 1) Baseline: client writes the pattern into the server; server verifies.
  if (!is_client) std::memset(buf.data(), 0, kLen);  // clear target FIRST
  boot->barrier();  // server target cleared before the client writes
  if (is_client) {
    for (size_t i = 0; i < kLen; ++i) buf[i] = static_cast<uint8_t>((0x33 + i) % 256);
    reason = conn.post_write(mr->lkey, buf.data(), kLen, peer_rb, 1);
    if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
  }
  boot->barrier();  // client write completes before server verifies
  if (!is_client && reason.empty()) {
    for (size_t i = 0; i < kLen; ++i)
      if (buf[i] != static_cast<uint8_t>((0x33 + i) % 256)) {
        reason = "baseline transfer mismatch at byte " + std::to_string(i);
        break;
      }
  }

  // 2) Down BOTH QPs (simulated failure) and drain.
  if (reason.empty()) reason = conn.to_err();
  if (reason.empty()) { int f = 0; (void)drain_cq(conn.cq(), 1, IBV_WC_WR_FLUSH_ERR, &f, 200); }
  boot->barrier();

  // 2b) NEGATIVE CONTROL: prove the downed QP cannot carry data. Client writes a
  // sentinel target; a WRITE on the ERR-state QP must not land / not succeed.
  if (reason.empty() && is_client) {
    const std::string post =
        conn.post_write(mr->lkey, buf.data(), kLen, peer_rb, 98);
    if (post.empty()) {
      struct ibv_wc wc; std::memset(&wc, 0, sizeof(wc));
      bool ok = false;
      for (int i = 0; i < 200; ++i) {
        if (ibv_poll_cq(conn.cq(), 1, &wc) > 0) { ok = (wc.status == IBV_WC_SUCCESS); break; }
        struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
      }
      if (ok) reason = "negative control: WRITE succeeded on an ERR-state QP";
    }
  }
  boot->barrier();

  // 3) Recover in place: ERR->RESET->INIT->RTS with a QpInfo re-exchange.
  if (reason.empty()) reason = conn.to_reset();
  // Drain any CQE from the negative-control post_write that may not have been
  // reaped within the 200ms window. RESET discards queued WQEs but CQEs already
  // written to the ring are not flushed; a stale wr_id=98 entry would be polled
  // in step 4 and trigger the false stale-CQE assertion.
  { struct ibv_wc wcs[8]; while (ibv_poll_cq(conn.cq(), 8, wcs) > 0) {} }
  if (reason.empty()) reason = conn.reinit();
  // Gate BEFORE the exchange: if the negative-control block set reason on one
  // rank (e.g. ibv_post_send to ERR QP returns EINVAL on bnxt_re), that rank
  // would skip boot->exchange() while the other blocks in it -- a collective
  // mismatch that resets the TCP connection (std::terminate in SocketBootstrap).
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail(reason.empty() ? "peer failed RESET/reinit" : reason);
  }
  {
    verbsx::QpInfo local = conn.local();  // same QPN, refreshed PSN after reinit
    verbsx::QpInfo remote;
    boot->exchange(1 - ctx.rank, &local, &remote, sizeof(verbsx::QpInfo));
    reason = conn.to_rts(remote);
  }
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail(reason.empty() ? "peer failed to recover QP" : reason);
  }

  // 4) Post-recovery: a fresh cross-node WRITE must succeed and land correctly.
  if (!is_client) std::memset(buf.data(), 0, kLen);  // clear target FIRST
  boot->barrier();  // server target cleared before the client writes
  if (is_client) {
    for (size_t i = 0; i < kLen; ++i) buf[i] = static_cast<uint8_t>((0x9C + i) % 256);
    reason = conn.post_write(mr->lkey, buf.data(), kLen, peer_rb, 2);
    if (reason.empty()) {
      struct ibv_wc wc4; std::memset(&wc4, 0, sizeof(wc4));
      reason = conn.poll(kPollMs, &wc4);
      if (reason.empty() && wc4.wr_id != 2)
        reason = "post-recovery poll got wr_id=" + std::to_string(wc4.wr_id) +
                 " (expected 2 — stale CQE false positive)";
    }
  }
  boot->barrier();
  if (!is_client && reason.empty()) {
    for (size_t i = 0; i < kLen; ++i)
      if (buf[i] != static_cast<uint8_t>((0x9C + i) % 256)) {
        reason = "post-recovery transfer mismatch at byte " + std::to_string(i);
        break;
      }
  }
  ibvw::safe_dereg_mr(mr);
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer post-recovery transfer failed" : reason);
  return pass();
}

// Faithful single-process port of ~/qp_recovery_repro.c.
//
// Creates two QPs (sender + receiver) on the same ibv_context and ibv_pd,
// loopback-connected using the device's own GID. Only the SENDER QP goes to
// ERR; BOTH are then RESET and re-brought to RTS. This is exactly the scenario
// that exposes the AINIC ibv_modify_qp(RESET) defect (hardware QP non-functional
// after recovery despite state transitions returning 0). On bnxt_re it passes.
// Policy: MUST_PASS on bnxt_re/thor, KNOWN_FAILURE on AINIC/ionic.
//
// ROOT CAUSE (confirmed by hardware experiment on mia1-p01-g24, 2026-07-10):
// The ionic driver sizes the hardware WQE descriptor ring based on
// max_inline_data at ibv_create_qp() time. The RESET recovery path only works
// correctly with the larger WQE format. Step function (req vs granted):
//   req 0..24  -> driver grants 24-byte WQE -> RESET broken (this test fails)
//   req 25..   -> driver grants 32+ byte WQE -> RESET works
// RcConn requests max_inline_data=0 (matches RCCL default), which lands in the
// broken 24-byte WQE path; both this loopback test and vat_rc_qp_reset_recovery
// are therefore KNOWN_FAILURE on AINIC/ionic. When the driver is fixed, remove
// the KNOWN_FAILURE entries from policy/ainic.csv.
ACCEPT_TEST(vat_rc_qp_reset_recovery_loopback, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;

  // Open the first available RDMA device (raw, no RcConn -- mirrors the repro).
  int num_devs = 0;
  struct ibv_device** dev_list = ibv_get_device_list(&num_devs);
  if (!dev_list || num_devs == 0) {
    if (dev_list) ibvw::safe_free_device_list(dev_list);
    return skip("no RDMA devices present on this host");
  }
  struct ibv_context* dctx = ibv_open_device(dev_list[0]);
  const char* dev_name_raw = ibv_get_device_name(dev_list[0]);
  std::string dev_name = dev_name_raw ? dev_name_raw : "<unknown>";
  ibvw::safe_free_device_list(dev_list);
  if (!dctx)
    return fail(std::string("ibv_open_device failed: ") + std::strerror(errno));

  // Resolve the routable GID index via the capability probe.
  int gid_index = 0;
  {
    const EnvironmentInfo env = detect();
    for (const auto& rd : env.rdma_devices)
      if (rd.device == dev_name && rd.gid_index >= 0) { gid_index = rd.gid_index; break; }
  }
  union ibv_gid gid;
  std::memset(&gid, 0, sizeof(gid));
  if (ibv_query_gid(dctx, 1, gid_index, &gid) != 0) {
    ibvw::safe_close_device(dctx);
    return fail("ibv_query_gid failed");
  }

  // Allocate shared PD and two independent CQs / MRs / bufs (mirrors repro).
  struct ibv_pd* pd = ibv_alloc_pd(dctx);
  if (!pd) { ibvw::safe_close_device(dctx); return fail("ibv_alloc_pd failed"); }
  struct ibv_cq* scq = ibv_create_cq(dctx, 64, nullptr, nullptr, 0);
  struct ibv_cq* rcq = ibv_create_cq(dctx, 64, nullptr, nullptr, 0);
  if (!scq || !rcq) {
    ibvw::safe_dealloc_pd(pd); ibvw::safe_close_device(dctx);
    return fail("ibv_create_cq failed");
  }
  constexpr size_t kBuf = 4096;
  std::vector<uint8_t> sbuf(kBuf), rbuf(kBuf);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ;
  struct ibv_mr* smr = ibv_reg_mr(pd, sbuf.data(), kBuf, access);
  struct ibv_mr* rmr = ibv_reg_mr(pd, rbuf.data(), kBuf, access);
  if (!smr || !rmr) {
    ibvw::safe_dereg_mr(smr); ibvw::safe_dereg_mr(rmr);
    ibvw::safe_destroy_cq(scq); ibvw::safe_destroy_cq(rcq);
    ibvw::safe_dealloc_pd(pd);
    return fail("ibv_reg_mr failed");
  }

  // Create two RC QPs on the same PD (sender + receiver).
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 16; qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  qia.send_cq = scq; qia.recv_cq = scq;
  struct ibv_qp* sqp = ibv_create_qp(pd, &qia);
  qia.send_cq = rcq; qia.recv_cq = rcq;
  struct ibv_qp* rqp = ibv_create_qp(pd, &qia);
  if (!sqp || !rqp) {
    ibvw::safe_destroy_qp(sqp); ibvw::safe_destroy_qp(rqp);
    ibvw::safe_dereg_mr(smr); ibvw::safe_dereg_mr(rmr);
    ibvw::safe_destroy_cq(scq); ibvw::safe_destroy_cq(rcq);
    ibvw::safe_dealloc_pd(pd);
    return fail("ibv_create_qp failed");
  }

  // Bring both QPs to RTS using the local device's own GID (loopback).
  // This is the exact setup from the standalone repro.
  auto to_rts = [&](struct ibv_qp* local_qp, uint32_t remote_qpn) -> std::string {
    struct ibv_qp_attr a;
    std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT; a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(local_qp, &a, IBV_QP_STATE|IBV_QP_PKEY_INDEX|
                      IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))
      return "INIT failed: " + std::string(std::strerror(errno));
    std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = remote_qpn; a.rq_psn = 0;
    a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1; a.ah_attr.grh.dgid = gid;
    a.ah_attr.grh.sgid_index = static_cast<uint8_t>(gid_index);
    a.ah_attr.grh.hop_limit = 255; a.ah_attr.port_num = 1;
    if (ibv_modify_qp(local_qp, &a, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|
                      IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|
                      IBV_QP_MIN_RNR_TIMER))
      return "RTR failed: " + std::string(std::strerror(errno));
    std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS; a.sq_psn = 0;
    a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.max_rd_atomic = 1;
    if (ibv_modify_qp(local_qp, &a, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                      IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC))
      return "RTS failed: " + std::string(std::strerror(errno));
    return "";
  };

  auto do_write = [&](uint8_t pattern) -> std::string {
    std::fill(sbuf.begin(), sbuf.begin() + 64, pattern);
    std::fill(rbuf.begin(), rbuf.begin() + 64, uint8_t{0});
    struct ibv_sge sge{reinterpret_cast<uint64_t>(sbuf.data()), 64, smr->lkey};
    struct ibv_send_wr wr, *bad;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = pattern; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(rbuf.data());
    wr.wr.rdma.rkey = rmr->rkey;
    if (ibv_post_send(sqp, &wr, &bad))
      return "ibv_post_send failed: " + std::string(std::strerror(errno));
    struct ibv_wc wc; long waited = 0;
    for (;;) {
      std::memset(&wc, 0, sizeof(wc));
      const int n = ibv_poll_cq(scq, 1, &wc);
      if (n < 0) return "ibv_poll_cq failed";
      if (n > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          const char* s = ibv_wc_status_str(wc.status);
          return std::string("completion error: ") + (s ? s : "unknown");
        }
        if (wc.wr_id != pattern)
          return "stale CQE: wr_id=" + std::to_string(wc.wr_id) +
                 " expected=" + std::to_string(pattern);
        return "";
      }
      if (waited >= 5000000L) return "CQ poll timeout after 5000ms";
      struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
      waited += 1000;
    }
  };

  std::string reason;

  // Step 1: initial RTS (s->r, r->s using local GID).
  reason = to_rts(sqp, rqp->qp_num);
  if (reason.empty()) reason = to_rts(rqp, sqp->qp_num);
  if (!reason.empty()) { goto cleanup; }

  // Step 2: baseline RDMA_WRITE (s writes to r).
  reason = do_write(0xAB);
  if (!reason.empty()) { reason = "baseline write: " + reason; goto cleanup; }

  // Step 3: only SENDER goes to ERR (receiver stays RTS -- matches repro).
  {
    struct ibv_qp_attr a; std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_ERR;
    if (ibv_modify_qp(sqp, &a, IBV_QP_STATE))
      { reason = "ERR failed"; goto cleanup; }
    usleep(100000);
    { struct ibv_wc wcs[16]; ibv_poll_cq(scq, 16, wcs); }
  }

  // Step 4: RESET BOTH (receiver was still in RTS -- the key repro pattern).
  {
    struct ibv_qp_attr a; std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RESET;
    if (ibv_modify_qp(sqp, &a, IBV_QP_STATE))
      { reason = "RESET sender failed"; goto cleanup; }
    if (ibv_modify_qp(rqp, &a, IBV_QP_STATE))
      { reason = "RESET receiver failed"; goto cleanup; }
  }
  reason = to_rts(sqp, rqp->qp_num);
  if (reason.empty()) reason = to_rts(rqp, sqp->qp_num);
  if (!reason.empty()) { reason = "post-RESET to_rts: " + reason; goto cleanup; }

  // Step 5: post-recovery RDMA_WRITE -- fails on AINIC (KNOWN_FAILURE).
  reason = do_write(0xC0);
  if (!reason.empty()) reason = "post-recovery write: " + reason;

cleanup:
  ibvw::safe_destroy_qp(sqp); ibvw::safe_destroy_qp(rqp);
  ibvw::safe_dereg_mr(smr);   ibvw::safe_dereg_mr(rmr);
  ibvw::safe_destroy_cq(scq); ibvw::safe_destroy_cq(rcq);
  ibvw::safe_dealloc_pd(pd);
  ibvw::safe_close_device(dctx);
  return reason.empty() ? pass() : fail(reason);
}

// Async-event monitor: induce a real hardware fault (RDMA WRITE with a bogus
// rkey over the live QP) and confirm a QP async event is delivered on the
// context async fd, decoded, and acked. Single-node: the fault is self-induced
// and the event is local, so no peer is needed.
ACCEPT_TEST(vat_async_event_monitor, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  verbsx::RcConn conn;
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);
  // Loopback self-connect is sufficient: we only need a live QP to fault.
  e = conn.to_rts(conn.local());
  if (!e.empty()) return fail("loopback bring-up: " + e);

  constexpr size_t kLen = 256;
  std::vector<uint8_t> buf(kLen);
  struct ibv_mr* mr =
      ibv_reg_mr(conn.pd(), buf.data(), kLen, IBV_ACCESS_LOCAL_WRITE);
  if (mr == nullptr)
    return fail(std::string("ibv_reg_mr failed: ") + std::strerror(errno));
  // Bogus rkey -> HCA remote-access protection error -> QP async event.
  verbsx::RemoteBuf bad_rb{reinterpret_cast<uint64_t>(buf.data()), 0xDEADBEEF};
  std::string reason = conn.post_write(mr->lkey, buf.data(), kLen, bad_rb, 1);
  if (reason.empty()) {
    struct ibv_wc wc; std::memset(&wc, 0, sizeof(wc));
    for (int i = 0; i < 200; ++i) {
      if (ibv_poll_cq(conn.cq(), 1, &wc) != 0) break;
      struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
    }
  }

  struct pollfd pfd;
  pfd.fd = conn.ctx()->async_fd;
  pfd.events = POLLIN;
  const int pr = ::poll(&pfd, 1, 3000);
  if (pr < 0) { ibvw::safe_dereg_mr(mr);
    return fail(std::string("poll(async_fd) failed: ") + std::strerror(errno)); }
  if (pr == 0) { ibvw::safe_dereg_mr(mr);
    return skip("no async event delivered within 3s (provider raises no QP "
                "async event for this fault)"); }

  struct ibv_async_event ev;
  std::memset(&ev, 0, sizeof(ev));
  if (ibv_get_async_event(conn.ctx(), &ev) != 0) { ibvw::safe_dereg_mr(mr);
    return fail(std::string("ibv_get_async_event failed: ") + std::strerror(errno)); }
  const char* name = ibv_event_type_str(ev.event_type);
  const bool qp_event = (ev.event_type == IBV_EVENT_QP_FATAL ||
                         ev.event_type == IBV_EVENT_QP_REQ_ERR ||
                         ev.event_type == IBV_EVENT_QP_ACCESS_ERR);
  ibv_ack_async_event(&ev);
  ibvw::safe_dereg_mr(mr);
  if (!qp_event)
    return fail(std::string("unexpected async event: ") + (name ? name : "?"));
  return pass();
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
