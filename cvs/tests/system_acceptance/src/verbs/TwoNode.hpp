/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#pragma once

// TwoNode -- shared helpers for real cross-node (rank0<->rank1) verbs tests.
// RCCL's datapath is node-to-node RDMA; a loopback self-connect sidesteps real
// addressing, PSN handling, remote protection, and cross-node QP recovery, so
// the datapath/recovery tests connect two ranks over common/Bootstrap and drive
// RDMA between them. These helpers factor out the collective gate + connect dance
// (mirrors verbs/hip_alloc_matrix.cc) so each test body stays about its opcode.
//
// All tests using this are Topology::TwoNode and skip cleanly when run with a
// single rank (needs 2). Built only with ACCEPT_HAVE_IBVERBS.

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

#include "common/Bootstrap.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {
namespace verbsx {

// Collective gate severity (0 ok < 1 skip < 2 fail): combine a per-rank outcome
// into a uniform decision so both ranks take the SAME branch and their collective
// sequences stay matched (an unmatched barrier/exchange deadlocks the pair).
enum Gate : uint8_t { G_OK = 0, G_SKIP = 1, G_FAIL = 2 };

inline Gate gate(Bootstrap* b, Gate mine) {
  assert(b->size() == 2);
  uint8_t s = static_cast<uint8_t>(mine);
  uint8_t all[2] = {0, 0};
  b->allgather(&s, all, 1);
  Gate g = G_OK;
  for (int i = 0; i < b->size(); ++i)
    g = std::max(g, static_cast<Gate>(all[i]));
  return g;
}

// Open a device + RC QP and connect it to the peer (INIT->RTR->RTS), exchanging
// QpInfo over the bootstrap. On success `conn` is at RTS wired to the peer.
// Returns "" on success (both ranks connected); otherwise a reason, with
// *out_skip set when the collective decision was SKIP (no device / no peer).
inline std::string connect_peer(TestContext& ctx, RcConn& conn, bool* out_skip) {
  *out_skip = false;
  Bootstrap* boot = ctx.boot;

  // Phase 1: bring up the QP to INIT.
  Gate mine = G_OK;
  std::string reason;
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  if (!e.empty()) {
    mine = no_dev ? G_SKIP : G_FAIL;
    reason = e;
  } else if (conn.gid_index() < 0) {
    mine = G_SKIP;
    reason = "no routable RoCEv2 GID on " + conn.dev_name();
  }
  if (Gate g = gate(boot, mine); g != G_OK) {
    *out_skip = (g == G_SKIP);
    if (g == G_SKIP) return reason.empty() ? "peer has no RDMA device" : reason;
    return reason.empty() ? "peer QP open failed" : reason;
  }

  // Phase 2: exchange QpInfo with the peer and drive to RTS.
  QpInfo local = conn.local();
  QpInfo remote;
  boot->exchange(1 - ctx.rank, &local, &remote, sizeof(QpInfo));
  e = conn.to_rts(remote);
  Gate cm = e.empty() ? G_OK : G_FAIL;
  if (!e.empty()) reason = "connect: " + e;
  if (Gate g = gate(boot, cm); g != G_OK)
    return reason.empty() ? "peer connect failed" : reason;
  return "";
}

// Exchange a local RemoteBuf (addr+rkey of a registered MR) with the peer so
// each side knows where to target its RDMA ops.
inline RemoteBuf exchange_rbuf(TestContext& ctx, const RemoteBuf& mine) {
  RemoteBuf peer;
  ctx.boot->exchange(1 - ctx.rank, &mine, &peer, sizeof(RemoteBuf));
  return peer;
}

// ScopedMr -- RAII helper that factors out the repeated register/gate/exchange/
// dereg block common to every TwoNode datapath test:
//
//   ScopedMr mr(conn, buf.data(), size, access, boot, ctx);
//   if (!mr) return fail("ibv_reg_mr failed on a rank");
//   // ... use mr.lkey() for local ops, mr.peer_rb() for remote target ...
//
// On construction: registers the MR, does a collective gate so both ranks either
// proceed or both fail, exchanges RemoteBufs. If the gate fails, mr.ok()==false
// and the test body should return immediately. Destructor calls safe_dereg_mr.
class ScopedMr {
 public:
  ScopedMr(RcConn& conn, void* buf, size_t size, int access,
           Bootstrap* boot, TestContext& ctx)
      : mr_(ibv_reg_mr(conn.pd(), buf, size, access)) {
    Gate mine = mr_ ? G_OK : G_FAIL;
    ok_ = (gate(boot, mine) == G_OK);
    if (!ok_) { ibvw::safe_dereg_mr(mr_); mr_ = nullptr; return; }
    RemoteBuf mine_rb{reinterpret_cast<uint64_t>(buf),
                      mr_ ? mr_->rkey : 0u};
    peer_rb_ = exchange_rbuf(ctx, mine_rb);
  }
  ~ScopedMr() { ibvw::safe_dereg_mr(mr_); }
  ScopedMr(const ScopedMr&) = delete;
  ScopedMr& operator=(const ScopedMr&) = delete;

  explicit operator bool() const { return ok_; }
  bool ok() const { return ok_; }
  uint32_t lkey() const { return mr_ ? mr_->lkey : 0u; }
  const RemoteBuf& peer_rb() const { return peer_rb_; }

 private:
  struct ibv_mr* mr_ = nullptr;
  bool           ok_ = false;
  RemoteBuf      peer_rb_{};
};

// gate_return -- evaluate a per-rank outcome and return the collective verdict.
// Usage at the end of a TwoNode test body:
//   return gate_return(boot, reason);
// Returns pass() if all ranks succeed, fail(reason) otherwise.
inline TestResult gate_return(Bootstrap* boot, const std::string& reason) {
  Gate g = gate(boot, reason.empty() ? G_OK : G_FAIL);
  if (g == G_OK) return {.status = Status::Pass};
  return {.status = Status::Fail,
          .reason = reason.empty() ? "peer failed" : reason};
}

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
