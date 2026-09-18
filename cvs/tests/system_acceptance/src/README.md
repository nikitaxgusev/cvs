# HIP/Verbs Acceptance Suite

A standalone acceptance-test harness (no RCCL, no GoogleTest, no third-party JSON)
that runs HIP and RDMA/Verbs primitives as a gate before the main test suite and
when validating new hardware, firmware, or a ROCm bump. It answers one question:
**are the low-level GPU + NIC primitives RCCL depends on healthy on this machine?**

New here? Start with the [Quickstart](#quickstart-real-hardware). Bringing the suite
up on a platform for the first time (e.g. MI450)? Read
[docs/NEW_PLATFORM.md](docs/NEW_PLATFORM.md). Adding a test? Read
[docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md).

---

## Quickstart (real hardware)

Prereqs: a C++20 compiler (g++ 13+) and CMake >= 3.20. A ROCm install
(`/opt/rocm`) is required only for the HIP tests (`-DACCEPT_WITH_HIP=ON`) and an
RDMA NIC only for the Verbs tests (`-DACCEPT_WITH_IBVERBS=ON`) — with both flags
off the harness still builds and runs the [synthetic set](#harness-smoke-test-no-gpunic)
(no ROCm needed). Build takes a couple of minutes; runs take seconds to a few
minutes.

### 1. Build

```bash
cd system_acceptance
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DACCEPT_WITH_HIP=ON -DACCEPT_WITH_IBVERBS=ON -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
# produces: build/system_acceptance   and   build/unit_tests
```

Drop `-DACCEPT_WITH_HIP=ON` / `-DACCEPT_WITH_IBVERBS=ON` to compile out the HIP or
Verbs tests (e.g. on a box with no GPU or no NIC). With both off you still get the
harness + [synthetic set](#harness-smoke-test-no-gpunic).

### 2. Run the HIP tests (single node)

```bash
./build/system_acceptance --filter-category 'hip/*' --kind func --report report.json
```

### 3. Run the Verbs tests (two ranks / two nodes)

The RDMA data-path tests need two endpoints. On one host with >=2 NICs, use two
local ranks:

```bash
./build/system_acceptance --local-ranks 2 --filter-category 'verbs/*' \
      --kind func --report report.json
```

Across two real nodes (rank 0 listens on `nodes[0]`, rank 1 is ssh-launched;
binary + policy must be at the **same absolute path** on both, e.g. a shared FS):

```bash
./build/system_acceptance --nodes host1,host2 --filter-category 'verbs/*' \
      --kind func --report report.json
#   [--ssh-user U] [--port P] [--remote-binary PATH] [--launch-timeout S]
```

The full `verbs/*` set includes RDMA data-path tests that need two working NIC
endpoints — run those across two real nodes. On a single host, the
`verbs/hip_mixed` subset (GPU memory registration / GDR paths) is a good local
check: `--local-ranks 2 --filter-category 'verbs/hip_mixed'`.

Swap `--kind func` for `--kind perf` to also collect latency/bandwidth metrics
into `report.json` (see [Performance](#performance)).

### 4. Read the result

Each test streams one line as it finishes; a summary and the `ACCEPTANCE:` verdict
close the run. Example (HIP tests on an MI300X-class box), showing every status:

```
[  PASS  ] Memcpy_Size_Sweep
[  PASS  ] Device_CanAccessPeer_Matrix
[  PASS  ] Device_P2P_AllPairs_Copy
[   NA   ] Hdp_Flush_Ordering  (gfx<90a only; gfx950 CDNA3+)
[   NA   ] Texture_Object_Lifecycle  (no image hw on CDNA3+ (gfx950))
[  SKIP  ] StreamGetId_Stable  (requires hipStreamGetId (ROCm >= 7.1))
[  FAIL  ] Vmm_Map_NoDeadlock_KernelInFlight  (ROCM-25387 deadlock signature)
report: mode=gate | pass=52 fail=2 skip=5 na=3 | flagged=0 | -> report.json
ACCEPTANCE: FAIL — 2 test(s) failed the gate
```

A clean run (here the `verbs/hip_mixed` subset, `--local-ranks 2`) ends:

```
report: mode=gate | pass=17 fail=0 skip=10 na=0 | flagged=0 | -> report.json
ACCEPTANCE: PASS
```

| Status | Meaning | Gate |
|---|---|---|
| `PASS` | test passed | ok |
| `FAIL` | a `MUST_PASS` test failed/was skipped, or a `KNOWN_FAILURE` reproduced | **fails** |
| `SKIP` | software-gated (feature/symbol/ROCm version absent); classified `SKIP` in policy | neutral |
| `NA`   | not applicable on this hardware (permanent); classified `NA` in policy | neutral |

`KNOWN_FAILURE` tests always report `FAIL` on purpose — to keep pressure on the
underlying bug (ADR-007). The gate is green only when nothing `MUST_PASS` fails and
no known bug is present, so `ACCEPTANCE: FAIL` above is the tracked ROCm regression,
not a suite defect.

**Exit codes:** `0` gate pass, `1` gate fail, `2` usage error, `4` launch failure
(ssh/unreachable/timeout).

---

## Parameters

| Flag | Purpose |
|---|---|
| `--filter <glob>` | select tests by id (fnmatch), e.g. `'Memcpy_*'` |
| `--filter-category <glob>` | select by category, e.g. `'hip/*'`, `'verbs/*'` |
| `--filter-class <class>` | select by policy class: `must_pass\|known_failure\|skip\|na` |
| `--kind func\|perf\|all` | `func` (default) runs functional tests; `perf`/`all` also record metrics |
| `--vendor <name>` | override NIC vendor auto-detection (picks `policy/<name>.csv`) |
| `--policy <file>` | use a specific policy CSV directly (highest priority) |
| `--report <file>` | where to write `report.json` (default: CWD) |
| `--local-ranks 2` | spawn two local ranks and reconcile (for `verbs/*` two-node tests) |
| `--nodes h1,h2` | ssh-launch one worker per node and reconcile |
| `--quiet` | print only the summary (suppress per-test lines) |
| `--list` | list all registered tests and exit |
| `--lint` | report registered tests missing from the policy; do not run |

Full list: `./build/system_acceptance --help`.

---

## Bringing up a new platform

When there is **no** `policy/<gpu>_<nic>.csv` (nor fallback `policy/<nic>.csv`) for
the detected platform, the suite runs in
**discovery mode**: every test runs and is treated as `MUST_PASS` with the gate
active. That first run tells you what passes, fails, skips, or is not applicable on
the new hardware — the raw material for authoring a policy. The full step-by-step
(classify results, write the policy, re-run gated) is in
[docs/NEW_PLATFORM.md](docs/NEW_PLATFORM.md).

```bash
# no --policy/--vendor: auto-detects the NIC and runs everything as MUST_PASS
./build/system_acceptance --filter-category 'hip/*' --kind func --report report.json
```

---

## Adding a test

The suite has a shared test skeleton for both HIP and Verbs. See
[docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md) for the template (including how to
wire in perf from the start) and where to register the test in each policy.

---

## Harness smoke test (no GPU/NIC)

The synthetic set exercises every harness behavior — pass/fail/skip/crash/hang/
timeout/corrupt/known-failure/NA and 2-node reconciliation — with no RDMA or GPU,
so it runs anywhere in seconds. Use it to confirm the harness itself is healthy:

```bash
./build/system_acceptance --vendor synthetic --policy policy/synthetic.csv \
      --filter 'syn_*' --report report.json
```

Expected: a per-test line then `ACCEPTANCE: FAIL` (exit 1) — **on purpose**, the
set deliberately contains failing/crash/hang/known-failure fixtures:

```
[  PASS   ] syn_pass
[  FAIL   ] syn_fail  (intentional failure)
[  FAIL   ] syn_crash_segv  (crash)
[  FAIL   ] syn_hang  (timeout)
[  FAIL   ] syn_known_failure_fail  (known failure behavior)
[   NA    ] syn_na  (not applicable on synthetic vendor)
report: mode=gate | pass=1 fail=11 skip=0 na=1 | flagged=0 | -> report.json
ACCEPTANCE: FAIL — 11 test(s) failed the gate
```

The authoritative "harness behaves correctly" check asserts the full matrix:

```bash
./build/unit_tests                                    # unit checks (reduce/classify, policy CSV, ssh builders)
bash scripts/selftest.sh ./build/system_acceptance .  # end-to-end checks against the synthetic oracle
```

The 2-node synthetic tests run on one host with `--local-ranks 2 --filter 'syn2_*'`.

---

## Reference

### Policy file (`policy/<gpu>_<nic>.csv`)

Per-platform "simplified CSV" (auto-selected as `<gpu>_<nic>.csv`, e.g.
`mi350_thor.csv`, with fallback to a bare `<nic>.csv`). Columns: `test,class,when,expected,ticket,reason`
(split on the first 5 commas; `reason` is a free-form remainder that may contain
commas). `#` lines are comments; `test` may be an fnmatch glob. Full guide:
[docs/POLICY_REFERENCE.md](docs/POLICY_REFERENCE.md).

- `class` in `MUST_PASS | NA | SKIP | KNOWN_FAILURE` (unlisted => `UNCLASSIFIED`,
  flagged, gate-neutral).
- `when` (optional) gates the row on the environment: `rocm>=7.1`, `driver<6.0`,
  `gpu<2`, `sym:hipStreamGetId`, `!sym:libionic` (`;`-join = AND).
- `expected` (`KNOWN_FAILURE` only): the expected reduced status (default `fail`).
- A missing policy for the detected platform (`<gpu>_<nic>.csv`, then `<nic>.csv`)
  => **discovery mode** (all tests `MUST_PASS`, gate active) — the seed for a new
  platform file.

### Report (`report.json`)

The machine artifact. Top level: `mode`, `summary` counts, per-node `environment`,
and `tests[]` with `id, category, class, status, reason, ticket, reduced_from,
perf, gate_fail, flagged, note, duration_ms`. Under `--kind perf`, each test's
`perf` map carries `<metric>.{median,min,max,avg,unit,iters,samples}`.

### Further docs

- [docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md) — add a HIP/Verbs test + template.
- [docs/NEW_PLATFORM.md](docs/NEW_PLATFORM.md) — first run on new hardware.
- [docs/POLICY_REFERENCE.md](docs/POLICY_REFERENCE.md) — policy CSV format.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design; [docs/ADRs.md](docs/ADRs.md) — decisions.

### Agentic skills

Repo-local Cursor skills under [.cursor/skills/](.cursor/skills/) wrap the whole
lifecycle — from "what's missing" through implement, run, and explain:

| Skill | Role |
|---|---|
| `extract-hip-requirements` | scan source vs a knowledge base + existing tests → punch-list of missing HIP tests (`proposed_tests.json`) |
| `extract-verbs-requirements` | same for Verbs (`ibv_*`/`rdma_*`/`mlx5dv_*`) |
| `acceptance-add-test` | scaffold a test (from a description **or** a `PROPOSE` item) + policy row |
| `acceptance-run` | build + run the requested tests, report the verdict |
| `acceptance-summarize` | summarize a `report.json` (counts, per-category, failures) |

**How to use them.** You do not invoke skills by name — just describe the task in
Cursor and the matching skill triggers automatically (the wording below is only an
example; any phrasing with the same intent works):

| You want to… | Ask Cursor something like… | Skill that fires |
|---|---|---|
| Find missing tests | "what HIP acceptance tests are still missing here?" | `extract-hip-requirements` |
| Run the suite | "run the HIP and Verbs acceptance tests with perf metrics and give me the verdict" | `acceptance-run` |
| Add a test | "add a HIP test that checks `hipDeviceSynchronize` on an idle device succeeds, with a latency metric" | `acceptance-add-test` |
| Explain results | "summarize the `report.json` from that run and show why the gate failed" | `acceptance-summarize` |

**Full loop (minimal input).** The five skills chain end-to-end — one request such
as *"run the tests, find a missing HIP scenario, implement it, run it, and show me
what changed"* drives the whole pipeline:

```
acceptance-run  →  extract-hip/verbs-requirements  →  acceptance-add-test  →  acceptance-run  →  acceptance-summarize
   baseline            proposed_tests.json               scaffold + policy        re-run             what changed
```

`extract-*` writes a `PROPOSE` list; `acceptance-add-test` consumes one item
(`title`→id, `sequence`/`verify`→body, `class`+`gate`→policy row) and, once
committed, the scenario flips from PROPOSE to COVERED on the next `extract-*` run —
the repo itself is the progress tracker. Each skill just drives the same
`./build/system_acceptance` CLI (and read-only `python3`/`bash` scans), so anything
a skill does you can also run by hand.

### Performance

`--kind perf` records latency/bandwidth per test into `report.json` (no separate
perf tests; perf is folded into the functional tests). Inspect metrics with:

```bash
python3 -c 'import json; d=json.load(open("report.json"));
[print(t["id"], t["perf"]) for t in d["tests"] if t.get("perf")]'
```
