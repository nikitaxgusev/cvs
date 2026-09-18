'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import shlex

from cvs.lib import globals

log = globals.log

# report.json contract this wrapper is written against. Bump when the harness
# changes its schema so mismatches surface as a warning instead of silent drift.
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)


def _q(value):
    return shlex.quote(str(value))


def detect_arch_cmd():
    """Shell command that prints the node's primary GPU arch (e.g. ``gfx942``).

    Wrapped in ``sh -c`` so the ``2>/dev/null`` redirect is valid regardless of
    the node's login shell (csh/tcsh reject bare ``2>`` with "Ambiguous output
    redirect").
    """
    return "sh -c \"rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1\""


def build_kernels_cmd(source_dir, arches):
    """Regenerate the multi-arch static-kernel object for ``arches``.

    ``env ARCHES=...`` (not ``ARCHES=... cmd``) so the assignment is valid
    under csh/tcsh login shells as well as POSIX shells.
    """
    arches_str = " ".join(arches)
    return f"cd {_q(source_dir)}/hip/kernel_launch && env ARCHES={_q(arches_str)} bash build_static_kernels.sh"


def cmake_build_cmd(
    source_dir,
    *,
    rocm_path="/opt/rocm",
    build_type="Release",
    with_hip=True,
    with_verbs=True,
):
    """Configure + build the harness; produces ``<source_dir>/build/system_acceptance``."""
    flags = [
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DACCEPT_WITH_HIP={'ON' if with_hip else 'OFF'}",
        f"-DACCEPT_WITH_IBVERBS={'ON' if with_verbs else 'OFF'}",
        f"-DCMAKE_PREFIX_PATH={rocm_path}",
    ]
    return f"cd {_q(source_dir)} && cmake -S . -B build {' '.join(flags)} && cmake --build build -j"


def binary_path(source_dir):
    """Path to the built binary under a harness source tree."""
    return f"{source_dir}/build/system_acceptance"


def build_accept_cmd(
    bin_path,
    *,
    report_path,
    filter_category=None,
    kind="func",
    policy=None,
    vendor=None,
    gid=None,
    device=None,
    quiet=True,
    extra_args=None,
):
    """Build a single-node ``system_acceptance`` invocation for ``orch.exec``.

    The harness fans a single node's tests out itself; here we drive one node.
    The command redirects the harness's own logs to ``<report_path>.log`` and
    then unconditionally ``cat``s ``report_path`` so the caller still receives
    report.json on stdout when the gate exits non-zero (``;`` not ``&&``).

    Args:
      bin_path: Absolute path to the system_acceptance binary on the node.
      report_path: Where the harness writes report.json on the node.
      filter_category: Category glob, e.g. ``hip/*`` or ``verbs/*``.
      kind: ``func`` | ``perf`` | ``all``.
      policy: Absolute path to a policy CSV, or None to let the harness auto-select.
      vendor: Optional NIC vendor hint for policy auto-selection.
      gid: Optional RoCE GID index override.
      device: Optional preferred RDMA device name.
      quiet: Suppress per-test console lines (summary only).
      extra_args: Optional list of extra raw CLI args.

    Returns:
      str: Shell command suitable for ``orch.exec(...)``.
    """
    parts = [_q(bin_path)]
    if filter_category:
        parts += ["--filter-category", _q(filter_category)]
    if kind:
        parts += ["--kind", _q(kind)]
    if policy:
        parts += ["--policy", _q(policy)]
    if vendor:
        parts += ["--vendor", _q(vendor)]
    if gid not in (None, ""):
        parts += ["--gid", _q(gid)]
    if device:
        parts += ["--device", _q(device)]
    if quiet:
        parts.append("--quiet")
    parts += ["--report", _q(report_path)]
    if extra_args:
        parts += [_q(a) for a in extra_args]
    run = " ".join(parts)
    # '>&' (not '> ... 2>&1') so the redirect is valid under both POSIX shells
    # and csh/tcsh login shells; ';' (not '&&') so report.json is still cat'd
    # back when the gate verdict exits non-zero.
    return f"{run} >& {_q(report_path)}.log ; cat {_q(report_path)}"


def build_multinode_cmd(
    bin_path,
    nodes,
    *,
    report_path,
    ssh_user=None,
    remote_binary=None,
    port=None,
    filter_category=None,
    kind="func",
    policy=None,
    vendor=None,
    launch_timeout=None,
    extra_args=None,
):
    """Build a head-node ``system_acceptance --nodes`` launcher invocation.

    The harness owns the ssh fan-out to the (1 or 2) worker nodes and produces a
    single reconciled report.json on the head node. This must run on the head
    node only, not broadcast to every node.

    Args:
      bin_path: Path to the harness binary on the head node.
      nodes: Sequence of 1 or 2 worker hostnames/IPs.
      report_path: Where the head node writes the reconciled report.json.
      ssh_user: SSH user the launcher uses to reach workers.
      remote_binary: Path to the harness binary on the worker nodes.
      port: Rendezvous port.
      launch_timeout: Overall ssh launch deadline in seconds.
      (remaining args mirror build_accept_cmd.)

    Returns:
      str: Shell command suitable for a head-node exec.
    """
    parts = [_q(bin_path), "--nodes", _q(",".join(nodes))]
    if ssh_user:
        parts += ["--ssh-user", _q(ssh_user)]
    if remote_binary:
        parts += ["--remote-binary", _q(remote_binary)]
    if port not in (None, ""):
        parts += ["--port", _q(port)]
    if filter_category:
        parts += ["--filter-category", _q(filter_category)]
    if kind:
        parts += ["--kind", _q(kind)]
    if policy:
        parts += ["--policy", _q(policy)]
    if vendor:
        parts += ["--vendor", _q(vendor)]
    if launch_timeout not in (None, ""):
        parts += ["--launch-timeout", _q(launch_timeout)]
    parts += ["--report", _q(report_path)]
    if extra_args:
        parts += [_q(a) for a in extra_args]
    run = " ".join(parts)
    return f"{run} >& {_q(report_path)}.log ; cat {_q(report_path)}"


def parse_report(report_text):
    """Parse ``system_acceptance`` report.json text into a normalized summary.

    Args:
      report_text: Raw report.json contents (stdout captured from the node).

    Returns:
      dict with keys:
        schema_version, mode, filtered, summary, tests, gate_failures, ok.
      ``tests`` is a list of per-test dicts; ``gate_failures`` is the subset with
      ``gate_fail == True``; ``ok`` is True iff there are no gate failures.

    Raises:
      ValueError: if ``report_text`` is not valid JSON or is missing the
        top-level ``summary``/``tests`` contract.
    """
    try:
        doc = json.loads(report_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"report.json is not valid JSON: {exc}") from exc

    if not isinstance(doc, dict) or "tests" not in doc or "summary" not in doc:
        raise ValueError("report.json missing required top-level keys 'summary'/'tests'")

    schema_version = doc.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        log.warning(
            "system_acceptance: unrecognized report schema_version %r (supported: %s)",
            schema_version,
            SUPPORTED_SCHEMA_VERSIONS,
        )

    tests = []
    gate_failures = []
    for t in doc.get("tests", []):
        entry = {
            "id": t.get("id"),
            "category": t.get("category"),
            "status": t.get("status"),
            "gate_fail": bool(t.get("gate_fail")),
            "expected_status": t.get("expected_status"),
            "reason": t.get("reason"),
            "ticket": t.get("ticket"),
            "note": t.get("note"),
            "duration_ms": t.get("duration_ms"),
        }
        tests.append(entry)
        if entry["gate_fail"]:
            gate_failures.append({k: entry[k] for k in ("id", "status", "reason", "ticket", "note")})

    return {
        "schema_version": schema_version,
        "mode": doc.get("mode"),
        "filtered": bool(doc.get("filtered")),
        "summary": doc.get("summary", {}),
        "tests": tests,
        "gate_failures": gate_failures,
        "ok": len(gate_failures) == 0,
    }


def summary_line(parsed):
    """One-line human summary of a parsed report (for logs)."""
    s = parsed.get("summary", {})
    return (
        f"mode={parsed.get('mode')} "
        f"pass={s.get('passed', 0)} fail={s.get('failed', 0)} "
        f"skip={s.get('skipped', 0)} na={s.get('not_applicable', 0)} "
        f"gate_failures={len(parsed.get('gate_failures', []))}"
    )


def format_gate_failures(gate_failures):
    """Render gate failures as an indented multiline block for ``fail_test``."""
    lines = []
    for f in gate_failures:
        bits = [str(f.get("id") or "<unknown>"), f"status={f.get('status')}"]
        if f.get("ticket"):
            bits.append(f"ticket={f['ticket']}")
        if f.get("reason"):
            bits.append(f"reason={f['reason']}")
        if f.get("note"):
            bits.append(f"note={f['note']}")
        lines.append("  - " + " ".join(bits))
    return "\n".join(lines)
