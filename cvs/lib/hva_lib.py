'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Command builders and report parsing for HIP/Verbs Acceptance (hva).

hva builds with only cmake and a C++ compiler: it ships the HIP, HSA and
ibverbs headers it needs and loads the runtimes with dlopen when a test runs.
Every run writes one report.json, and each test in it carries a ``gate_fail``
flag; that flag, not the process exit code, is the verdict CVS reports.
'''

import json
import re
import shlex

REPORT_BEGIN = '===HVA_REPORT_BEGIN==='
REPORT_END = '===HVA_REPORT_END==='
LOG_BEGIN = '===HVA_LOG_TAIL==='

_EXIT_RE = re.compile(r'^HVA_EXIT=(\d+)\s*$', re.M)
_VERSION_RE = re.compile(r'^hva (\S+) \(git ([^)]*)\)', re.M)
_ROCM_RE = re.compile(r'^HVA_ROCM=(\S+)\s*$', re.M)
_LOG_TAIL_LINES = 40


def _q(value):
    return shlex.quote(str(value))


def binary_path(path):
    '''Installed hva binary. Its policy/ directory sits next to it.'''
    return f'{path}/install/bin/hva'


def clone_cmd(git_url, path):
    return f"bash -c {_q(f'rm -rf {_q(path)} && git clone -q {_q(git_url)} {_q(path)}')}"


def checkout_cmd(path, git_tag):
    '''Check out ``git_tag`` and print the exact tag now at HEAD.'''
    inner = f'cd {_q(path)} && git checkout -q {_q(git_tag)} && git describe --tags --exact-match'
    return f'bash -c {_q(inner)}'


def build_cmd(path):
    '''Configure, build and install hva under ``<path>/install``.'''
    inner = (
        f'cd {_q(path)}'
        ' && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DACCEPT_WITH_HIP=ON -DACCEPT_WITH_IBVERBS=ON'
        ' && cmake --build build -j --target hva'
        ' && cmake --install build --prefix install'
    )
    return f'bash -c {_q(inner)}'


def version_cmd(path):
    return f'{_q(binary_path(path))} --version'


def parse_version(output):
    '''Return (version, git_sha) from ``hva --version`` output, or (None, None).'''
    match = _VERSION_RE.search(output or '')
    if not match:
        return None, None
    return match.group(1), match.group(2)


def tag_matches_version(git_tag, version):
    '''A release tag is ``v`` followed by the harness version.'''
    return bool(version) and git_tag in (version, f'v{version}')


def rocm_probe_cmd(config_rocm_path=''):
    '''Print ``HVA_ROCM=<dir>`` for the first ROCm tree holding libamdhip64.

    The configured path is tried first, then the newest /opt/rocm/core-X.Y,
    then /opt/rocm. One probe per node keeps the lookup to a single exec.
    '''
    configured = _q(config_rocm_path) if config_rocm_path else ''
    inner = (
        f'for d in {configured} $(ls -d /opt/rocm/core-* 2>/dev/null | sort -V -r) /opt/rocm; do '
        'if ls "$d"/lib/libamdhip64.so* >/dev/null 2>&1; then echo "HVA_ROCM=$d"; exit 0; fi; '
        'done; exit 1'
    )
    return f'bash -c {_q(inner)}'


def parse_rocm_probe(out_dict):
    '''Return {node: rocm_dir or None} from rocm_probe_cmd output.'''
    found = {}
    for node, output in (out_dict or {}).items():
        text = output if isinstance(output, str) else (output or {}).get('output', '')
        match = _ROCM_RE.search(text or '')
        found[node] = match.group(1) if match else None
    return found


def run_cmd(
    path,
    rocm_path,
    category,
    report_name,
    policy='',
    allow_known_failures=False,
    ranks=1,
    hosts=None,
    launch_timeout=None,
    extra_args=None,
):
    '''Build one hva run that prints its exit code, report.json and log tail.

    Reports go under ``<path>/reports`` with the host name in the file name, so
    nodes sharing an NFS install do not overwrite each other. hva exits 1 when
    the gate fails; the command itself always succeeds so the report is read.

    Args:
      path: hva checkout; the binary is ``binary_path(path)``.
      rocm_path: ROCm tree whose lib/ holds libamdhip64.
      category: category glob, e.g. ``hip/*`` or ``verbs/*``.
      report_name: report file stem, e.g. ``hip``.
      policy: empty for hva's own per-node choice, a file name under the
        installed policy/ directory, or an absolute path.
      ranks: 1, or 2 for the two-rank Verbs tests.
      hosts: for two-node Verbs, the two addresses ranks are placed on.
      launch_timeout: overall deadline (s) hva gives a multi-rank run.
    '''
    binary = binary_path(path)
    args = [binary, '--filter-category', category, '--kind', 'func', '--quiet']
    if ranks > 1:
        args += ['-n', str(ranks)]
    if hosts:
        args += ['-ppn', '1', '-hosts', ','.join(hosts)]
    if launch_timeout:
        args += ['--launch-timeout', str(int(launch_timeout))]
    if policy:
        policy_path = policy if policy.startswith('/') else f'{path}/install/bin/policy/{policy}'
        args += ['--policy', policy_path]
    if allow_known_failures:
        args.append('--allow-known-failures')
    args += list(extra_args or [])

    report = f'{path}/reports/$(hostname -s)-{report_name}.json'
    inner = (
        f'export ROCM_PATH={_q(rocm_path)} LD_LIBRARY_PATH={_q(rocm_path)}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; '
        f'mkdir -p {_q(path)}/reports; r="{report}"; rm -f "$r"; '
        f'{" ".join(_q(a) for a in args)} --report "$r" > "$r.log" 2>&1; '
        f'echo "HVA_EXIT=$?"; echo {REPORT_BEGIN}; cat "$r" 2>/dev/null; echo; echo {REPORT_END}; '
        f'echo {LOG_BEGIN}; tail -n {_LOG_TAIL_LINES} "$r.log" 2>/dev/null; true'
    )
    return f'bash -c {_q(inner)}'


def parse_run_output(output):
    '''Split run_cmd output into exit code, parsed report and log tail.

    Returns:
      dict: ``exit_code`` (int or None), ``report`` (dict or None),
        ``log_tail`` (str) and ``error`` (str, empty when the report parsed).
    '''
    text = output if isinstance(output, str) else (output or {}).get('output', '')
    text = text or ''
    result = {'exit_code': None, 'report': None, 'log_tail': '', 'error': ''}
    exit_match = _EXIT_RE.search(text)
    if exit_match:
        result['exit_code'] = int(exit_match.group(1))
    if LOG_BEGIN in text:
        result['log_tail'] = text.split(LOG_BEGIN, 1)[1].strip()
    begin = text.find(REPORT_BEGIN)
    end = text.find(REPORT_END, begin + 1) if begin >= 0 else -1
    body = text[begin + len(REPORT_BEGIN) : end].strip() if begin >= 0 and end > begin else ''
    if not exit_match:
        result['error'] = 'hva did not run (no exit code in output)'
    elif not body:
        result['error'] = f'hva exited {result["exit_code"]} without writing report.json'
    else:
        try:
            result['report'] = json.loads(body)
        except ValueError as exc:
            result['error'] = f'report.json is not valid JSON: {exc}'
    return result


def describe_test(test):
    '''One-line description of a report.json test entry for failure messages.'''
    parts = [f"{test.get('id', '?')}: {test.get('status', '?')}"]
    detail = [test.get('class') or '']
    if test.get('ticket'):
        detail.append(test['ticket'])
    parts.append(f"({', '.join(d for d in detail if d)})")
    reason = test.get('note') or test.get('reason')
    if reason:
        parts.append(f'- {reason}')
    return ' '.join(parts)


def verdict(parsed):
    '''Summarize one parsed run.

    Returns:
      dict: ``passed`` (bool), ``counts`` (summary dict), ``gate_failures``
        and ``known_failures`` (lists of report test entries), ``version``
        and ``error``. A known failure is a KNOWN_FAILURE test that failed
        as expected and did not fail the gate.
    '''
    report = parsed.get('report')
    if report is None:
        return {
            'passed': False,
            'counts': {},
            'gate_failures': [],
            'known_failures': [],
            'version': '',
            'error': parsed.get('error') or 'no report',
        }
    tests = report.get('tests') or []
    gate_failures = [t for t in tests if t.get('gate_fail')]
    known = [
        t for t in tests if not t.get('gate_fail') and t.get('status') == 'fail' and t.get('class') == 'known_failure'
    ]
    harness = report.get('harness') or {}
    version = harness.get('version') or ''
    if harness.get('git_sha'):
        version = f"{version} ({harness['git_sha']})" if version else harness['git_sha']
    error = ''
    if not gate_failures and parsed.get('exit_code') not in (0, None):
        error = f"hva exited {parsed['exit_code']} with no failing test in report.json"
    return {
        'passed': not gate_failures and not error,
        'counts': dict(report.get('summary') or {}),
        'gate_failures': gate_failures,
        'known_failures': known,
        'version': version,
        'error': error,
    }


def verbs_pairs(hosts):
    '''Pair nodes for two-node Verbs runs.

    Consecutive nodes form a pair; an odd last node is paired with the first so
    every node is tested. A single node runs both ranks locally.
    '''
    hosts = list(hosts or [])
    if len(hosts) < 2:
        return [tuple(hosts)] if hosts else []
    pairs = [(hosts[i], hosts[i + 1]) for i in range(0, len(hosts) - 1, 2)]
    if len(hosts) % 2:
        pairs.append((hosts[-1], hosts[0]))
    return pairs


def node_address(cluster_dict, node):
    '''Address other nodes use to reach ``node``: its vpc_ip, else the node key.'''
    info = ((cluster_dict or {}).get('node_dict') or {}).get(node) or {}
    return info.get('vpc_ip') or node
