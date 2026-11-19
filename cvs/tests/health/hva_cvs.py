'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

HIP/Verbs Acceptance (hva): HIP runtime and RDMA verbs functional tests.

Run install_hva first. HIP tests run on every node at once. Verbs tests need
two ranks: on a multi-node cluster each pair of nodes runs one two-node run,
launched from the first node of the pair, which reaches its peer over ssh at
the peer's vpc_ip; a single node runs both ranks locally.
'''

import json

import pytest

from cvs.lib import globals, hva_lib, hva_rundeck
from cvs.lib.report.health_lifecycle import HealthLifecycle, timed_stage
from cvs.lib.utils_lib import *

log = globals.log

_DEFAULT_HIP_TIMEOUT = 1800
_DEFAULT_VERBS_TIMEOUT = 1800


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = resolve_test_config_placeholders(config_dict_t['hip_verbs_acceptance'], cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def lifecycle():
    """Wall-clock of each hva run, bound as the deck lifecycle source."""
    return HealthLifecycle()


@pytest.fixture(scope="module")
def hva_res_dict():
    """
    Module-scoped structured hva results for the Run Deck status matrix.

    The hva_cvs profile names this fixture in sources.results, so session
    binding captures it at module teardown.
    """
    return {}


@pytest.fixture(scope="module")
def rocm_path(orch, config_dict):
    """ROCm tree hva loads libamdhip64 from, taken from the head node."""
    found = hva_lib.parse_rocm_probe(orch.exec(hva_lib.rocm_probe_cmd(config_dict.get('rocm_path') or ''), timeout=60))
    paths = {path for path in found.values() if path}
    if not paths:
        pytest.fail(f'No ROCm installation with libamdhip64 found on any node: {found}')
    if len(paths) > 1:
        log.warning('Nodes resolve different ROCm paths: %s', found)
    return found.get(orch.head_node) or sorted(paths)[0]


def _run_kwargs(config_dict):
    return {
        'policy': config_dict.get('policy') or '',
        'allow_known_failures': bool(config_dict.get('allow_known_failures', False)),
    }


def _check_run(node, parsed, label):
    result = hva_lib.verdict(parsed)
    if parsed.get('report') is None:
        fail_test(f'hva {label} on node {node}: {result["error"]}. Log tail:\n{parsed.get("log_tail", "")}')
        return
    for test in result['gate_failures']:
        fail_test(f'hva {label} on node {node}: {hva_lib.describe_test(test)}')
    if result['error']:
        fail_test(f'hva {label} on node {node}: {result["error"]}')
    for test in result['known_failures']:
        log.info('hva %s on node %s: known failure %s', label, node, hva_lib.describe_test(test))
    log.info(
        'hva %s on node %s: %s',
        label,
        node,
        hva_rundeck.summarize_counts(result['counts'], len(result['known_failures'])),
    )


def _capture_rundeck(res_dict, cluster_dict, group, node_runs):
    """Best-effort: a reporting problem must not change the hva pass/fail."""
    try:
        meta = hva_rundeck.make_meta(cluster_dict, 'hva_cvs')
        hva_rundeck.record_run(res_dict, group, node_runs, meta=meta)
    except Exception as exc:
        log.warning('hva: could not capture Run Deck results: %s', exc)


def test_hva_hip(orch, config_dict, cluster_dict, rocm_path, hva_res_dict, lifecycle):
    globals.error_list = []
    log.info('Testcase Run hva HIP tests')
    cmd = hva_lib.run_cmd(config_dict['path'], rocm_path, 'hip/*', 'hip', **_run_kwargs(config_dict))
    timeout = int(config_dict.get('hip_timeout') or _DEFAULT_HIP_TIMEOUT)
    with timed_stage(lifecycle, 'hip'):
        out_dict = orch.exec(cmd, timeout=timeout, print_console=False)
    node_runs = {}
    for node, output in out_dict.items():
        parsed = hva_lib.parse_run_output(output)
        _check_run(node, parsed, 'HIP')
        node_runs[node] = (parsed, None)
    _capture_rundeck(hva_res_dict, cluster_dict, 'hip', node_runs)
    update_test_result()


def test_hva_verbs(orch, config_dict, cluster_dict, rocm_path, hva_res_dict, lifecycle):
    globals.error_list = []
    log.info('Testcase Run hva Verbs tests')
    timeout = int(config_dict.get('verbs_timeout') or _DEFAULT_VERBS_TIMEOUT)
    node_runs = {}
    with timed_stage(lifecycle, 'verbs'):
        for pair in hva_lib.verbs_pairs(orch.hosts):
            launcher = pair[0]
            peer = pair[1] if len(pair) > 1 else None
            hosts = [hva_lib.node_address(cluster_dict, node) for node in pair] if peer else None
            cmd = hva_lib.run_cmd(
                config_dict['path'],
                rocm_path,
                'verbs/*',
                'verbs',
                ranks=2,
                hosts=hosts,
                launch_timeout=timeout - 60 if timeout > 120 else timeout,
                **_run_kwargs(config_dict),
            )
            label = f'Verbs with {peer}' if peer else 'Verbs (2 local ranks)'
            out_dict = orch.exec(cmd, hosts=[launcher], timeout=timeout, print_console=False)
            parsed = hva_lib.parse_run_output(out_dict.get(launcher, ''))
            _check_run(launcher, parsed, label)
            node_runs[launcher] = (parsed, peer)
            if peer and peer not in node_runs:
                node_runs[peer] = (parsed, launcher)
    _capture_rundeck(hva_res_dict, cluster_dict, 'verbs', node_runs)
    update_test_result()
