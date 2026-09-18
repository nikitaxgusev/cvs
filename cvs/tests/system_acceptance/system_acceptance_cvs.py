'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

import pytest

from cvs.lib import system_acceptance_lib
from cvs.lib.utils_lib import *

from cvs.lib import globals

log = globals.log


# Importing additional cmd line args to script ..
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
    """Load the ``system_acceptance`` section of the test config file.

    Expects the JSON at ``--config_file`` to contain a top-level
    ``system_acceptance`` key. ``<changeme>`` placeholders are hard-rejected by
    ``resolve_test_config_placeholders`` at run time, so a user must fill in
    ``bin_path`` (and typically ``policy``) before running.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['system_acceptance']
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


def _run_and_gate(orch, config_dict, filter_category, timeout_key, default_timeout):
    """Run the harness for a category glob on every node and gate on report.json.

    Executes ``system_acceptance --filter-category <glob> ...`` across the
    cluster via ``orch.exec``, then parses each node's report.json and fails the
    test if any test on any node carries ``gate_fail: true``.
    """
    globals.error_list = []

    bin_path = config_dict['bin_path']
    report_path = config_dict.get('report_path', '/tmp/system_acceptance_report.json')
    timeout = int(config_dict.get(timeout_key, default_timeout))

    cmd = system_acceptance_lib.build_accept_cmd(
        bin_path,
        report_path=report_path,
        filter_category=filter_category,
        kind=config_dict.get('kind', 'func'),
        policy=config_dict.get('policy'),
        vendor=config_dict.get('vendor'),
        gid=config_dict.get('gid_index'),
        device=config_dict.get('rdma_device'),
        quiet=True,
        extra_args=config_dict.get('extra_args'),
    )

    log.info("Executing on all nodes: %s", cmd)
    out_dict = orch.exec(cmd, timeout=timeout)

    for node in out_dict.keys():
        try:
            parsed = system_acceptance_lib.parse_report(out_dict[node])
        except ValueError as exc:
            fail_test(f"[{node}] could not parse report.json for '{filter_category}': {exc}")
            continue

        log.info("[%s] system_acceptance %s -> %s", node, filter_category, system_acceptance_lib.summary_line(parsed))
        if not parsed['ok']:
            fail_test(
                f"[{node}] acceptance gate FAILED for '{filter_category}':\n"
                + system_acceptance_lib.format_gate_failures(parsed['gate_failures'])
            )

    update_test_result()


################################################################################
# Testcases
################################################################################


def test_hip_single_node(orch, config_dict):
    """HIP-runtime acceptance gate (category ``hip/*``) on every node."""
    _run_and_gate(orch, config_dict, 'hip/*', 'hip_timeout', 1800)


def test_verbs_single_node(orch, config_dict):
    """Single-node Verbs / RDMA acceptance gate (category ``verbs/*``) on every node."""
    _run_and_gate(orch, config_dict, 'verbs/*', 'verbs_timeout', 1800)


def test_verbs_two_node(orch, config_dict, cluster_dict):
    """Two-node Verbs gate via the harness's own ssh launcher (``--nodes h1,h2``).

    Skeleton / extension point. Skipped unless the config sets ``two_node: true``
    and the cluster file has >= 2 nodes. Unlike the single-node cases, the
    harness owns the ssh fan-out here and must run on the head node only, so it
    cannot go through ``orch.exec``'s broadcast. Wire this to a head-node handle
    (SSH creds live in ``cluster_dict``: ``username``, ``priv_key_file``,
    per-node ``vpc_ip``) and parse the reconciled report.json the same way.
    """
    if str(config_dict.get('two_node', '')).lower() not in ('true', '1', 'yes'):
        pytest.skip('two_node disabled in config (set "two_node": true to enable)')

    node_list = list(cluster_dict['node_dict'].keys())
    if len(node_list) < 2:
        pytest.skip('two_node requires at least 2 nodes in the cluster file')

    pytest.skip(
        'two_node launcher not wired yet; build the command with '
        'system_acceptance_lib.build_multinode_cmd(...) and run it on the head node only'
    )
