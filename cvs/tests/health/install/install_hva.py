'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Install HIP/Verbs Acceptance (hva) on every node from a pinned release tag.
'''

import json
import uuid

import pytest

from cvs.lib import globals, hva_lib
from cvs.lib.utils_lib import *

log = globals.log


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


def _fail_nodes(out_dict, step):
    failed = False
    for node, res in out_dict.items():
        if res.get('exit_code') != 0:
            fail_test(f'hva {step} failed on node {node}: {(res.get("output") or "").strip()[-2000:]}')
            failed = True
    return failed


def _install_hosts(orch, path):
    """Head node plus every node that does not share ``path`` with it."""
    token = uuid.uuid4().hex
    orch.exec(hva_lib.share_mark_cmd(path, token), hosts=[orch.head_node], timeout=60)
    try:
        check_out = orch.exec(hva_lib.share_check_cmd(path, token), timeout=60)
    finally:
        orch.exec(hva_lib.share_clear_cmd(path, token), hosts=[orch.head_node], timeout=60)
    return hva_lib.install_hosts(orch.hosts, orch.head_node, check_out)


def test_install_hva(orch, config_dict):
    """
    Clone hva at ``git_tag``, build and install it under ``path`` on every node.

    Nodes that share ``path`` with the head node (NFS) are installed once, by
    the head node.

    The tag is required: CVS pins hva to a release so a CVS release always runs
    the same tests. After the build, ``hva --version`` must report that release.
    """
    globals.error_list = []
    log.info('Testcase install hva')
    path = config_dict['path']
    git_tag = config_dict.get('git_tag')
    if not git_tag:
        fail_test('hip_verbs_acceptance config is missing the required "git_tag" field')
        update_test_result()
        return

    hosts = _install_hosts(orch, path)
    log.info('Installing hva at %s on %s', path, ', '.join(hosts))

    out_dict = orch.exec(hva_lib.clone_cmd(config_dict['git_url'], path), hosts=hosts, timeout=300, detailed=True)
    if _fail_nodes(out_dict, 'clone'):
        update_test_result()
        return

    out_dict = orch.exec(hva_lib.checkout_cmd(path, git_tag), hosts=hosts, timeout=120, detailed=True)
    if _fail_nodes(out_dict, f'checkout of {git_tag}'):
        update_test_result()
        return
    for node, res in out_dict.items():
        lines = (res.get('output') or '').strip().splitlines()
        actual = lines[-1].strip() if lines else ''
        if actual != git_tag:
            fail_test(f'hva not at git tag {git_tag} on node {node} after checkout (git describe: {actual!r})')

    out_dict = orch.exec(hva_lib.build_cmd(path), hosts=hosts, timeout=900, detailed=True, print_console=False)
    if _fail_nodes(out_dict, 'build'):
        update_test_result()
        return

    out_dict = orch.exec(hva_lib.version_cmd(path), timeout=60)
    for node, output in out_dict.items():
        version, sha = hva_lib.parse_version(output)
        if not hva_lib.tag_matches_version(git_tag, version):
            fail_test(f'hva on node {node} reports version {version!r}, expected {git_tag}: {output.strip()}')
        else:
            log.info('hva %s (git %s) installed on node %s', version, sha, node)
    update_test_result()
