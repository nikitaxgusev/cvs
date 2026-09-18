'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import json

import pytest

from cvs.lib import system_acceptance_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *

from cvs.lib import globals

log = globals.log

# The harness C++ tree is vendored next to this suite
# (cvs/tests/system_acceptance/src/). It is uploaded to the nodes and built
# there; there is no git clone -- the source of truth lives inside the CVS
# package.
VENDORED_SRC = os.path.join(os.path.dirname(__file__), "src")


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
    config_dict = config_dict_t['system_acceptance']
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """Parallel SSH handle across all cluster nodes."""
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    return Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """Parallel SSH handle for the head node only (shared-filesystem builds)."""
    env_vars = cluster_dict.get("env_vars")
    head_node = list(cluster_dict['node_dict'].keys())[0]
    return Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)


def _detect_arches(hdl):
    """Union of GPU arches reported across the handle's nodes (e.g. {'gfx942'})."""
    arches = set()
    out_dict = hdl.exec(system_acceptance_lib.detect_arch_cmd(), timeout=60)
    for node, output in out_dict.items():
        arch = (output or "").strip().splitlines()[-1].strip() if output.strip() else ""
        if arch.startswith("gfx"):
            arches.add(arch)
            log.info("[%s] detected GPU arch: %s", node, arch)
        else:
            fail_test(f"[{node}] could not detect a gfx arch (rocminfo output empty)")
    return sorted(arches)


def test_install_system_acceptance(phdl, shdl, config_dict):
    """Upload the vendored harness tree to the nodes and build it (build-on-node).

    Mirrors the CVS install pattern (install_ibperf_tools / install_ompi) but the
    source comes from the in-repo vendored tree via SFTP upload instead of a git
    clone. Reads the ``build`` sub-section of the ``system_acceptance`` config.
    """
    globals.error_list = []
    build = config_dict.get('build', {})

    install_dir = build['install_dir'].rstrip('/')
    # The vendored tree lands directly in install_dir and is built in place.
    remote_src = install_dir
    rocm_path = build.get('rocm_dir') or '/opt/rocm'
    build_type = build.get('build_type', 'Release')
    build_timeout = int(build.get('build_timeout', 3600))
    nfs_install = str(build.get('nfs_install', False)).lower() in ('true', '1', 'yes')

    # Build once on the head node for shared-filesystem clusters, else on every node.
    hdl = shdl if nfs_install else phdl

    # 1) Arch set for the multi-arch static kernels: configured, else auto-detect.
    arches = build.get('arches') or _detect_arches(hdl)
    if not arches:
        fail_test("no GPU arches to build for (set build.arches or fix rocminfo)")
        update_test_result()
        return
    log.info("building static kernels for arches: %s", arches)

    # 2) Ship the vendored source tree to the node(s) (SFTP, key-based, recursive).
    log.info("uploading %s -> %s", VENDORED_SRC, remote_src)
    hdl.exec(f"rm -rf {remote_src} && mkdir -p {install_dir}", timeout=60)
    hdl.upload_file(VENDORED_SRC, remote_src, recurse=True)

    # 3) Regenerate the arch-specific static kernels, 4) configure + build.
    log.info("build step [static-kernels]")
    hdl.exec(system_acceptance_lib.build_kernels_cmd(remote_src, arches), timeout=build_timeout)
    log.info("build step [cmake-build]")
    hdl.exec(
        system_acceptance_lib.cmake_build_cmd(remote_src, rocm_path=rocm_path, build_type=build_type), timeout=build_timeout
    )

    # 5) Verify the binary exists and registers tests, on every node.
    binary = system_acceptance_lib.binary_path(remote_src)
    out_dict = phdl.exec(f"test -x {binary} && {binary} --list", timeout=120)
    for node in out_dict.keys():
        if 'hip/' not in out_dict[node]:
            fail_test(f"[{node}] build failed or binary missing: {binary} --list produced no tests")
        else:
            log.info("[%s] system_acceptance built OK at %s", node, binary)

    update_test_result()
