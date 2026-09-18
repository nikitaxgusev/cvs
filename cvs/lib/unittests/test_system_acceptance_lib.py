'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import unittest

from cvs.lib import system_acceptance_lib


def _report(tests, *, summary=None, mode="gate", schema_version="1.0"):
    doc = {
        "schema_version": schema_version,
        "mode": mode,
        "filtered": False,
        "summary": summary or {"passed": 0, "failed": 0, "skipped": 0, "not_applicable": 0},
        "homogeneous": True,
        "nodes": [],
        "tests": tests,
    }
    return json.dumps(doc)


class TestBuildAcceptCmd(unittest.TestCase):
    def test_minimal_cmd_has_report_and_cat(self):
        cmd = system_acceptance_lib.build_accept_cmd("/opt/sa/system_acceptance", report_path="/tmp/r.json")
        self.assertIn("/opt/sa/system_acceptance", cmd)
        self.assertIn("--report /tmp/r.json", cmd)
        self.assertIn("--kind func", cmd)
        self.assertIn("--quiet", cmd)
        # report is preserved even when the gate exits non-zero (';' not '&&')
        self.assertIn(" ; cat /tmp/r.json", cmd)
        self.assertNotIn("&&", cmd)

    def test_all_options_rendered(self):
        cmd = system_acceptance_lib.build_accept_cmd(
            "/opt/sa/system_acceptance",
            report_path="/tmp/r.json",
            filter_category="hip/*",
            kind="all",
            policy="/opt/sa/policy/mi350_ionic.csv",
            vendor="ionic",
            gid=3,
            device="rdma0",
            quiet=False,
            extra_args=["--hide-na"],
        )
        self.assertIn("--filter-category 'hip/*'", cmd)
        self.assertIn("--kind all", cmd)
        self.assertIn("--policy /opt/sa/policy/mi350_ionic.csv", cmd)
        self.assertIn("--vendor ionic", cmd)
        self.assertIn("--gid 3", cmd)
        self.assertIn("--device rdma0", cmd)
        self.assertIn("--hide-na", cmd)
        self.assertNotIn("--quiet", cmd)

    def test_glob_is_shell_quoted(self):
        cmd = system_acceptance_lib.build_accept_cmd(
            "/opt/sa/system_acceptance", report_path="/tmp/r.json", filter_category="verbs/*"
        )
        self.assertIn("'verbs/*'", cmd)


class TestBuildMultinodeCmd(unittest.TestCase):
    def test_nodes_and_launcher_flags(self):
        cmd = system_acceptance_lib.build_multinode_cmd(
            "/opt/sa/system_acceptance",
            ["h1", "h2"],
            report_path="/tmp/r.json",
            ssh_user="testuser",
            remote_binary="/opt/sa/system_acceptance",
            port=18515,
            filter_category="verbs/*",
            policy="/opt/sa/policy/mi350_ionic.csv",
            launch_timeout=180,
        )
        self.assertIn("--nodes h1,h2", cmd)
        self.assertIn("--ssh-user testuser", cmd)
        self.assertIn("--remote-binary /opt/sa/system_acceptance", cmd)
        self.assertIn("--port 18515", cmd)
        self.assertIn("--launch-timeout 180", cmd)
        self.assertIn(" ; cat /tmp/r.json", cmd)


class TestBuildPhaseCmds(unittest.TestCase):
    def test_detect_arch_is_shell_agnostic(self):
        # Wrapped in 'sh -c' so the 2>/dev/null redirect is valid under csh/tcsh
        # login shells too (bare '2>' raises "Ambiguous output redirect" there).
        cmd = system_acceptance_lib.detect_arch_cmd()
        self.assertTrue(cmd.startswith("sh -c"))
        self.assertIn("rocminfo", cmd)
        self.assertIn("gfx", cmd)

    def test_build_kernels_uses_env_prefix(self):
        # 'env ARCHES=...' (not 'ARCHES=... cmd') so it is valid under csh/tcsh too.
        cmd = system_acceptance_lib.build_kernels_cmd("/home/u/src/system_acceptance", ["gfx942", "gfx950"])
        self.assertIn("hip/kernel_launch", cmd)
        self.assertIn("env ARCHES='gfx942 gfx950' bash build_static_kernels.sh", cmd)

    def test_cmake_build_flags(self):
        cmd = system_acceptance_lib.cmake_build_cmd("/home/u/src", rocm_path="/opt/rocm-7", build_type="Release")
        self.assertIn("cmake -S . -B build", cmd)
        self.assertIn("-DACCEPT_WITH_HIP=ON", cmd)
        self.assertIn("-DACCEPT_WITH_IBVERBS=ON", cmd)
        self.assertIn("-DCMAKE_PREFIX_PATH=/opt/rocm-7", cmd)
        self.assertIn("cmake --build build -j", cmd)

    def test_binary_path(self):
        self.assertEqual(
            system_acceptance_lib.binary_path("/home/u/src/system_acceptance"),
            "/home/u/src/system_acceptance/build/system_acceptance",
        )


class TestParseReport(unittest.TestCase):
    def test_all_pass_is_ok(self):
        text = _report(
            [
                {"id": "hip/memcpy", "category": "hip", "status": "PASS", "gate_fail": False},
                {"id": "verbs/rc_pingpong", "category": "verbs", "status": "PASS", "gate_fail": False},
            ],
            summary={"passed": 2, "failed": 0, "skipped": 0, "not_applicable": 0},
        )
        parsed = system_acceptance_lib.parse_report(text)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["gate_failures"], [])
        self.assertEqual(len(parsed["tests"]), 2)
        self.assertEqual(parsed["summary"]["passed"], 2)

    def test_gate_fail_flags_not_ok(self):
        text = _report(
            [
                {
                    "id": "hip/launch",
                    "category": "hip",
                    "status": "FAIL",
                    "gate_fail": True,
                    "reason": "device kernel image is invalid",
                    "ticket": "TICKET-1",
                    "note": None,
                },
                {"id": "hip/memcpy", "category": "hip", "status": "PASS", "gate_fail": False},
            ],
            summary={"passed": 1, "failed": 1, "skipped": 0, "not_applicable": 0},
        )
        parsed = system_acceptance_lib.parse_report(text)
        self.assertFalse(parsed["ok"])
        self.assertEqual(len(parsed["gate_failures"]), 1)
        self.assertEqual(parsed["gate_failures"][0]["id"], "hip/launch")

    def test_known_failure_does_not_gate(self):
        # A KNOWN_FAILURE status that the policy expects: status may be FAIL but
        # gate_fail is False, so the overall gate stays green.
        text = _report(
            [
                {
                    "id": "verbs/reset_recovery",
                    "category": "verbs",
                    "status": "FAIL",
                    "gate_fail": False,
                    "expected_status": "FAIL",
                    "ticket": "TICKET-2",
                },
            ],
            summary={"passed": 0, "failed": 0, "skipped": 0, "not_applicable": 1},
        )
        parsed = system_acceptance_lib.parse_report(text)
        self.assertTrue(parsed["ok"])

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            system_acceptance_lib.parse_report("not json at all")

    def test_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            system_acceptance_lib.parse_report(json.dumps({"schema_version": "1.0"}))

    def test_format_gate_failures_renders_fields(self):
        text = _report(
            [
                {
                    "id": "hip/x",
                    "category": "hip",
                    "status": "FAIL",
                    "gate_fail": True,
                    "reason": "boom",
                    "ticket": "TICKET-9",
                }
            ],
            summary={"passed": 0, "failed": 1, "skipped": 0, "not_applicable": 0},
        )
        parsed = system_acceptance_lib.parse_report(text)
        rendered = system_acceptance_lib.format_gate_failures(parsed["gate_failures"])
        self.assertIn("hip/x", rendered)
        self.assertIn("ticket=TICKET-9", rendered)
        self.assertIn("reason=boom", rendered)


if __name__ == "__main__":
    unittest.main()
