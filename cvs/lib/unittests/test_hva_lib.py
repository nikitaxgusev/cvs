'''Unit tests for cvs.lib.hva_lib.'''

import json
import shlex
import unittest

from cvs.lib import hva_lib


def _test(test_id, status, cls='must_pass', gate_fail=False, **extra):
    entry = {'id': test_id, 'status': status, 'class': cls, 'gate_fail': gate_fail}
    entry.update(extra)
    return entry


def _report(tests, summary=None):
    return {
        'schema_version': '1.0',
        'harness': {'version': '0.1.0', 'git_sha': 'abc1234'},
        'summary': summary or {'passed': 1, 'failed': 0, 'skipped': 0, 'not_applicable': 0},
        'tests': tests,
    }


def _output(exit_code, report=None, log='last log line'):
    body = json.dumps(report) if report is not None else ''
    return f'HVA_EXIT={exit_code}\n{hva_lib.REPORT_BEGIN}\n{body}\n{hva_lib.REPORT_END}\n{hva_lib.LOG_BEGIN}\n{log}\n'


def _inner(cmd):
    '''The script a ``bash -c <script>`` command runs.'''
    argv = shlex.split(cmd)
    assert argv[:2] == ['bash', '-c'], argv
    return argv[2]


class TestInstallCommands(unittest.TestCase):
    def test_clone_replaces_checkout(self):
        inner = _inner(hva_lib.clone_cmd('https://github.com/ROCm/hip-verbs-acceptance.git', '/opt/hva'))
        self.assertIn('rm -rf /opt/hva', inner)
        self.assertIn('git clone -q https://github.com/ROCm/hip-verbs-acceptance.git /opt/hva', inner)

    def test_checkout_prints_exact_tag(self):
        inner = _inner(hva_lib.checkout_cmd('/opt/hva', 'v0.1.0'))
        self.assertIn('git checkout -q v0.1.0', inner)
        self.assertTrue(inner.endswith('git describe --tags --exact-match'))

    def test_build_enables_hip_and_verbs_and_installs(self):
        inner = _inner(hva_lib.build_cmd('/opt/hva'))
        self.assertIn('-DACCEPT_WITH_HIP=ON', inner)
        self.assertIn('-DACCEPT_WITH_IBVERBS=ON', inner)
        self.assertIn('cmake --install build --prefix install', inner)

    def test_binary_is_the_installed_one(self):
        self.assertEqual(hva_lib.binary_path('/opt/hva'), '/opt/hva/install/bin/hva')
        self.assertEqual(hva_lib.version_cmd('/opt/hva'), '/opt/hva/install/bin/hva --version')


class TestSharedPathProbe(unittest.TestCase):
    def test_marker_sits_beside_path(self):
        mark = _inner(hva_lib.share_mark_cmd('/home/u/INSTALL/hva/', 'tok'))
        self.assertIn('mkdir -p /home/u/INSTALL', mark)
        self.assertIn('touch /home/u/INSTALL/.cvs-hva-probe-tok', mark)
        self.assertIn(
            'test -e /home/u/INSTALL/.cvs-hva-probe-tok', _inner(hva_lib.share_check_cmd('/home/u/INSTALL/hva', 'tok'))
        )
        self.assertEqual(
            hva_lib.share_clear_cmd('/home/u/INSTALL/hva', 'tok'), 'rm -f /home/u/INSTALL/.cvs-hva-probe-tok'
        )

    def test_marker_under_root_and_relative_paths(self):
        self.assertIn('touch /.cvs-hva-probe-tok', _inner(hva_lib.share_mark_cmd('/hva', 'tok')))
        self.assertIn('touch ./.cvs-hva-probe-tok', _inner(hva_lib.share_mark_cmd('hva', 'tok')))

    def test_shared_nodes_are_installed_by_head(self):
        check = {'n1': 'HVA_SHARED=1\n', 'n2': 'HVA_SHARED=1\n', 'n3': 'HVA_SHARED=0\n'}
        self.assertEqual(hva_lib.install_hosts(['n1', 'n2', 'n3'], 'n1', check), ['n1', 'n3'])

    def test_node_local_path_installs_everywhere(self):
        check = {'n1': 'HVA_SHARED=1\n', 'n2': 'HVA_SHARED=0\n'}
        self.assertEqual(hva_lib.install_hosts(['n1', 'n2'], 'n1', check), ['n1', 'n2'])

    def test_missing_probe_output_installs(self):
        self.assertEqual(hva_lib.install_hosts(['n1', 'n2'], 'n1', {}), ['n1', 'n2'])


class TestVersion(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(hva_lib.parse_version('hva 0.1.0 (git 8cfe768)\n'), ('0.1.0', '8cfe768'))

    def test_parse_version_missing(self):
        self.assertEqual(hva_lib.parse_version('bash: hva: command not found'), (None, None))

    def test_tag_matches_version(self):
        self.assertTrue(hva_lib.tag_matches_version('v0.1.0', '0.1.0'))
        self.assertTrue(hva_lib.tag_matches_version('0.1.0', '0.1.0'))
        self.assertFalse(hva_lib.tag_matches_version('v0.2.0', '0.1.0'))
        self.assertFalse(hva_lib.tag_matches_version('v0.1.0', None))


class TestRocmProbe(unittest.TestCase):
    def test_configured_path_is_tried_first(self):
        inner = _inner(hva_lib.rocm_probe_cmd('/opt/rocm-7.0.2'))
        self.assertLess(inner.index('/opt/rocm-7.0.2'), inner.index('/opt/rocm/core-'))

    def test_auto_detect_without_configured_path(self):
        inner = _inner(hva_lib.rocm_probe_cmd(''))
        self.assertTrue(inner.startswith('for d in  $(ls -d /opt/rocm/core-*'))

    def test_parse_probe(self):
        found = hva_lib.parse_rocm_probe(
            {'n1': 'HVA_ROCM=/opt/rocm/core-7.13\n', 'n2': {'output': 'HVA_ROCM=/opt/rocm'}, 'n3': ''}
        )
        self.assertEqual(found, {'n1': '/opt/rocm/core-7.13', 'n2': '/opt/rocm', 'n3': None})


class TestRunCmd(unittest.TestCase):
    def test_single_rank_hip(self):
        inner = _inner(hva_lib.run_cmd('/opt/hva', '/opt/rocm', 'hip/*', 'hip'))
        self.assertIn("/opt/hva/install/bin/hva --filter-category 'hip/*' --kind func --quiet", inner)
        self.assertIn('ROCM_PATH=/opt/rocm', inner)
        self.assertIn('/opt/hva/reports/$(hostname -s)-hip.json', inner)
        self.assertNotIn('-n 2', inner)
        self.assertNotIn('--allow-known-failures', inner)
        self.assertNotIn('--policy', inner)

    def test_two_node_verbs(self):
        inner = _inner(
            hva_lib.run_cmd(
                '/opt/hva',
                '/opt/rocm',
                'verbs/*',
                'verbs',
                ranks=2,
                hosts=['10.0.0.1', '10.0.0.2'],
                launch_timeout=1740,
            )
        )
        self.assertIn('-n 2 -ppn 1 -hosts 10.0.0.1,10.0.0.2 --launch-timeout 1740', inner)

    def test_policy_name_resolves_under_install(self):
        inner = _inner(hva_lib.run_cmd('/opt/hva', '/opt/rocm', 'hip/*', 'hip', policy='mi350_thor.csv'))
        self.assertIn('--policy /opt/hva/install/bin/policy/mi350_thor.csv', inner)

    def test_absolute_policy_is_kept(self):
        inner = _inner(hva_lib.run_cmd('/opt/hva', '/opt/rocm', 'hip/*', 'hip', policy='/site/mi300.csv'))
        self.assertIn('--policy /site/mi300.csv', inner)

    def test_allow_known_failures(self):
        inner = _inner(hva_lib.run_cmd('/opt/hva', '/opt/rocm', 'hip/*', 'hip', allow_known_failures=True))
        self.assertIn('--allow-known-failures', inner)

    def test_report_is_printed_even_when_gate_fails(self):
        inner = _inner(hva_lib.run_cmd('/opt/hva', '/opt/rocm', 'hip/*', 'hip'))
        run, rest = inner.split('echo "HVA_EXIT=$?"', 1)
        self.assertTrue(run.rstrip().endswith(';'))
        self.assertIn(hva_lib.REPORT_BEGIN, rest)
        self.assertTrue(inner.endswith('true'))


class TestParseRunOutput(unittest.TestCase):
    def test_report_parsed(self):
        parsed = hva_lib.parse_run_output(_output(0, _report([_test('A', 'pass')])))
        self.assertEqual(parsed['exit_code'], 0)
        self.assertEqual(parsed['report']['tests'][0]['id'], 'A')
        self.assertEqual(parsed['log_tail'], 'last log line')
        self.assertEqual(parsed['error'], '')

    def test_detailed_result_dict(self):
        parsed = hva_lib.parse_run_output({'output': _output(1, _report([])), 'exit_code': 0})
        self.assertEqual(parsed['exit_code'], 1)
        self.assertIsNotNone(parsed['report'])

    def test_missing_report(self):
        parsed = hva_lib.parse_run_output(_output(2))
        self.assertIsNone(parsed['report'])
        self.assertIn('exited 2 without writing report.json', parsed['error'])

    def test_not_run(self):
        parsed = hva_lib.parse_run_output('ssh: connect to host n1 port 22: Connection refused')
        self.assertIsNone(parsed['exit_code'])
        self.assertIn('did not run', parsed['error'])

    def test_invalid_json(self):
        text = f'HVA_EXIT=1\n{hva_lib.REPORT_BEGIN}\n{{"tests": [\n{hva_lib.REPORT_END}\n'
        parsed = hva_lib.parse_run_output(text)
        self.assertIn('not valid JSON', parsed['error'])


class TestVerdict(unittest.TestCase):
    def test_pass(self):
        result = hva_lib.verdict(hva_lib.parse_run_output(_output(0, _report([_test('A', 'pass')]))))
        self.assertTrue(result['passed'])
        self.assertEqual(result['version'], '0.1.0 (abc1234)')

    def test_gate_failure(self):
        tests = [_test('A', 'pass'), _test('B', 'fail', gate_fail=True, reason='hipMalloc returned 2')]
        result = hva_lib.verdict(hva_lib.parse_run_output(_output(1, _report(tests))))
        self.assertFalse(result['passed'])
        self.assertEqual([t['id'] for t in result['gate_failures']], ['B'])

    def test_tolerated_known_failure(self):
        tests = [_test('A', 'pass'), _test('K', 'fail', cls='known_failure', ticket='ROCM-1')]
        result = hva_lib.verdict(hva_lib.parse_run_output(_output(0, _report(tests))))
        self.assertTrue(result['passed'])
        self.assertEqual([t['id'] for t in result['known_failures']], ['K'])

    def test_known_failure_that_gates(self):
        tests = [_test('K', 'fail', cls='known_failure', gate_fail=True, ticket='ROCM-1')]
        result = hva_lib.verdict(hva_lib.parse_run_output(_output(1, _report(tests))))
        self.assertFalse(result['passed'])
        self.assertEqual(result['known_failures'], [])

    def test_nonzero_exit_without_failing_test(self):
        result = hva_lib.verdict(hva_lib.parse_run_output(_output(3, _report([_test('A', 'pass')]))))
        self.assertFalse(result['passed'])
        self.assertIn('exited 3', result['error'])

    def test_no_report(self):
        result = hva_lib.verdict(hva_lib.parse_run_output(_output(2)))
        self.assertFalse(result['passed'])
        self.assertIn('without writing report.json', result['error'])


class TestDescribeTest(unittest.TestCase):
    def test_with_ticket_and_note(self):
        text = hva_lib.describe_test(
            _test('Vmm_Map', 'fail', cls='known_failure', ticket='ROCM-25387', note='known failure', reason='timeout')
        )
        self.assertEqual(text, 'Vmm_Map: fail (known_failure, ROCM-25387) - known failure')

    def test_reason_only(self):
        self.assertEqual(hva_lib.describe_test(_test('A', 'fail', reason='boom')), 'A: fail (must_pass) - boom')


class TestVerbsPairs(unittest.TestCase):
    def test_single_node(self):
        self.assertEqual(hva_lib.verbs_pairs(['n1']), [('n1',)])

    def test_no_nodes(self):
        self.assertEqual(hva_lib.verbs_pairs([]), [])

    def test_even(self):
        self.assertEqual(hva_lib.verbs_pairs(['n1', 'n2', 'n3', 'n4']), [('n1', 'n2'), ('n3', 'n4')])

    def test_odd_last_pairs_with_first(self):
        self.assertEqual(hva_lib.verbs_pairs(['n1', 'n2', 'n3']), [('n1', 'n2'), ('n3', 'n1')])


class TestVerbsWaves(unittest.TestCase):
    def test_disjoint_pairs_share_one_wave(self):
        pairs = hva_lib.verbs_pairs(['n1', 'n2', 'n3', 'n4'])
        self.assertEqual(hva_lib.verbs_waves(pairs), [[('n1', 'n2'), ('n3', 'n4')]])

    def test_wrap_around_pair_runs_later(self):
        pairs = hva_lib.verbs_pairs(['n1', 'n2', 'n3', 'n4', 'n5'])
        self.assertEqual(hva_lib.verbs_waves(pairs), [[('n1', 'n2'), ('n3', 'n4')], [('n5', 'n1')]])

    def test_single_node(self):
        self.assertEqual(hva_lib.verbs_waves([('n1',)]), [[('n1',)]])

    def test_no_pairs(self):
        self.assertEqual(hva_lib.verbs_waves([]), [])


class TestNodeAddress(unittest.TestCase):
    def test_vpc_ip(self):
        cluster = {'node_dict': {'n1': {'vpc_ip': '10.0.0.1'}, 'n2': {'bmc_ip': 'NA'}}}
        self.assertEqual(hva_lib.node_address(cluster, 'n1'), '10.0.0.1')
        self.assertEqual(hva_lib.node_address(cluster, 'n2'), 'n2')
        self.assertEqual(hva_lib.node_address({}, 'n3'), 'n3')


if __name__ == '__main__':
    unittest.main()
