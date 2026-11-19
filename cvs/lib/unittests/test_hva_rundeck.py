'''Unit tests for cvs.lib.hva_rundeck.'''

import unittest

from cvs.lib import hva_rundeck


def _test(test_id, status, cls='must_pass', gate_fail=False, **extra):
    entry = {'id': test_id, 'status': status, 'class': cls, 'gate_fail': gate_fail}
    entry.update(extra)
    return entry


def _parsed(tests, summary, exit_code=0):
    report = {'harness': {'version': '0.1.0', 'git_sha': 'abc1234'}, 'summary': summary, 'tests': tests}
    return {'exit_code': exit_code, 'report': report, 'log_tail': '', 'error': ''}


SUMMARY = {'passed': 1, 'failed': 1, 'skipped': 1, 'not_applicable': 0}


class TestMakeMeta(unittest.TestCase):
    def test_labels_hva(self):
        meta = hva_rundeck.make_meta({'cluster_name': 'ruby'}, 'hva_cvs')
        self.assertEqual(meta['cluster'], 'ruby')
        self.assertEqual(meta['version_label'], 'hva')
        self.assertEqual(meta['suite'], 'hva_cvs')


class TestSummarizeCounts(unittest.TestCase):
    def test_with_known(self):
        counts = {'passed': 58, 'failed': 5, 'skipped': 4, 'not_applicable': 5}
        self.assertEqual(hva_rundeck.summarize_counts(counts, 5), '58 pass, 5 fail (5 known), 4 skip, 5 na')

    def test_empty(self):
        self.assertEqual(hva_rundeck.summarize_counts({}), '0 pass, 0 fail, 0 skip, 0 na')


class TestNodeRecord(unittest.TestCase):
    def test_pass_lists_only_non_passing_tests(self):
        tests = [_test('A', 'pass'), _test('S', 'skip', cls='skip', reason='needs dmabuf')]
        record = hva_rundeck.node_record_from_run(_parsed(tests, SUMMARY))
        self.assertEqual(record['status'], 'pass')
        self.assertEqual([i['name'] for i in record['items']], ['S'])
        self.assertEqual(record['items'][0]['status'], 'na')

    def test_gate_failure_fails_node(self):
        tests = [_test('A', 'pass'), _test('B', 'fail', gate_fail=True, reason='boom')]
        record = hva_rundeck.node_record_from_run(_parsed(tests, SUMMARY, exit_code=1))
        self.assertEqual(record['status'], 'fail')
        self.assertEqual(record['items'], [{'name': 'B', 'status': 'fail', 'message': 'B: fail (must_pass) - boom'}])

    def test_tolerated_known_failure_is_na_item(self):
        tests = [_test('A', 'pass'), _test('K', 'fail', cls='known_failure', ticket='ROCM-1')]
        record = hva_rundeck.node_record_from_run(_parsed(tests, SUMMARY))
        self.assertEqual(record['status'], 'pass')
        self.assertEqual(record['items'][0]['status'], 'na')
        self.assertIn('ROCM-1', record['items'][0]['message'])
        self.assertIn('(1 known)', record['items_summary'])

    def test_nothing_ran_is_na(self):
        tests = [_test('N', 'na', cls='na')]
        record = hva_rundeck.node_record_from_run(_parsed(tests, {'not_applicable': 1}))
        self.assertEqual(record['status'], 'na')

    def test_missing_report_fails(self):
        parsed = {'exit_code': 2, 'report': None, 'log_tail': '', 'error': 'hva exited 2 without writing report.json'}
        record = hva_rundeck.node_record_from_run(parsed, peer='n2')
        self.assertEqual(record['status'], 'fail')
        self.assertEqual(record['items_summary'], 'with n2: hva exited 2 without writing report.json')

    def test_peer_named_in_summary(self):
        record = hva_rundeck.node_record_from_run(_parsed([_test('A', 'pass')], SUMMARY), peer='n2')
        self.assertTrue(record['items_summary'].startswith('with n2: '))


class TestRecordRun(unittest.TestCase):
    def test_groups_nodes_and_version(self):
        res = {}
        meta = hva_rundeck.make_meta({}, 'hva_cvs')
        parsed = _parsed([_test('A', 'pass')], SUMMARY)
        hva_rundeck.record_run(res, 'verbs', {'n1': (parsed, 'n2'), 'n2': (parsed, 'n1')}, meta=meta)
        self.assertEqual(sorted(res['groups']['verbs']['nodes']), ['n1', 'n2'])
        self.assertEqual(res['_meta']['version'], '0.1.0 (abc1234)')
        self.assertEqual(res['_meta']['suite'], 'hva_cvs')

    def test_rerun_overwrites_group(self):
        res = {}
        hva_rundeck.record_run(res, 'hip', {'n1': (_parsed([_test('B', 'fail', gate_fail=True)], SUMMARY, 1), None)})
        hva_rundeck.record_run(res, 'hip', {'n1': (_parsed([_test('A', 'pass')], SUMMARY), None)})
        self.assertEqual(res['groups']['hip']['nodes']['n1']['status'], 'pass')


class TestBuildNodeRecord(unittest.TestCase):
    def test_unknown_status_is_na(self):
        self.assertEqual(hva_rundeck.build_node_record('weird')['status'], 'na')


if __name__ == '__main__':
    unittest.main()
