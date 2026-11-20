'''
Payload/render test for the HIP/Verbs Acceptance status_matrix deck.

A registered hva_cvs profile renders the run card, health overview, and the
node x run (hip, verbs) matrix built by hva_rundeck, hides measurements, and
must not use inference gate vocabulary.
'''

import copy
import unittest

from cvs.lib import hva_lib, hva_rundeck
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _parsed(tests, summary, exit_code=0):
    report = {'harness': {'version': '0.1.0', 'git_sha': 'v0.1.0'}, 'summary': summary, 'tests': tests}
    return {'exit_code': exit_code, 'report': report, 'log_tail': '', 'error': ''}


def _results():
    res = {}
    meta = {'cluster': 'helios', 'version': '—', 'version_label': 'hva', 'suite': 'hva_cvs', 'generated_at': 't0'}
    hip_pass = _parsed(
        [
            {'id': 'Stream_Create', 'status': 'pass', 'class': 'must_pass', 'gate_fail': False},
            {
                'id': 'PointerGetAttributes_DeviceVsHost',
                'status': 'fail',
                'class': 'known_failure',
                'gate_fail': False,
                'ticket': 'ROCM-2696',
            },
        ],
        {'passed': 1, 'failed': 1, 'skipped': 0, 'not_applicable': 0},
    )
    hip_fail = _parsed(
        [
            {'id': 'Stream_Create', 'status': 'pass', 'class': 'must_pass', 'gate_fail': False},
            {
                'id': 'Memcpy_D2H_Ordering',
                'status': 'fail',
                'class': 'must_pass',
                'gate_fail': True,
                'reason': 'mismatch at 0x40',
            },
        ],
        {'passed': 1, 'failed': 1, 'skipped': 0, 'not_applicable': 0},
        exit_code=1,
    )
    hva_rundeck.record_run(res, 'hip', {'node-a': (hip_pass, None), 'node-b': (hip_fail, None)}, meta=meta)
    verbs = _parsed(
        [{'id': 'vat_rc_write', 'status': 'pass', 'class': 'must_pass', 'gate_fail': False}],
        {'passed': 1, 'failed': 0, 'skipped': 0, 'not_applicable': 0},
    )
    hva_rundeck.record_run(res, 'verbs', {'node-a': (verbs, 'node-b'), 'node-b': (verbs, 'node-a')}, meta=meta)
    return res


class TestHvaDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile('hva_cvs')
        self.results = _results()
        store = {'cvs_results_dict': self.results, 'inf_res_dict': self.results}
        self.payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version='dev')
        self.html = render_rundeck_html(self.payload)

    def test_profile_cards_overview_and_viewer(self):
        types = [card['type'] for card in self.profile['cards']]
        self.assertEqual(
            types,
            ['run_card', 'lifecycle_timeline', 'status_overview', 'metric_charts', 'table'],
        )
        self.assertEqual(self.profile['sources']['results'], 'hva_res_dict')
        self.assertEqual(self.profile['dataset_builder'], 'status_matrix')
        self.assertTrue(self.profile['interactive_viewer'])
        self.assertIn('Pass rate', self.html)
        linked_payload = SummaryMetaApplier(resolve_report_config(self.profile)).apply(copy.deepcopy(self.payload))
        self.assertIn('hva_run_deck_viewer.html', render_rundeck_html(linked_payload))
        self.assertNotIn('Measurements', self.html)

    def test_lifecycle_draws_hip_and_verbs(self):
        store = {
            'cvs_results_dict': self.results,
            'inf_res_dict': self.results,
            'lifecycle_report': {'health': [('hip', 75.0, 's'), ('verbs', 5.5, 's')]},
        }
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version='dev')
        html = render_rundeck_html(payload)
        self.assertIn('Lifecycle timeline', html)
        self.assertIn('75.0s', html)
        self.assertIn('5.5s', html)

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload['overall_status'], 'fail')

    def test_run_card_names_hva_version(self):
        self.assertIn('helios', self.html)
        self.assertIn('0.1.0 (v0.1.0)', self.html)
        self.assertNotIn('ANC version', self.html)

    def test_status_matrix_shows_gate_and_known_failures(self):
        self.assertIn('results-table', self.html)
        for text in ('hip', 'verbs', 'node-a', 'node-b', 'with node-b'):
            self.assertIn(text, self.html)
        self.assertIn(
            hva_lib.describe_test(
                {'id': 'Memcpy_D2H_Ordering', 'status': 'fail', 'class': 'must_pass', 'reason': 'mismatch at 0x40'}
            ),
            self.html,
        )
        self.assertIn('ROCM-2696', self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn('Gate matrix', self.html)
        self.assertNotIn('hover for tier status', self.html)


if __name__ == '__main__':
    unittest.main()
