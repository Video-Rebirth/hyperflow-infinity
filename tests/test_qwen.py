import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperflow_infinity import prompts, qwen_service
from hyperflow_infinity.qwen import (
    QwenClient,
    fix_dialogue_tags,
    normalize_director_section,
    parse_json,
    parse_memory_json,
    validate_director,
)

VALID = {'prompt': 'integrated_multimodal_description: a overall_soundscape: b non_diegetic_music: c',
         'scene': 'shore', 'scene_change': False, 'beat': 'walks'}


class Parsing(unittest.TestCase):
    def test_first_object_inside_fences_and_prose(self):
        self.assertEqual(parse_json('```json\n{"a": "}"}\n```'), {'a': '}'})
        self.assertEqual(parse_json('note {"a": {"b": 1}} trailing {"c": 2}'), {'a': {'b': 1}})
        self.assertEqual(parse_json('{"prompt": "a:\n\nb:\tc"}'), {'prompt': 'a:\n\nb:\tc'})
        with self.assertRaisesRegex(ValueError, 'unterminated'):
            parse_json('{"a": 1')

    def test_memory_repair_only_closes_the_root(self):
        self.assertEqual(parse_memory_json('{"add_nodes":[],"update_nodes":[],"add_events":[]'),
                         {'add_nodes': [], 'update_nodes': [], 'add_events': []})
        for text in ('{"add_nodes":[],"update_nodes":[],"add_events":[{"action":"unfinished',
                     '{"add_nodes":[],"update_nodes":[],"add_events":[],"invented":[]'):
            with self.assertRaises(ValueError):
                parse_memory_json(text)

    def test_dialogue_tags_and_section_alias(self):
        self.assertEqual(fix_dialogue_tags({'p': ['<tool_call>[English] hi</tool_call>']}), {'p': ['<d>[English] hi</d>']})
        original = {'prompt': 'integrated_multimodality_description: lift a stone'}
        self.assertEqual(normalize_director_section(original)['prompt'], 'integrated_multimodal_description: lift a stone')
        self.assertEqual(original['prompt'], 'integrated_multimodality_description: lift a stone')
        for prompt in ('x integrated_multimodality_description: test',
                       'integrated_multimodality_description: integrated_multimodal_description: x'):
            self.assertEqual(normalize_director_section({'prompt': prompt}), {'prompt': prompt})

    def test_swallowed_n_in_the_music_label_is_restored(self):
        slipped = json.loads('{"prompt": "integrated_multimodal_description: a\\noverall_soundscape: b\\non_diegetic_music: c"}')
        self.assertTrue(slipped['prompt'].endswith('\non_diegetic_music: c'))
        self.assertTrue(normalize_director_section(slipped)['prompt'].endswith('\nnon_diegetic_music: c'))
        intact = {'prompt': 'a\nnon_diegetic_music: c'}
        self.assertIs(normalize_director_section(intact), intact)


class Journal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = QwenClient('http://unused', self.root, max_attempts=3, guidance='be inventive')

    def tearDown(self):
        self.tmp.cleanup()

    def test_request_carries_the_native_video_and_director_suffix(self):
        video = self.root / 'clip.mp4'
        video.write_bytes(b'test')
        r = self.client.build_request('director', 's', 'u', video, attempt=2, budget=2400,
                                      rejected=('{"bad"', 'Missing keys'))
        self.assertEqual(r['images'], [])
        self.assertEqual(r['videos'][0]['path'], str(video))
        self.assertEqual((r['max_tokens'], r['temperature']), (3840, 0.75))
        self.assertTrue(r['user'].startswith('user:\nu\n\nassistant:\n{"bad"\n\nuser:\nThat answer was rejected: Missing keys.'))
        self.assertTrue(r['user'].endswith(prompts.EVIDENCE_PRIORITY + '\nUser creative direction: be inventive\n'))
        memory = self.client.build_request('memory', 's', 'u', None, attempt=1, budget=16000)
        self.assertEqual((memory['user'], memory['max_tokens']), ('u', 8000))

    def test_rejected_answers_exhaust_the_budget_without_extra_requests(self):
        folder = self.root / 'calls/C0002_director'
        folder.mkdir(parents=True)
        for i in range(1, 4):
            (folder / f'{i:02d}.response.json').write_text(json.dumps({'text': json.dumps(VALID)}))
        with patch('urllib.request.urlopen', side_effect=AssertionError('must not call')):
            with self.assertRaisesRegex(RuntimeError, 'exhausted 3'):
                self.client._call('C0002', 'director', 's', 'u', None, 100, ('prompt',), validate=lambda p: 'repeat')
        self.assertFalse((folder / 'validated.json').exists())
        self.assertEqual(json.loads((folder / '01.error.json').read_text())['error'], 'repeat')

    def test_saved_answers_replay_and_uncertain_requests_stop(self):
        folder = self.root / 'calls/C0003_director'
        folder.mkdir(parents=True)
        (folder / '01.response.json').write_text(json.dumps({'text': json.dumps(VALID)}))
        with patch('urllib.request.urlopen', side_effect=AssertionError('must not call')):
            self.assertEqual(self.client._call('C0003', 'director', 's', 'u', None, 100, ('prompt',)), VALID)
        folder = self.root / 'calls/C0001_memory'
        folder.mkdir(parents=True)
        (folder / '01.request.json').write_text('{}')
        with patch('urllib.request.urlopen', side_effect=AssertionError('must not call')):
            with self.assertRaisesRegex(RuntimeError, 'Uncertain'):
                self.client._call('C0001', 'memory', 's', 'u', None, 100, ('a',))

    def test_a_server_error_leaves_the_request_safe_to_resend(self):
        import io
        import urllib.error
        error = urllib.error.HTTPError('http://unused/generate', 500, 'boom', {}, io.BytesIO(b'{"error": "decoder"}'))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaises(urllib.error.HTTPError):
                self.client._call('C0005', 'memory', 's', 'u', None, 100, ('a',))
        folder = self.root / 'calls/C0005_memory'
        self.assertFalse((folder / '01.request.json').exists())
        self.assertEqual(json.loads((folder / '01.http_error.json').read_text())['status'], 500)

    def test_an_unreachable_service_leaves_the_request_safe_to_resend_but_a_timeout_does_not(self):
        import socket
        import urllib.error
        folder = self.root / 'calls/C0006_memory'
        unreachable = urllib.error.URLError(ConnectionRefusedError(111, 'Connection refused'))
        with patch('urllib.request.urlopen', side_effect=unreachable):
            with self.assertRaises(urllib.error.URLError):
                self.client._call('C0006', 'memory', 's', 'u', None, 100, ('a',))
        self.assertFalse((folder / '01.request.json').exists())
        self.assertIn('unreachable', json.loads((folder / '01.http_error.json').read_text())['error'])
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError(socket.timeout('timed out'))):
            with self.assertRaises(urllib.error.URLError):
                self.client._call('C0006', 'memory', 's', 'u', None, 100, ('a',))
        self.assertTrue((folder / '01.request.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'Uncertain'):
            self.client._call('C0006', 'memory', 's', 'u', None, 100, ('a',))

    def test_director_names_the_missing_section_label(self):
        bad = dict(VALID, prompt=VALID['prompt'].replace('overall_soundscape:', 'diegetic_soundscape:'))
        with self.assertRaisesRegex(ValueError, 'exact labels: overall_soundscape:'):
            validate_director(bad)

    def test_director_rejects_material_tags(self):
        bad = dict(VALID, prompt=VALID['prompt'] + ' <Picture 1>')
        folder = self.root / 'calls/C0004_director'
        folder.mkdir(parents=True)
        (folder / '01.response.json').write_text(json.dumps({'text': json.dumps(bad)}))
        (folder / '02.response.json').write_text(json.dumps({'text': json.dumps(VALID)}))
        with patch('urllib.request.urlopen', side_effect=AssertionError('must not call')):
            self.assertEqual(self.client._call('C0004', 'director', 's', 'u', None, 100, ('prompt',)), VALID)
        self.assertIn('material tags', json.loads((folder / '01.error.json').read_text())['error'])

    def test_sglang_mm_kwargs_wrap_processor_tensors(self):
        self.assertEqual(qwen_service.sglang_mm_kwargs({'input_ids': [1]}), {})
        video_in = {'input_ids': [[1, 2]], 'pixel_values_videos': 'pv', 'video_grid_thw': [[5, 30, 54]]}
        self.assertEqual(qwen_service.sglang_mm_kwargs(video_in), {
            'video_data': [{'format': 'processor_output', 'input_ids': [[1, 2]],
                            'pixel_values_videos': 'pv', 'video_grid_thw': [[5, 30, 54]]}]})
        image_in = {'input_ids': [[1, 2]], 'pixel_values': 'pi', 'image_grid_thw': [[1, 20, 20]]}
        self.assertEqual(qwen_service.sglang_mm_kwargs(image_in), {
            'image_data': [{'format': 'processor_output', 'input_ids': [[1, 2]],
                            'pixel_values': 'pi', 'image_grid_thw': [[1, 20, 20]]}]})


if __name__ == '__main__':
    unittest.main()
