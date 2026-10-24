"""OpenRouter transport and journal rules. No network, no paid calls."""
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperflow_infinity import cli
from hyperflow_infinity import openrouter as api
from hyperflow_infinity.qwen import QwenClient

REPO = Path(__file__).resolve().parents[1]
MESSAGES = [{'role': 'user', 'content': [{'type': 'text', 'text': 'JSON'}]}]


def cfg(model='qwen/qwen3-vl-32b-instruct', **extra):
    return dict(vlm_provider='openrouter', openrouter_model=model, **extra)


class Config(unittest.TestCase):
    def test_modes_and_rejections(self):
        api.validate_config({})
        self.assertEqual(api.video_mode(cfg()), 'frames')
        self.assertEqual(api.video_mode(cfg('qwen/qwen3.8-flash')), 'native')
        self.assertEqual(api.video_mode(cfg('qwen/qwen3.8-flash', openrouter_video_mode='frames')), 'frames')
        for bad in ({'vlm_provider': 'other'}, cfg('unknown/model'), cfg(openrouter_video_mode='native'),
                    cfg(openrouter_video_mode='sometimes')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                api.validate_config(bad)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
            api.validate_config(cfg(), credentials=True)


@patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret-test-key'})
class Payload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / 'clip.mp4'
        self.video.write_bytes(b'mp4')
        self.request = dict(system='system', user='observe', max_tokens=42, temperature=.5, images=[],
                            videos=[dict(path=str(self.video), sha256=hashlib.sha256(b'mp4').hexdigest(),
                                         label='Previous generated segment')])

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_video_goes_before_the_text_and_nothing_secret_is_journaled(self):
        payload, audit = api.prepare(cfg('qwen/qwen3.8-flash'), self.request)
        content = payload['messages'][1]['content']
        self.assertEqual([c['type'] for c in content], ['text', 'video_url', 'text'])
        self.assertEqual((content[0]['text'], content[-1]['text']), ('Previous generated segment', 'observe'))
        self.assertEqual(payload['reasoning'], {'enabled': False})
        self.assertEqual((audit['video_mode'], audit['media'][0]['kind']), ('native', 'video'))
        self.assertNotIn('secret-test-key', json.dumps([payload, audit]))
        self.assertNotIn('base64', json.dumps(audit))

    def test_frames_mode_uses_the_sampler_and_records_its_timestamps(self):
        sampled = ([{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}],
                   {'sampled_frames': [{'index': 0, 'seconds': 0}]})
        with patch.object(api, 'sampled_frames', return_value=sampled) as sampler:
            payload, audit = api.prepare(cfg(), self.request)
        self.assertEqual(sampler.call_args.args, (self.video, 2.0))
        self.assertEqual(payload['messages'][1]['content'][-2]['type'], 'image_url')
        self.assertNotIn('reasoning', payload)
        self.assertEqual(audit['media'][0]['sampled_frames'][0]['seconds'], 0)

    def test_images_are_sent_with_their_labels(self):
        image = Path(self.tmp.name) / 'f1.png'
        image.write_bytes(b'png')
        request = dict(self.request, videos=[], images=[{'path': str(image), 'label': 'Image 1:'}])
        payload, audit = api.prepare(cfg('qwen/qwen3.8-flash'), request)
        content = payload['messages'][1]['content']
        self.assertEqual([c['type'] for c in content], ['text', 'image_url', 'text'])
        self.assertEqual(content[0]['text'], 'Image 1:')
        self.assertEqual(audit['media'][0]['sha256'], hashlib.sha256(b'png').hexdigest())

    def test_complete_reports_tokens_like_the_local_service(self):
        import io
        body = {'id': 'gen-1', 'model': 'qwen/qwen3.5-flash-02-23', 'provider': 'Alibaba',
                'choices': [{'finish_reason': 'stop', 'message': {'content': [{'type': 'text', 'text': '{"ok": 1}'}]}}],
                'usage': {'prompt_tokens': 1740, 'completion_tokens': 536}}

        class Opener:
            def open(self, request, timeout):
                self.request = request
                return io.BytesIO(json.dumps(body).encode())

        opener = Opener()
        with patch.object(api.urllib.request, 'build_opener', return_value=opener):
            response = api.complete({'model': 'm'})
        self.assertEqual(opener.request.get_header('Authorization'), 'Bearer secret-test-key')
        self.assertEqual((response['text'], response['input_tokens'], response['output_tokens'], response['upstream_provider']),
                         ('{"ok": 1}', 1740, 536, 'Alibaba'))
        self.assertNotIn('transport_error', response)

    def test_dns_failure_is_a_known_rejection_but_a_timeout_is_not(self):
        import socket
        import urllib.error

        class Opener:
            def __init__(self, error):
                self.error = error

            def open(self, request, timeout):
                raise self.error

        dns = urllib.error.URLError(socket.gaierror(-3, 'Temporary failure in name resolution'))
        with patch.object(api.urllib.request, 'build_opener', return_value=Opener(dns)):
            response = api.complete({'model': 'm'})
        self.assertIn('unreachable', response['transport_error'])
        timeout = urllib.error.URLError(TimeoutError('timed out'))
        with patch.object(api.urllib.request, 'build_opener', return_value=Opener(timeout)):
            with self.assertRaises(urllib.error.URLError):
                api.complete({'model': 'm'})

    def test_a_changed_video_is_refused(self):
        self.video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            api.prepare(cfg('qwen/qwen3.8-flash'), self.request)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('qwen_vl_utils'),
                         'needs torch and qwen-vl-utils')
    def test_sampler_matches_qwen_vl_utils_on_a_short_clip(self):
        import av
        import numpy as np
        import torch
        from qwen_vl_utils.vision_process import smart_nframes
        path = Path(self.tmp.name) / 'real.mp4'
        with av.open(str(path), 'w') as out:
            stream = out.add_stream('libx264', rate=24)
            stream.width = stream.height = 64
            stream.pix_fmt = 'yuv420p'
            for i in range(119):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        parts, audit = api.sampled_frames(path)
        n = smart_nframes({'fps': 2.0}, 119, 24)
        self.assertEqual([f['index'] for f in audit['sampled_frames']], torch.linspace(0, 118, n).round().long().tolist())
        self.assertAlmostEqual(audit['sampled_frames'][-1]['seconds'], 118 / 24)
        self.assertEqual(len(parts), 2 * n)


class Journal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = QwenClient('http://unused', self.root, max_attempts=3, provider=cfg())
        self.folder = self.root / 'calls/C0001_compact'

    def tearDown(self):
        self.tmp.cleanup()

    def call(self):
        return self.client._call('C0001', 'compact', 's', 'u', None, 30, ('summary',))

    def test_local_provider_is_the_default(self):
        self.assertIsNone(QwenClient('http://unused', self.root, max_attempts=3).provider)
        self.assertIsNone(QwenClient('http://unused', self.root, 3, provider={'vlm_provider': 'local'}).provider)
        with self.assertRaises(ValueError):
            QwenClient('http://unused', self.root, 3, provider=cfg('unknown/model'))

    def test_a_lost_response_is_never_resubmitted(self):
        with patch.object(api, 'prepare', return_value=({}, {'provider': 'openrouter'})), \
                patch.object(api, 'complete', side_effect=TimeoutError) as complete:
            with self.assertRaises(TimeoutError):
                self.call()
            with self.assertRaisesRegex(RuntimeError, 'Uncertain'):
                self.call()
        self.assertEqual(complete.call_count, 1)
        self.assertEqual(json.loads((self.folder / '01.request.json').read_text())['transport'], {'provider': 'openrouter'})

    def test_rejected_answers_spend_the_attempt_budget_once(self):
        with patch.object(api, 'prepare', return_value=({}, {})), \
                patch.object(api, 'complete', return_value={'text': '{}'}) as complete:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'exhausted 3'):
                    self.call()
        self.assertEqual(complete.call_count, 3)

    def test_an_http_rejection_stops_the_run_and_is_safe_to_resend(self):
        rejected = {'text': '', 'provider': 'openrouter', 'http_status': 401, 'transport_error': 'OpenRouter HTTP 401'}
        with patch.object(api, 'prepare', return_value=({}, {})), \
                patch.object(api, 'complete', return_value=rejected) as complete:
            with self.assertRaisesRegex(RuntimeError, '401'):
                self.call()
            self.assertFalse((self.folder / '01.request.json').exists())
            self.assertEqual(json.loads((self.folder / '01.http_error.json').read_text())['http_status'], 401)
            with self.assertRaisesRegex(RuntimeError, '401'):
                self.call()
        self.assertEqual(complete.call_count, 2)


class Cli(unittest.TestCase):
    def test_init_records_the_provider_and_launch_needs_the_key(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'run'
            (base / 'model_index.json').write_text('{}')
            argv = ['init', '--run-dir', str(root), '--config', str(REPO / 'configs/alice.json'),
                    '--h3-dir', str(base), '--hyperflow-weights', str(base / 'w.safetensors'), '--qwen-image-dir', str(base),
                    '--generation-gpus', '0,1', '--conditioning-gpu', '2', '--vlm-provider', 'openrouter']
            with self.assertRaisesRegex(ValueError, 'openrouter_model'):
                cli.main(argv)
            self.assertFalse(root.exists())
            cli.main(argv + ['--openrouter-model', 'qwen/qwen3.8-flash'])
            config = json.loads((root / 'config.json').read_text())
            self.assertEqual((config['vlm_provider'], config['openrouter_model']), ('openrouter', 'qwen/qwen3.8-flash'))
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                cli.main(['launch', '--run-dir', str(root), '--resume'])
            self.assertTrue(json.loads((root / 'control.json').read_text())['pause_after_commit'])


if __name__ == '__main__':
    unittest.main()
