import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from hyperflow_infinity.engine import request_contract, validate_conditioning, validate_observed_progress

BINDING = 'Robinson is the same person as <Picture 1>.'

IMAGE_OPENING_AUDIT = dict(
    contract='ref2va_opening_v1', latent_keyframes=0, boundary_rows=0, reference_clock_advance_removed=0,
    rgb_reference_count=1, reference_video_count=0, vae_reference_encode_calls=1, vae_reference_frames=[1],
    reference_latent_frames=[1], text_reference_image_tokens=405, text_reference_video_tokens=0,
    text_reference_vision_tokens=405, all_condition_video_rows=405, reference_audio_rows=0,
    target_video_origin=12, target_audio_origin=12, text_rows=11, reference_sha256=hashlib.sha256(b'image').hexdigest())


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'identity.png').write_bytes(b'image')
        (self.root / 'opening.mp4').write_bytes(b'video')
        (self.root / 'tail.pt').write_bytes(b'latent')
        sha = hashlib.sha256(b'image').hexdigest()
        self.config = {'height': 480, 'width': 864, 'seed': 100, 'identity_anchor': {
            'path': 'identity.png', 'sha256': sha, 'effective_from_chunk': 2, 'character_id': 'robinson',
            'binding': BINDING}, 'opening_reference': {'kind': 'image', 'source': 'identity_image',
                                                       'path': 'identity.png', 'sha256': sha}}
        self.plan = {'prompt': 'integrated_multimodal_description: walk', 'scene_change': True}

    def tearDown(self):
        self.tmp.cleanup()

    def continuation(self):
        return request_contract(self.config, self.root, 2, self.plan, {'latents': str(self.root / 'tail.pt')})

    def video_config(self):
        config = copy.deepcopy(self.config)
        config['opening_reference'] = {'kind': 'video', 'source': 'user', 'path': 'opening.mp4', 'frames': 120,
                                       'sha256': hashlib.sha256(b'video').hexdigest()}
        return config

    def test_opening_and_continuation_geometry(self):
        first = request_contract(self.config, self.root, 1, self.plan, None)
        self.assertEqual((first['mode'], first['generated_frames'], first['kept_frames'], first['overlap_frames'],
                          first['seed']), ('ref2va', 345, 345, 0, 100))
        self.assertEqual(first['generation_contract'], 'ref2va_opening_v1')
        self.assertEqual([m['kind'] for m in first['media']], ['image'])
        self.assertIsNone(first['boundary_latents'])
        self.assertIn('Use <Picture 1> as the visual reference for the opening.', first['prompt'])
        self.assertEqual(first['director_prompt'], self.plan['prompt'])
        video = request_contract(self.video_config(), self.root, 1, self.plan, None)
        self.assertIn('Use <Video 1> as the visual reference', video['prompt'])
        self.assertEqual(video['media'][0]['frames'], 120)
        later = self.continuation()
        self.assertEqual((later['mode'], later['generated_frames'], later['kept_frames'], later['seed']),
                         ('ref2va', 141, 119, 101))
        self.assertEqual(later['prompt'], f'integrated_multimodal_description: {BINDING}  walk')
        self.assertNotIn('opening_reference', later)
        with self.assertRaises(ValueError):
            request_contract(self.config, self.root, 1, dict(self.plan, scene_change=False), None)

    def test_opening_audit_must_match_the_request(self):
        req = request_contract(self.config, self.root, 1, self.plan, None)
        result = {'actual_frames': 345, 'reference_inputs': copy.deepcopy(req['media']),
                  'conditioning_audit': dict(IMAGE_OPENING_AUDIT)}
        validate_conditioning(req, result)
        for key, value in [('latent_keyframes', 7), ('boundary_rows', 1), ('vae_reference_encode_calls', 0),
                           ('vae_reference_frames', [17]), ('text_reference_image_tokens', 0),
                           ('text_reference_video_tokens', 8), ('reference_audio_rows', 1),
                           ('target_audio_origin', 13), ('target_video_origin', 20), ('reference_sha256', 'x'),
                           ('contract', 'ref2va_identity_tail_v1')]:
            bad = copy.deepcopy(result)
            bad['conditioning_audit'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_conditioning(req, bad)
        video = request_contract(self.video_config(), self.root, 1, self.plan, None)
        audit = dict(IMAGE_OPENING_AUDIT, rgb_reference_count=0, reference_video_count=1, vae_reference_frames=[120],
                     reference_latent_frames=[37], text_reference_image_tokens=0, text_reference_video_tokens=640,
                     text_reference_vision_tokens=640,
                     target_video_origin=48, target_audio_origin=48, reference_sha256=video['media'][0]['sha256'])
        validate_conditioning(video, {'actual_frames': 345, 'reference_inputs': copy.deepcopy(video['media']),
                                      'conditioning_audit': audit})
        with self.assertRaises(ValueError):
            validate_conditioning(video, {'actual_frames': 345, 'reference_inputs': copy.deepcopy(video['media']),
                                          'conditioning_audit': dict(audit, vae_reference_frames=[1])})

    def test_audit_must_match_the_request(self):
        req = self.continuation()
        audit = dict(contract='ref2va_identity_tail_v1', latent_keyframes=7, target_position_alignment=True,
                     rgb_reference_count=1, vae_reference_encode_calls=1, text_reference_vision_tokens=405,
                     reference_video_count=0, reference_audio_rows=0, all_condition_video_rows=3240,
                     boundary_rows=2835, target_video_origin=12, text_rows=11)
        result = {'actual_frames': 141, 'reference_inputs': copy.deepcopy(req['media']), 'conditioning_audit': audit}
        validate_conditioning(req, result)
        for key, value in [('rgb_reference_count', 0), ('vae_reference_encode_calls', 0),
                           ('text_reference_vision_tokens', 0), ('reference_video_count', 1),
                           ('reference_audio_rows', 1), ('target_video_origin', 11), ('latent_keyframes', 6)]:
            bad = copy.deepcopy(result)
            bad['conditioning_audit'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_conditioning(req, bad)
        (self.root / 'tail.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            validate_conditioning(req, result)

    def test_changed_references_are_refused(self):
        (self.root / 'opening.mp4').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'Opening reference changed'):
            request_contract(self.video_config(), self.root, 1, self.plan, None)
        (self.root / 'identity.png').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'Fixed identity image changed'):
            self.continuation()
        ungenerated = copy.deepcopy(self.config)
        ungenerated['opening_reference'].update(source='qwen_image', sha256=None)
        with self.assertRaises(ValueError):
            request_contract(ungenerated, self.root, 1, self.plan, None)

    def test_progress_rules(self):
        previous = {'prompt': f'{BINDING} p', 'director_prompt': 'p', 'scene': 'Shore'}
        self.assertIn('verbatim', validate_observed_progress({'prompt': 'p'}, previous, None))
        self.assertIsNone(validate_observed_progress({'prompt': 'q', 'scene': 'Shore'}, previous, None))
        stuck = {'chunks': ['C1', 'C2', 'C3']}
        self.assertIsNotNone(validate_observed_progress({'prompt': 'q', 'scene': 'shore', 'scene_change': True}, previous, stuck))
        self.assertIsNone(validate_observed_progress({'prompt': 'q', 'scene': 'Cave', 'scene_change': True}, previous, stuck))


if __name__ == '__main__':
    unittest.main()
