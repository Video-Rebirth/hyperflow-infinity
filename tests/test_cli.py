import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from hyperflow_infinity import cli

REPO = Path(__file__).resolve().parents[1]


class Init(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.image = self.base / 'identity.png'
        Image.new('RGB', (32, 32)).save(self.image)
        self.video = self.base / 'opening.mp4'
        self.video.write_bytes(b'not decoded: probe is patched')
        self.qwen_image = self.base / 'qwen-image'
        self.qwen_image.mkdir()
        (self.qwen_image / 'model_index.json').write_text('{}')

    def tearDown(self):
        self.tmp.cleanup()

    def init(self, name, *extra):
        root = self.base / name
        cli.main(['init', '--run-dir', str(root), '--config', str(REPO / 'configs/alice.json'),
                  '--h3-dir', str(self.base), '--hyperflow-weights', str(self.base / 'w.safetensors'),
                  '--generation-gpus', '0,1', '--conditioning-gpu', '2', *extra])
        return root, json.loads((root / 'config.json').read_text())

    def test_init_is_paused_frozen_and_hash_checked(self):
        root, config = self.init('run', '--identity-image', str(self.image))
        self.assertTrue(json.loads((root / 'control.json').read_text())['pause_after_commit'])
        self.assertNotIn('identity', config)
        anchor = config['identity_anchor']
        self.assertEqual((anchor['effective_from_chunk'], anchor['character_id'], anchor['source']),
                         (2, 'alice', 'user_image'))
        self.assertEqual(anchor['sha256'], cli.sha256_file(self.image))
        opening = config['opening_reference']
        self.assertEqual((opening['kind'], opening['source'], opening['path']), ('image', 'identity_image', anchor['path']))
        self.assertTrue((root / 'runtime/code/hyperflow_infinity/engine.py').is_file())
        cli.verify(root)
        with self.assertRaises(ValueError):
            self.init('run', '--identity-image', str(self.image))
        (root / 'runtime/code/hyperflow_infinity/engine.py').write_text('changed')
        with self.assertRaises(ValueError):
            cli.verify(root)

    def test_without_an_image_qwen_image_draws_the_opening_and_identity(self):
        from hyperflow_infinity.engine import Engine
        with self.assertRaisesRegex(ValueError, 'qwen-image-dir'):
            self.init('missing')
        self.assertFalse((self.base / 'missing').exists())
        root, config = self.init('run', '--qwen-image-dir', str(self.qwen_image))
        opening, anchor = config['opening_reference'], config['identity_anchor']
        self.assertEqual((opening['source'], opening['sha256'], anchor['source'], anchor['path']),
                         ('qwen_image', None, 'opening_reference', opening['path']))
        self.assertIn('dark blonde hair', opening['prompt'])
        cli.verify(root)

        class Generator:
            def make_image(self, spec, path):
                self.spec = spec
                Path(path).write_bytes(b'drawn')
                return {'load_seconds': 1.0, 'generate_seconds': 2.0, 'peak_allocated_gib': 3.0}
        generator = Generator()
        engine = Engine(config, generator)
        engine.make_opening_image()
        self.assertEqual((generator.spec['seed'], generator.spec['width']), (config['seed'], 1664))
        drawn = cli.sha256_file(root / opening['path'])
        self.assertEqual((engine.config['opening_reference']['sha256'], engine.config['identity_anchor']['sha256']),
                         (drawn, drawn))
        resumed = Engine(json.loads((root / 'config.json').read_text()), generator=None)
        resumed.make_opening_image()
        self.assertEqual((resumed.config['opening_reference']['sha256'], resumed.config['identity_anchor']['sha256']),
                         (drawn, drawn))

    def test_video_opening_binds_the_identity_from_segment_1(self):
        from hyperflow_infinity.engine import Engine
        for frames, fps in ((10, 24.0), (400, 24.0), (300, 20.0)):
            with self.subTest(frames=frames, fps=fps), \
                    patch('hyperflow_infinity.media.probe', return_value={'frames': frames, 'fps': fps}), \
                    self.assertRaisesRegex(ValueError, 'opening video must last'):
                self.init('bad', '--opening-reference', str(self.video))
        self.assertFalse((self.base / 'bad').exists())
        with patch('hyperflow_infinity.media.probe', return_value={'frames': 150, 'fps': 30.0}):
            root, config = self.init('run', '--opening-reference', str(self.video))
        opening = config['opening_reference']
        self.assertEqual((opening['kind'], opening['frames'], opening['use_audio']), ('video', 150, False))
        self.assertEqual((config['identity_anchor']['source'], config['identity_anchor']['sha256']),
                         ('segment_1_best_face', None))
        cli.verify(root)
        (root / opening['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            cli.verify(root)
        (root / opening['path']).write_bytes(self.video.read_bytes())
        frames = []
        for i in range(12):
            (self.base / f'f{i}.png').write_bytes(f'frame {i}'.encode())
            frames.append({'frame_index': i, 'path': str(self.base / f'f{i}.png')})
        first = {'chunk_id': 'C0001', 'raw_video': 'raw.mp4', 'overlap_frames': 0, 'kept_frames': 345}
        engine = Engine(config, generator=None)
        with patch('hyperflow_infinity.engine.evidence_frames', return_value=frames), \
                patch.object(engine.client, 'pick_identity', return_value=None):
            with self.assertRaisesRegex(ValueError, 'identity-image'):
                engine.bind_identity(first)
        with patch('hyperflow_infinity.engine.evidence_frames', return_value=frames), \
                patch.object(engine.client, 'pick_identity', return_value=7) as pick:
            engine.bind_identity(first)
        self.assertEqual(pick.call_args.args[2]['id'], 'alice')
        self.assertEqual((root / 'references/identity.png').read_bytes(), b'frame 7')
        resumed = Engine(json.loads((root / 'config.json').read_text()), generator=None)
        self.assertEqual(resumed.config['identity_anchor']['sha256'], cli.sha256_file(self.base / 'f7.png'))

    def test_a_separate_opening_image_keeps_the_identity_image(self):
        other = self.base / 'opening.png'
        Image.new('RGB', (32, 32), 'navy').save(other)
        root, config = self.init('run', '--identity-image', str(self.image), '--opening-reference', str(other))
        self.assertEqual((config['opening_reference']['source'], config['opening_reference']['sha256']),
                         ('user', cli.sha256_file(other)))
        self.assertEqual(config['identity_anchor']['sha256'], cli.sha256_file(self.image))
        cli.verify(root)

    def test_gpu_sets_must_be_disjoint(self):
        for gen, cond in (('0', '1'), ('0,0', '1'), ('0,1', '1')):
            with self.subTest(gen=gen, cond=cond), self.assertRaises(ValueError):
                cli.split_gpus(gen, cond)
        self.assertEqual(cli.split_gpus('0,1,2,3', '4'), ['0', '1', '2', '3'])

    def test_attention_backend_is_recorded(self):
        _, config = self.init('fa3_run', '--identity-image', str(self.image), '--attention-backend', 'fa3')
        self.assertEqual(config['attention_backend'], 'fa3')


if __name__ == '__main__':
    unittest.main()
