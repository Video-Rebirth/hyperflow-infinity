"""Latent handoff geometry and the opening path. Needs PyTorch; the native-block test also needs diffusers."""
import copy
import inspect
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

try:
    import torch
except ImportError:
    torch = None
try:
    import diffusers.modular_pipelines.minimax_h3  # noqa: F401
    has_h3 = True
except Exception:
    has_h3 = False


@unittest.skipIf(torch is None, 'PyTorch not installed')
class Handoff(unittest.TestCase):
    def test_tail_is_exact_and_invalid_latents_are_refused(self):
        from hyperflow_infinity import handoff
        anchor = {'sha256': 'x'}
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'z.pt'
            z = torch.randn(1, 16, 42, 30, 54)
            torch.save({'video': z}, p)
            handoff.set_boundary(p, 'ref2va', identity_anchor=anchor)
            self.assertTrue(torch.equal(handoff._state['tail'], z[:, :, -7:]))
            self.assertIsNone(handoff._state['opening'])
            with self.assertRaises(ValueError):
                handoff.set_boundary(p, 'ref2va', identity_anchor=None)
            with self.assertRaises(ValueError):
                handoff.set_boundary(p, 'ref2va', opening={'kind': 'image', 'sha256': 'x'})
            z[0, 0, 0, 0, 0] = float('nan')
            torch.save({'video': z}, p)
            with self.assertRaises(ValueError):
                handoff.set_boundary(p, 'ref2va', identity_anchor=anchor)
        with self.assertRaises(ValueError):
            handoff.set_boundary(None, 'ref2va', identity_anchor=anchor)
        with self.assertRaises(ValueError):
            handoff.set_boundary(None, 'ref2va', opening={'kind': 'audio', 'sha256': 'x'})
        with self.assertRaises(ValueError):
            handoff.set_boundary(None, 'ref2va', identity_anchor=anchor, opening={'kind': 'image', 'sha256': 'x'})
        handoff.set_boundary(None, 'ref2va', opening={'kind': 'video', 'sha256': 'x'})
        self.assertIsNone(handoff._state['tail'])

    def test_guides_take_the_target_positions_and_the_clock_starts_after_the_image(self):
        from hyperflow_infinity.handoff import align_positions
        text, image, rows, guides, target, audio = 11, 4, 3, 7, 30, 5
        n = guides * rows
        count = text + image + n + target + audio
        pos = torch.zeros(count, 3)
        vi = torch.arange(text, text + image + n + target)
        ai = torch.arange(text + image + n + target, count)
        ti = torch.arange(text)
        # Native layout: one clock tick per reference (1 image + 7 guides), target after them.
        pos[vi[:image], 0] = text
        for k in range(guides):
            pos[vi[image + k * rows:image + (k + 1) * rows], 0] = text + 1 + k
        pos[vi[image + n:], 0] = text + 1 + guides + torch.arange(target).float() // rows
        pos[ai, 0] = text + 1 + guides
        ncv = image + n
        out, *_ = align_positions((pos, None, vi, ai, ti, ncv, 0), guides, rows)
        self.assertEqual(float(out[vi[ncv], 0]), text + 1)
        self.assertEqual(float(out[ai[0], 0]), text + 1)
        self.assertTrue(torch.equal(out[vi[image:ncv]], out[vi[ncv:ncv + n]]))
        self.assertTrue(torch.equal(out[vi[:image]], pos[vi[:image]]))


@unittest.skipIf(torch is None or not has_h3 or not shutil.which('ffmpeg'), 'needs PyTorch, diffusers MiniMax-H3, ffmpeg')
class NativeOpening(unittest.TestCase):
    """The real diffusers Ref2VA blocks with stub components: the opening keeps the native layout."""

    def setUp(self):
        import hashlib

        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / 'opening.png'
        Image.new('RGB', (32, 32), 'navy').save(self.image)
        self.video = self.root / 'opening.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:r=12',
                        '-frames:v', '72', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(self.video)], check=True)
        sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.image, self.video)}
        self.configs = {
            'image': {'height': 480, 'width': 864, 'seed': 1, 'opening_reference': {
                'kind': 'image', 'source': 'user', 'path': self.image.name, 'sha256': sha[self.image]}},
            'video': {'height': 480, 'width': 864, 'seed': 1, 'opening_reference': {
                'kind': 'video', 'source': 'user', 'path': self.video.name, 'sha256': sha[self.video], 'frames': 72}}}

    def tearDown(self):
        from hyperflow_infinity import handoff
        handoff._state['opening'] = None
        self.tmp.cleanup()

    def test_video_loader_keeps_the_source_frame_rate_and_drops_sound(self):
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference

        from hyperflow_infinity.generator import load_reference
        ref = load_reference({'kind': 'video', 'path': str(self.video), 'frames': 72})
        self.assertIsInstance(ref, MiniMaxH3VideoReference)
        self.assertEqual((len(ref.frames), ref.fps, ref.has_audio), (72, 12.0, False))
        with self.assertRaises(ValueError):
            load_reference({'kind': 'video', 'path': str(self.video), 'frames': 71})

    def native_result(self, request):
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.modular_pipelines.minimax_h3 import before_denoise as bd
        from diffusers.modular_pipelines.minimax_h3 import before_encoder as be
        from diffusers.modular_pipelines.minimax_h3 import encoders as en
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        from hyperflow_infinity import handoff
        from hyperflow_infinity.generator import load_reference
        handoff.install()
        handoff.set_boundary(None, 'ref2va', opening=request['opening_reference'])
        comp = NS(canvas_multiple=32, min_duration=5, max_duration=15, fps=24, vae_frames_per_chunk=17,
                  vae_latents_per_chunk=5, audio_sampling_rate=32000, pixel_mean=(.5, .5, .5), pixel_std=(.5, .5, .5),
                  config=NS(canvas_short_edge=32, canvas_max_pixels=1024, reference_image_short_edge=32),
                  image_processor=VaeImageProcessor(vae_scale_factor=16))
        state = PipelineState(values={'references': [load_reference(request['media'][0])],
                                      'height': 480, 'width': 864, 'num_frames': 345})
        be.MiniMaxH3Ref2VASetupStep()(comp, state)

        class Images:
            merge_size = 2

            def __call__(self, **kw):
                return {'pixel_values': torch.zeros(1), 'image_grid_thw': torch.tensor([[1, 2, 2]])}

        class Videos:
            temporal_patch_size = 2

            def __call__(self, videos, **kw):
                return {'pixel_values_videos': torch.zeros(1), 'video_grid_thw': torch.tensor([[len(videos[0]) // 2, 2, 2]])}

        en.MiniMaxH3Ref2VATextEncoderStep()._gather_vision_features(
            NS(image_processor=Images(), video_processor=Videos()), state.values['normalized_references'], 24)

        class VAE:
            config = NS(latents_mean=[0.] * 16, latents_std=[1.] * 16)

            def encode(self, pixels, **kw):
                frames = 1 if pixels.shape[2] == 1 else ((pixels.shape[2] - 5) // 17) * 5 + 2
                z = torch.ones(1, 16, frames, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
                return (NS(sample=lambda **kw: z),)
        comp.vae = VAE()
        comp.audio_vae = NS(config=NS(latents_mean=[0.] * 4, latents_std=[1.] * 4))
        comp._execution_device = 'cpu'
        comp.keyframe_encode_seed = 42
        en.MiniMaxH3Ref2VAReferenceEncoderStep()(comp, state)
        self.assertEqual(len(state.values['condition_latents']), 1)
        args = (torch.ones(11, dtype=torch.long), state.values['normalized_references'],
                state.values['condition_latents'], [], 102, 30, 54, 600, (1, 2, 2), 2, 2, 0)
        wrapper = bd.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
        original = inspect.getclosurevars(wrapper).nonlocals['original_layout']
        for a, b in zip(wrapper(*args), original(*args)):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            else:
                self.assertEqual(a, b)
        return {'actual_frames': 345, 'reference_inputs': copy.deepcopy(request['media']),
                'conditioning_audit': copy.deepcopy(handoff.audit())}

    def test_native_blocks_keep_the_opening_layout_and_the_audit_fails_closed(self):
        from hyperflow_infinity.engine import request_contract, validate_conditioning
        plan = {'prompt': 'integrated_multimodal_description: Robinson lifts a wooden cup.', 'scene_change': True}
        for kind, config in self.configs.items():
            with self.subTest(kind=kind):
                request = request_contract(config, self.root, 1, plan, None)
                result = self.native_result(request)
                validate_conditioning(request, result)
                audit = result['conditioning_audit']
                self.assertEqual(audit['latent_keyframes'], 0)
                self.assertEqual(audit['vae_reference_frames'][0] == 1, kind == 'image')
                for key, value in [('latent_keyframes', 7), ('boundary_rows', 1), ('reference_clock_advance_removed', 7),
                                   ('vae_reference_encode_calls', 0), ('text_reference_vision_tokens', 0),
                                   ('text_reference_image_tokens', 0 if kind == 'image' else 4),
                                   ('reference_audio_rows', 1)]:
                    bad = copy.deepcopy(result)
                    bad['conditioning_audit'][key] = value
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        validate_conditioning(request, bad)


if __name__ == '__main__':
    unittest.main()
