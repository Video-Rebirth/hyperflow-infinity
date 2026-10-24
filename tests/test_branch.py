import fcntl
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hyperflow_infinity import branch, cli
from hyperflow_infinity.engine import Engine, request_contract
from hyperflow_infinity.qwen import QwenClient
from hyperflow_infinity.store import atomic_json, read_json, sha256_file

REPO = Path(__file__).resolve().parents[1]
GUIDANCE = '改走云海，寻找空中花园'


def source_run(base):
    """A launched run with 3 committed segments. Media are placeholder bytes; only their hashes matter."""
    source = base / 'source'
    image = base / 'identity.png'
    Image.new('RGB', (32, 32)).save(image)
    cli.main(['init', '--run-dir', str(source), '--config', str(REPO / 'configs/alice.json'),
              '--h3-dir', str(base), '--hyperflow-weights', str(base / 'w.safetensors'),
              '--generation-gpus', '0,1', '--conditioning-gpu', '2', '--identity-image', str(image)])
    for name in ('run.lock', 'supervisor.lock'):
        (source / name).touch()
    chunks = []
    for index in range(1, 4):
        cid = f'C{index:04d}'
        folder = source / 'chunks' / cid
        folder.mkdir(parents=True)
        media = {}
        for key, name in (('raw_video', 'raw.mp4'), ('raw_audio', 'audio.wav'), ('latents', 'latents.pt'),
                          ('edited_video', 'clip.mp4')):
            (folder / name).write_bytes(f'{key} {cid}'.encode())
            media[key] = str(folder / name)
        frames, overlap = (345, 0) if index == 1 else (141, 22)
        plan = {'scene': 'shore', 'scene_change': index == 1, 'beat': f'action {index}',
                'prompt': f'integrated_multimodal_description: source {index}',
                'new_characters': ([{'name': 'Friend', 'appearance': 'A sailor'}] if index == 2 else
                                   [{'name': 'FUTURE_SECRET', 'appearance': 'after the fork'}] if index == 3 else [])}
        start = sum(c['kept_frames'] for c in chunks)
        chunk = dict(media, index=index, chunk_id=cid, scene='shore', scene_change=plan['scene_change'],
                     beat=plan['beat'], prompt=plan['prompt'] + ' <Picture 1> binding',
                     director_prompt=plan['prompt'], actual_frames=frames, overlap_frames=overlap,
                     kept_frames=frames - overlap, edited_video_sha256=sha256_file(media['edited_video']),
                     raw_video_sha256=sha256_file(media['raw_video']), latent_sha256=sha256_file(media['latents']),
                     film_start_frame=start, film_end_frame=start + frames - overlap)
        chunks.append(chunk)
        atomic_json(folder / 'commit.json', chunk)
        atomic_json(folder / 'director.json', plan)
        atomic_json(source / 'memory_snapshots' / f'{cid}.json', {
            'graph': {'nodes': [{'id': 'robinson', 'label': 'Robinson', 'state': f'observed {index}'}],
                      'events': [{'subject': 'robinson', 'action': f'observed {index}'}], 'chapters': []},
            'archive': [], 'observed_through': index, 'video_sha256': chunk['edited_video_sha256'],
            'delta': {'add_nodes': [], 'update_nodes': [], 'add_events': []}})
        for role in ('director', 'memory'):
            atomic_json(source / 'calls' / f'{cid}_{role}' / '01.request.json', {'chunk': cid, 'role': role})
            atomic_json(source / 'calls' / f'{cid}_{role}' / '01.response.json', {'text': 'answer'})
    atomic_json(source / 'state.json', {'chunks': chunks, 'film_frames': chunks[-1]['film_end_frame'],
                                        'cast': {'Robinson': 'x', 'Friend': 'A sailor', 'FUTURE_SECRET': 'x'}})
    snapshot = read_json(source / 'memory_snapshots/C0003.json')
    atomic_json(source / 'memory.json', {k: snapshot[k] for k in ('graph', 'archive', 'observed_through')})
    return source


def inventory(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob('*') if p.is_file()}


class Branch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = source_run(self.base)
        self.child = self.base / 'child'

    def tearDown(self):
        self.tmp.cleanup()

    def branch(self, after='C0002', **kw):
        return branch.prepare(self.source, self.child, after, GUIDANCE, **kw)

    def test_only_the_shared_past_is_inherited_and_the_source_is_untouched(self):
        before = inventory(self.source)
        record = self.branch(new_chunks=2, seed=42)
        state = read_json(self.child / 'state.json')
        self.assertEqual(state['chunks'], read_json(self.source / 'state.json')['chunks'][:2])
        self.assertEqual(set(state['cast']), {'Alice / 爱丽丝', 'Friend'})
        self.assertNotIn('FUTURE_SECRET', json.dumps(inventory(self.child)))
        self.assertFalse((self.child / 'memory_snapshots/C0003.json').exists())
        self.assertFalse((self.child / 'calls').exists())
        self.assertTrue((self.child / 'runtime/source_calls/C0002_memory/01.response.json').is_file())
        self.assertFalse((self.child / 'runtime/source_calls/C0003_director').exists())
        self.assertEqual(read_json(self.child / 'memory.json')['observed_through'], 2)
        self.assertEqual(record['stop_after_chunk'], 4)
        config = read_json(self.child / 'config.json')
        self.assertEqual((config['seed'], config['branch_guidance']), (42, {'after_chunk': 2, 'text': GUIDANCE}))
        self.assertTrue(read_json(self.child / 'control.json')['pause_after_commit'])
        previous = state['chunks'][-1]
        req = request_contract(config, self.child, 3, {'prompt': 'integrated_multimodal_description: a',
                                                        'scene_change': False}, previous)
        self.assertEqual((req['boundary_latents'], req['boundary_source_sha256'], req['kept_frames']),
                         (previous['latents'], previous['latent_sha256'], 119))
        cli.verify(self.child)
        self.assertEqual(inventory(self.source), before)

    def test_director_sees_the_guidance_as_direction(self):
        self.branch()
        config = read_json(self.child / 'config.json')
        engine = Engine(config, generator=None)
        text = engine.client.build_request('director', 's', 'u', None, 1, 100)['user']
        self.assertIn(f'<user_story_guidance>\n{GUIDANCE}\n</user_story_guidance>', text)
        self.assertIn('after C0002', text)
        for role in ('memory', 'compact'):
            self.assertNotIn(GUIDANCE, engine.client.build_request(role, 's', 'u', None, 1, 100)['user'])
        plain = QwenClient('http://unused', self.base, 3).build_request('director', 's', 'u', None, 1, 100)
        self.assertNotIn('user_story_guidance', plain['user'])

    def test_cli_reads_unicode_guidance_from_a_file(self):
        text = '不要再走旧路。\n转动齿轮，让门外成为云海。'
        path = self.base / 'direction.txt'
        path.write_text(text, encoding='utf-8')
        cli.main(['branch', '--source-run', str(self.source), '--run-dir', str(self.child),
                  '--after-segment', '2', '--guidance-file', str(path), '--new-segments', '12'])
        config = read_json(self.child / 'config.json')
        self.assertEqual((config['branch_guidance']['text'], config['stop_after_chunk']), (text, 14))

    def test_a_running_source_is_refused(self):
        with (self.source / 'supervisor.lock').open('rb') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'Source is running'):
                self.branch()
        self.assertFalse(self.child.exists())

    def test_an_unobserved_last_segment_is_observed_in_the_branch(self):
        (self.source / 'memory_snapshots/C0002.json').unlink()
        shutil.rmtree(self.source / 'calls/C0002_memory')
        self.branch()
        self.assertEqual(read_json(self.child / 'memory.json')['observed_through'], 1)

    def test_a_failed_memory_or_unknown_request_is_not_retried_in_the_branch(self):
        (self.source / 'memory_snapshots/C0002.json').unlink()
        with self.assertRaisesRegex(ValueError, 'attempted but has no snapshot'):
            self.branch()
        shutil.rmtree(self.source / 'calls/C0002_memory')
        (self.source / 'calls/C0001_director/01.response.json').unlink()
        with self.assertRaisesRegex(ValueError, 'no recorded result'):
            self.branch()
        self.assertFalse(self.child.exists())

    def test_memory_must_belong_to_the_rendered_segment(self):
        path = self.source / 'memory_snapshots/C0002.json'
        atomic_json(path, dict(read_json(path), video_sha256='other'))
        with self.assertRaisesRegex(ValueError, 'not bound'):
            self.branch()

    def test_a_trimmed_last_segment_is_refused(self):
        state = read_json(self.source / 'state.json')
        chunk = state['chunks'][1]
        chunk['kept_frames'] -= 1
        chunk['film_end_frame'] -= 1
        atomic_json(self.source / 'chunks/C0002/commit.json', chunk)
        atomic_json(self.source / 'state.json', state)
        with self.assertRaisesRegex(ValueError, 'trimmed after generation'):
            self.branch()

    def test_changed_shared_media_or_history_stops_launch(self):
        self.branch()
        latents = self.source / 'chunks/C0002/latents.pt'
        original = latents.read_bytes()
        latents.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Shared segment media changed'):
            cli.verify(self.child)
        latents.write_bytes(original)
        state = read_json(self.child / 'state.json')
        state['chunks'][0]['beat'] = 'rewritten'
        atomic_json(self.child / 'state.json', state)
        with self.assertRaisesRegex(ValueError, 'Shared segments changed'):
            branch.verify(self.child)

    def test_bad_inputs_and_existing_directories_are_refused(self):
        for after, text in (('0', 'a'), ('C9999', 'a'), ('2', ' ')):
            with self.subTest(after=after), self.assertRaises(ValueError):
                branch.prepare(self.source, self.child, after, text)
        self.assertFalse(self.child.exists())
        self.branch()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.branch()
        with self.assertRaisesRegex(ValueError, 'non-nested'):
            branch.prepare(self.source, self.source / 'nested', 2, 'a')

    def test_a_branch_of_a_branch(self):
        self.branch()
        for name in ('run.lock', 'supervisor.lock'):
            (self.child / name).touch()
        grandchild = self.base / 'grandchild'
        branch.prepare(self.child, grandchild, 'C0001', 'another way')
        self.assertEqual(len(read_json(grandchild / 'state.json')['chunks']), 1)
        cli.verify(grandchild)

    def test_new_chunks_stops_the_loop(self):
        self.branch(new_chunks=1)
        config = read_json(self.child / 'config.json')
        atomic_json(self.child / 'control.json', {'pause_after_commit': False})
        engine = Engine(config, generator=None)
        engine.observe = lambda chunk: None
        engine.publish = lambda: None
        state = read_json(self.child / 'state.json')
        state['chunks'].append(dict(state['chunks'][-1], index=3, chunk_id='C0003'))
        engine.state = state
        engine.execute()
        status = read_json(self.child / 'status.json')
        self.assertEqual((status['status'], status['stop_after_chunk']), ('completed', 3))


if __name__ == '__main__':
    unittest.main()
