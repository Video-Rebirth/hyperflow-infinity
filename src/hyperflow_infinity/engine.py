"""The open-ended loop: observe the last segment, plan the next one, generate it, commit it.

Every artifact is written atomically and checked by SHA256, so a run can be paused at any segment
boundary and resumed without re-sending a request whose outcome is unknown.
"""
import copy
import html
import json
import shutil
import time
from pathlib import Path

from . import handoff, prompts
from .media import assemble, edit_chunk, evidence_frames
from .memory import (
    compact_scene_graph,
    graph_as_text,
    merge_scene_graph,
    repeated_observations,
    scene_length_of,
    seconds_to_mmss,
)
from .qwen import QwenClient
from .store import atomic_json, digest, read_json, require, sha256_file

FPS = 24
OPENING_FRAMES = 345       # 14.375 s; the longest 17n+5 clip within H3's 15 s limit
CONTINUATION_FRAMES = 141  # 5.875 s generated per continuation
OVERLAP_FRAMES = 22        # redraw of the previous ending, trimmed; 119 new frames remain
IDENTITY_CANDIDATES = 12   # evenly spaced frames of segment 1 offered to Qwen as face references
OPENING_VIDEO_FRAMES = (22, 345)  # an opening video must fit H3's clip range once resampled to 24 fps
# Qwen-Image's 16:9 canvas and its recommended sampler settings. H3 shrinks the still to its 480p canvas.
OPENING_IMAGE = {'width': 1664, 'height': 928, 'steps': 50, 'true_cfg_scale': 4.0,
                 'negative_prompt': 'text, letters, caption, subtitles, watermark, logo, blurry face, deformed hands'}


def request_contract(config, root, index, plan, previous):
    """The exact H3 request for segment `index`. Saved before generation and compared on resume."""
    first = previous is None
    require(first == (index == 1), 'Missing previous committed segment')
    require(not first or plan['scene_change'] is True, 'Opening must start a new scene')
    n = OPENING_FRAMES if first else CONTINUATION_FRAMES
    overlap = 0 if first else OVERLAP_FRAMES
    request = {'chunk_id': f'C{index:04d}', 'chunk_dir': str(root / 'chunks' / f'C{index:04d}'),
               'mode': 'ref2va', 'director_prompt': plan['prompt'],
               'generated_frames': n, 'overlap_frames': overlap,
               'kept_frames': n - overlap, 'height': config['height'], 'width': config['width'],
               'seed': config['seed'] + index - 1,
               'boundary_latents': None if first else previous['latents'],
               'boundary_source_sha256': None if first else sha256_file(previous['latents']),
               'director_scene_change': plan['scene_change']}
    section = prompts.H3_SECTIONS[0]
    if first:
        ref = config['opening_reference']
        path = root / ref['path']
        require(ref['sha256'] is not None and sha256_file(path) == ref['sha256'], 'Opening reference changed')
        tag = '<Picture 1>' if ref['kind'] == 'image' else '<Video 1>'
        media = {'id': 'opening_reference', 'kind': ref['kind'], 'path': str(path), 'sha256': ref['sha256']}
        if ref['kind'] == 'video':
            media['frames'] = ref['frames']
        request.update(
            generation_contract=handoff.OPENING_CONTRACT, media=[media],
            opening_reference={'kind': ref['kind'], 'sha256': ref['sha256']},
            prompt=plan['prompt'].replace(section, f'{section} {prompts.opening_binding(tag)} ', 1))
        return request
    anchor = config['identity_anchor']
    path = root / anchor['path']
    require(sha256_file(path) == anchor['sha256'], 'Fixed identity image changed')
    request.update(
        generation_contract=handoff.CONTRACT,
        media=[{'id': f"{anchor['character_id']}_identity", 'kind': 'image', 'path': str(path),
                'sha256': anchor['sha256'], 'character_id': anchor['character_id']}],
        identity_anchor=copy.deepcopy(anchor),
        prompt=plan['prompt'].replace(section, f"{section} {anchor['binding']} ", 1))
    return request


def validate_opening(request, result):
    require(request['chunk_id'] == 'C0001' and request['overlap_frames'] == 0 and not request['boundary_latents'],
            'Only segment 1 opens from a reference, with no saved latents')
    m, refs, a = request['media'], result['reference_inputs'], result['conditioning_audit']
    kind = request['opening_reference']['kind']
    require(len(m) == len(refs) == 1 and m[0]['kind'] == refs[0]['kind'] == kind, 'Exactly one opening reference required')
    require(m[0]['sha256'] == refs[0]['sha256'] == request['opening_reference']['sha256'] == sha256_file(m[0]['path']),
            'Opening reference hash differs')
    require(refs[0]['path'] == m[0]['path'], 'Opening reference path differs')
    require(a and a['contract'] == handoff.OPENING_CONTRACT and a['reference_sha256'] == m[0]['sha256'],
            'Opening audit absent or for another reference')
    require(a['latent_keyframes'] == 0 and a['boundary_rows'] == 0 and a['reference_clock_advance_removed'] == 0,
            'Opening received latent guides')
    require(a['vae_reference_encode_calls'] == 1 and a['all_condition_video_rows'] > 0 and a['reference_audio_rows'] == 0,
            'Opening reference did not reach the VAE as one silent reference')
    require(a['text_reference_vision_tokens'] == a['text_reference_image_tokens'] + a['text_reference_video_tokens'] > 0,
            'Opening reference did not reach the text vision encoder')
    require(a['target_audio_origin'] == a['target_video_origin'], 'Opening audio and video clocks differ')
    if kind == 'image':
        require(a['rgb_reference_count'] == 1 and a['reference_video_count'] == 0 and
                a['vae_reference_frames'] == [1] and a['reference_latent_frames'] == [1], 'Opening image was not encoded as one still')
        require(a['text_reference_image_tokens'] > 0 and a['text_reference_video_tokens'] == 0,
                'Opening image did not reach the text vision encoder')
        require(a['target_video_origin'] == a['text_rows'] + 1, 'Opening image position origin differs')
    else:
        require(a['rgb_reference_count'] == 0 and a['reference_video_count'] == 1 and
                a['vae_reference_frames'][0] > 1 and a['reference_latent_frames'][0] > 1,
                'Opening video was not encoded as a temporal reference')
        require(a['text_reference_video_tokens'] > 0 and a['text_reference_image_tokens'] == 0,
                'Opening video did not reach the text vision encoder')
        require(a['target_video_origin'] > a['text_rows'], 'Opening video position origin differs')


def validate_conditioning(request, result):
    """Reject a generation whose audit does not match the request, before it is committed."""
    require(result['actual_frames'] == request['generated_frames'], 'Raw frame count mismatch')
    if request['generation_contract'] == handoff.OPENING_CONTRACT:
        return validate_opening(request, result)
    require(request['generation_contract'] == handoff.CONTRACT, 'Unknown generation contract')
    m, refs, anchor = request['media'], result['reference_inputs'], request['identity_anchor']
    require(len(m) == len(refs) == 1 and m[0]['kind'] == refs[0]['kind'] == 'image', 'Exactly one fixed identity image required')
    require(m[0]['sha256'] == refs[0]['sha256'] == anchor['sha256'] == sha256_file(m[0]['path']), 'Identity reference hash differs')
    require(refs[0]['path'] == m[0]['path'], 'Identity reference path differs')
    a = result['conditioning_audit']
    require(a and a['latent_keyframes'] == handoff.TAIL_LATENTS and a['target_position_alignment'], 'Latent guide not applied')
    require(a['contract'] == handoff.CONTRACT and a['rgb_reference_count'] == 1, 'Identity RGB reference absent')
    require(a['vae_reference_encode_calls'] == 1 and a['text_reference_vision_tokens'] > 0,
            'Identity did not reach VAE and text vision encoder')
    require(a['reference_video_count'] == 0 and a['reference_audio_rows'] == 0, 'Unexpected historical AV reference')
    require(a['all_condition_video_rows'] > a['boundary_rows'], 'Identity latent rows absent')
    require(a['target_video_origin'] == a['text_rows'] + 1, 'Identity/tail position origin mismatch')
    require(request['boundary_source_sha256'] == sha256_file(request['boundary_latents']), 'Source latent changed')


def validate_observed_progress(plan, previous, stagnation):
    # `prompt` is what H3 received, with the renderer's binding inserted; the Director wrote `director_prompt`.
    if plan.get('prompt') == previous['director_prompt']:
        return 'Prompt repeats the preceding segment verbatim'
    if stagnation and (plan.get('scene_change') is not True or
                       (plan.get('scene') or '').strip().lower() == previous['scene'].strip().lower()):
        return ('Three consecutive observed segments repeated the same state/actions. Set scene_change=true and '
                'choose a different scene with a completed visible consequence, not another reaction or restart.')
    return None


class Engine:
    def __init__(self, config, generator):
        self.config = config
        self.generator = generator
        self.root = Path(config['output_dir']).resolve()
        self.client = QwenClient(config['qwen_url'], self.root, config['max_role_attempts'],
                                 config.get('director_guidance', ''), provider=config,
                                 branch=config.get('branch_guidance'))
        seed = config['seed_story']
        self.topic = seed['inspiration']
        self.seed_context = dict(seed, visual_style=config['visual_style'])
        self.state = read_json(self.root / 'state.json', {
            'chunks': [], 'film_frames': 0, 'cast': {c['name']: c['identity_text'] for c in seed['characters']}})
        self.memory = read_json(self.root / 'memory.json', {
            'graph': {'nodes': [], 'events': [], 'chapters': []}, 'archive': [], 'observed_through': 0})
        for key in ('opening_reference', 'identity_anchor'):
            bound = read_json(self.root / 'runtime' / f'{key}.json')
            if bound and self.config[key].get('sha256') is None:
                require(bound['path'] == self.config[key]['path'], f'Bound {key} path differs')
                self.config[key]['sha256'] = bound['sha256']

    def make_opening_image(self):
        """Draw the opening still with Qwen-Image once, when the run was started without an image."""
        ref = self.config['opening_reference']
        if ref['sha256'] is not None:
            return
        require(ref['source'] == 'qwen_image', 'Opening reference has no hash')
        self.status('opening_image')
        path = self.root / ref['path']
        spec = dict(OPENING_IMAGE, model_dir=self.config['qwen_image_dir'], prompt=ref['prompt'], seed=self.config['seed'])
        timing = self.generator.make_image(spec, path)
        ref['sha256'] = sha256_file(path)
        record = {'path': ref['path'], 'sha256': ref['sha256'], 'spec': spec, 'timing': timing}
        anchor = self.config['identity_anchor']
        if anchor['source'] == 'opening_reference':
            anchor['sha256'] = ref['sha256']
            atomic_json(self.root / 'runtime/identity_anchor.json', record)
        atomic_json(self.root / 'runtime/opening_reference.json', record)
        print(json.dumps({'OPENING_IMAGE': ref['path'], **timing}), flush=True)

    def bind_identity(self, opening):
        """Freeze the frame of segment 1 that best shows the protagonist's face, unless one was supplied."""
        anchor = self.config['identity_anchor']
        if anchor.get('sha256') is not None:
            return
        frames = evidence_frames(opening['raw_video'], self.root / 'chunks' / opening['chunk_id'] / 'identity_candidates',
                                 opening['chunk_id'], opening['overlap_frames'], opening['kept_frames'],
                                 fractions=[i / (IDENTITY_CANDIDATES - 1) for i in range(IDENTITY_CANDIDATES)])
        character = next(c for c in self.config['seed_story']['characters'] if c['id'] == anchor['character_id'])
        best = self.client.pick_identity(opening['chunk_id'], frames, character)
        require(best is not None, f"No frame of segment 1 shows {character['name']}'s face clearly; "
                                  'start a new run with --identity-image')
        path = self.root / anchor['path']
        shutil.copyfile(frames[best]['path'], path)
        anchor['sha256'] = sha256_file(path)
        atomic_json(self.root / 'runtime/identity_anchor.json',
                    {'path': anchor['path'], 'sha256': anchor['sha256'], 'source': frames[best],
                     'candidates': len(frames)})

    def status(self, stage, **kw):
        atomic_json(self.root / 'status.json', {
            'status': stage, 'committed_chunks': len(self.state['chunks']),
            'film_frames': self.state['film_frames'], 'film_seconds': self.state['film_frames'] / FPS,
            'updated_at': time.time(), **kw})

    def observe(self, previous):
        """Memory watches the previous segment; its delta is merged, old events folded into chapters."""
        index = previous['index']
        if self.memory['observed_through'] >= index:
            return
        cid = previous['chunk_id']
        self.status('memory', chunk_id=cid)
        delta = self.client.scene_graph_delta(
            cid, previous['edited_video'], copy.deepcopy(self.memory['graph']), self.seed_context, index,
            seconds_to_mmss(previous['film_start_frame'] / FPS), seconds_to_mmss(previous['film_end_frame'] / FPS))
        graph = merge_scene_graph(copy.deepcopy(self.memory['graph']), delta)
        archive = copy.deepcopy(self.memory['archive'])
        if len(graph['events']) > self.config['memory_compact_trigger']:
            folded = copy.deepcopy(graph['events'][:self.config['memory_fold_count']])
            chapter = self.client.compact_history(cid, folded, graph['chapters'])
            archive.extend(compact_scene_graph(graph, chapter, folded))
        self.memory = {'graph': graph, 'archive': archive, 'observed_through': index}
        atomic_json(self.root / 'memory_snapshots' / f'{cid}.json',
                    dict(self.memory, delta=delta, video_sha256=previous['edited_video_sha256']))
        atomic_json(self.root / 'memory.json', self.memory)

    def plan(self, index, previous):
        cid = f'C{index:04d}'
        self.status('director', chunk_id=cid)
        if previous is None:
            return self.client.first_prompt(cid, self.topic, self.seed_context, OPENING_FRAMES / FPS)
        chunks = self.state['chunks']
        stagnation = repeated_observations(self.root, chunks)
        if stagnation:
            atomic_json(self.root / 'runtime/observed_stagnation' / f'{cid}.json', stagnation)
        return self.client.next_prompt(
            cid, previous['edited_video'], topic=self.topic, seed=self.seed_context, index=index,
            seconds=CONTINUATION_FRAMES / FPS, prev_prompt=previous['director_prompt'],
            prev_seconds=previous['kept_frames'] / FPS, cast=self.state['cast'],
            beats=[[c['index'], c['beat'][:110]] for c in chunks[-self.config['beat_history']:]],
            current_scene=previous['scene'], scene_length=scene_length_of(chunks),
            max_scene_segments=self.config['max_scene_segments'],
            memory_text=graph_as_text(self.memory['graph'], self.config['memory_recent_events']),
            memory_note=prompts.stagnation_note(stagnation) if stagnation else '',
            validate_extra=lambda p: validate_observed_progress(p, previous, stagnation))

    def publish(self):
        """Assemble every committed segment into one MP4 and refresh `index.html`."""
        chunks = self.state['chunks']
        cid = chunks[-1]['chunk_id']
        out = self.root / 'previews' / cid
        self.status('preview', chunk_id=cid)
        if not (out / 'validation.json').exists():
            out.mkdir(parents=True, exist_ok=True)
            film, details = assemble(out, chunks, self.state['film_frames'])
            atomic_json(out / 'validation.json', {
                'status': 'verified_preview', 'at': time.time(), 'chunks': len(chunks),
                'film_frames': self.state['film_frames'], 'probe': details, 'sha256': sha256_file(film)})
        validation = read_json(out / 'validation.json')
        rel = f'previews/{cid}/film.mp4'
        title = html.escape(self.config['run_id'])
        anchor = self.config['identity_anchor']
        anchor_from = anchor['effective_from_chunk']
        source = {'segment_1_best_face': 'the frame of segment 1 that best shows the face',
                  'opening_reference': 'the opening image'}.get(anchor['source'], 'supplied at init')
        ref = self.config['opening_reference']
        opening = {'qwen_image': 'an image drawn by Qwen-Image', 'identity_image': 'the identity image',
                   'user': f"a supplied {ref['kind']}"}[ref['source']]
        (self.root / 'index.html').write_text(
            f'''<!doctype html><meta charset="utf-8"><title>{title} · HyperFlow Infinity</title>
<style>body{{background:#121820;color:#eee;font:17px system-ui;max-width:1100px;margin:35px auto}}a{{color:#8ed4ff}}video{{width:100%}}</style>
<h1>{title}</h1><p>{len(chunks)} segments · {self.state['film_frames'] / FPS:.3f} seconds · {FPS} fps</p>
<video controls src="{rel}"></video><p><a href="{rel}">MP4</a> · <a href="memory.json">Scene Graph</a> · <a href="state.json">Segment records</a> · <a href="status.json">Live status</a></p>
<p>Seed-only and open-ended: no outline, no ending. Segment 1 is generated from its prompt and {opening}. From segment {anchor_from} on, H3 also receives the previous segment's last seven latents and one fixed identity image ({source}).</p>''')
        atomic_json(self.root / 'latest_preview.json', dict(validation, url=rel))

    def execute(self):
        self.make_opening_image()
        if self.state['chunks']:
            self.bind_identity(self.state['chunks'][0])
        limit = self.config.get('stop_after_chunk')
        while True:
            if read_json(self.root / 'control.json', {}).get('pause_after_commit'):
                self.status('paused')
                return
            if limit is not None and len(self.state['chunks']) >= limit:
                self.observe(self.state['chunks'][-1])
                self.publish()
                self.status('completed', stop_after_chunk=limit)
                return
            previous = self.state['chunks'][-1] if self.state['chunks'] else None
            index = len(self.state['chunks']) + 1
            cid = f'C{index:04d}'
            folder = self.root / 'chunks' / cid
            folder.mkdir(parents=True, exist_ok=True)
            timing = {}
            started = time.time()
            if previous:
                self.observe(previous)
            timing['memory_seconds'] = round(time.time() - started, 3)
            started = time.time()
            plan = self.plan(index, previous)
            timing['director_seconds'] = round(time.time() - started, 3)
            atomic_json(folder / 'director.json', plan)
            request = request_contract(self.config, self.root, index, plan, previous)
            saved = read_json(folder / 'request.json')
            require(saved is None or saved == request, 'Existing generation request differs')
            atomic_json(folder / 'request.json', request)
            started = time.time()
            existing = read_json(folder / 'generation_result.json')
            if existing:
                require(existing['request_sha256'] == digest(request), 'Saved result request mismatch')
                result = existing['result']
            else:
                require(not (folder / 'generation_started.json').exists(),
                        'Generation already submitted without complete receipt; inspect preserved media before recovery')
                atomic_json(folder / 'generation_started.json', {'at': time.time(), 'request_sha256': digest(request)})
                self.status('generate', chunk_id=cid)
                result = self.generator.generate(request)
            timing['generation_seconds'] = round(time.time() - started, 3)
            validate_conditioning(request, result)
            started = time.time()
            edited = folder / 'clip.mp4'
            edit_chunk(result['raw_video'], edited, request['overlap_frames'], request['kept_frames'])
            evidence = evidence_frames(result['raw_video'], folder / 'evidence', cid,
                                       request['overlap_frames'], request['kept_frames'])
            timing['edit_seconds'] = round(time.time() - started, 3)
            chunk = dict(result, index=index, chunk_id=cid, scene=plan['scene'], scene_change=plan['scene_change'],
                         beat=plan['beat'], prompt=request['prompt'], director_prompt=plan['prompt'],
                         overlap_frames=request['overlap_frames'], kept_frames=request['kept_frames'],
                         edited_video=str(edited), edited_video_sha256=sha256_file(edited),
                         latent_sha256=sha256_file(result['latents']), raw_video_sha256=sha256_file(result['raw_video']),
                         film_start_frame=self.state['film_frames'],
                         film_end_frame=self.state['film_frames'] + request['kept_frames'],
                         evidence=evidence, timing=dict(result.get('timing') or {}, **timing),
                         committed_at=time.time())
            atomic_json(folder / 'commit.json', chunk)
            self.state['chunks'].append(chunk)
            self.state['film_frames'] = chunk['film_end_frame']
            for c in plan.get('new_characters') or []:
                if c.get('name') and c.get('appearance'):
                    self.state['cast'].setdefault(c['name'], c['appearance'])
            atomic_json(self.root / 'state.json', self.state)
            atomic_json(self.root / 'manifest.json', {'chunks': self.state['chunks'], 'film_frames': self.state['film_frames']})
            if index == 1:
                self.bind_identity(chunk)
            self.status('committed', chunk_id=cid)
            print(json.dumps({'COMMITTED': cid, 'frames': self.state['film_frames'], 'beat': plan['beat']}), flush=True)
            if index == 1 or index % self.config['preview_every_chunks'] == 0:
                self.publish()
