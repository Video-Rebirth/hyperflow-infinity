"""Journaled Qwen3-VL client for the Director, Memory and Compact roles.

Every request and response is written under `calls/<chunk>_<role>/` before it is used, so a
resumed run replays saved answers instead of asking again, and refuses to guess when a request
was sent but its response never landed.
"""
import copy
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import openrouter, prompts
from .store import atomic_json, read_json, require, sha256_file

PROMPT_BUDGET = 2400
MEMORY_BUDGET = 16000
COMPACT_BUDGET = 1200
IDENTITY_BUDGET = 200
MAX_TOKENS = 8000


def parse_json(text):
    """Pull the first complete JSON object out of a model response."""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    start = text.find('{')
    if start == -1:
        raise ValueError(f'no JSON object in director output:\n{text[:600]}')
    depth, end, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f'unterminated JSON in director output:\n{text[:600]}')
    # Qwen sometimes breaks a long string with raw newlines between the H3 sections.
    return json.loads(text[start:end], strict=False)


def parse_memory_json(raw):
    """Like `parse_json`, but also accept three complete Memory arrays missing only the root brace."""
    try:
        return parse_json(raw)
    except ValueError:
        text = raw.strip()
        if not text.startswith('{') or not text.endswith(']'):
            raise
        data = json.loads(text + '}')
        if not isinstance(data, dict) or set(data) != {'add_nodes', 'update_nodes', 'add_events'}:
            raise ValueError('Incomplete Memory object')
        if not all(isinstance(value, list) for value in data.values()):
            raise ValueError('Memory values must be complete arrays')
        return data


def fix_dialogue_tags(data):
    """Map `<tool_call>` / `</tool_call>` back to `<d>` / `</d>`.

    `<d>` is three ordinary tokens while the tool-call tags are single special tokens in the
    Qwen3 vocabulary, so the model emits those where a dialogue tag belongs.
    """
    if isinstance(data, str):
        return data.replace('</tool_call>', '</d>').replace('<tool_call>', '<d>')
    if isinstance(data, list):
        return [fix_dialogue_tags(x) for x in data]
    if isinstance(data, dict):
        return {k: fix_dialogue_tags(v) for k, v in data.items()}
    return data


def normalize_director_section(data):
    """Canonicalize the two observed section-label slips.

    `integrated_multimodality_description:` at the very start of a prompt; and `\\non_diegetic_music:`
    in the JSON, where the escape swallowed the label's first letter, decoding to `on_diegetic_music:`.
    """
    prompt = data.get('prompt')
    if not isinstance(prompt, str):
        return data
    alias, canonical = 'integrated_multimodality_description:', 'integrated_multimodal_description:'
    if prompt.startswith(alias) and prompt.count(alias) == 1 and canonical not in prompt:
        prompt = canonical + prompt[len(alias):]
    if 'non_diegetic_music:' not in prompt and prompt.count('\non_diegetic_music:') == 1:
        prompt = prompt.replace('\non_diegetic_music:', '\nnon_diegetic_music:')
    if prompt == data.get('prompt'):
        return data
    data = copy.deepcopy(data)
    data['prompt'] = prompt
    return data


def beat_overlap(a, b):
    """Jaccard overlap of content words; catches a beat that only rewords an earlier one."""
    ta = set(re.findall(r'[a-z]{4,}', (a or '').lower()))
    tb = set(re.findall(r'[a-z]{4,}', (b or '').lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def validate_director(data):
    require(isinstance(data.get('prompt'), str) and data['prompt'].strip(), 'Empty generation prompt')
    require(isinstance(data.get('scene_change'), bool), 'scene_change must be boolean')
    missing = [label for label in prompts.H3_SECTIONS if label not in data['prompt']]
    require(not missing, 'Missing H3 prompt sections, use these exact labels: ' + ' '.join(missing))
    require(not any(tag in data['prompt'] for tag in ('<Picture ', '<Video ', '<Audio ')),
            'Legacy latent-only prompt must not invent RGB material tags')
    require(isinstance(data.get('beat'), str) and data['beat'].strip(), 'Director must name the new story beat')


class QwenClient:
    def __init__(self, url, root, max_attempts, guidance='', provider=None, branch=None):
        """`provider` is the run config when `vlm_provider` is `openrouter`; None means the local service at `url`.
        `branch` is the config's `branch_guidance` in a run made by `hyperflow-infinity branch`."""
        self.url = url
        self.root = Path(root)
        self.max_attempts = max_attempts
        self.guidance = guidance
        self.branch = branch
        self.provider = provider if provider and openrouter.enabled(provider) else None
        if self.provider:
            openrouter.validate_config(self.provider)

    def first_prompt(self, chunk_id, topic, seed, seconds):
        return self._call(chunk_id, 'director', prompts.DIRECTOR_SYSTEM,
                          prompts.first_prompt(topic, seed, seconds), None,
                          PROMPT_BUDGET, ('prompt', 'scene'))

    def next_prompt(self, chunk_id, video, *, topic, seed, index, seconds, prev_prompt, prev_seconds,
                    cast, beats, current_scene, scene_length, max_scene_segments, memory_text,
                    memory_note='', validate_extra=None):
        must_move = scene_length >= max_scene_segments
        recent = [b for _, b in beats[-8:]]

        def validate(data):
            if must_move:
                if not data.get('scene_change'):
                    return 'the scene budget is spent; this segment must start a new scene'
                if (data.get('scene') or '').strip().lower() == current_scene.strip().lower():
                    return (f'a new scene needs a different place: {current_scene!r} is where '
                            'we already are, so name the location the story moves to')
            beat = data.get('beat') or ''
            for earlier in recent:
                if beat_overlap(beat, earlier) > 0.5:
                    return (f'this beat just restates an earlier one ({earlier[:60]!r}); '
                            'advance the story instead')
            return validate_extra(data) if validate_extra else None

        user = prompts.next_prompt(topic, seed, index, seconds, prev_prompt, prev_seconds, cast, beats,
                                   current_scene, scene_length, max_scene_segments, memory_text, memory_note)
        return self._call(chunk_id, 'director', prompts.DIRECTOR_SYSTEM, user, video,
                          PROMPT_BUDGET, ('prompt', 'scene'), validate)

    def scene_graph_delta(self, chunk_id, video, graph, seed, index, clip_start, clip_end, event_window=40):
        """Ask only for what this segment adds; the builder sees all nodes but only recent events."""
        graph = graph or {'nodes': [], 'events': []}
        older = max(0, len(graph.get('events') or []) - event_window)
        view = {'nodes': graph.get('nodes') or [],
                'events_recent': (graph.get('events') or [])[-event_window:],
                'events_older_not_shown': older}
        known = {n.get('id') for n in view['nodes']}

        def check(data):
            for key in ('add_nodes', 'update_nodes', 'add_events'):
                if key in data and not isinstance(data[key], list):
                    return f'"{key}" must be a list'
            new_ids = set()
            for node in data.get('add_nodes') or []:
                if not isinstance(node, dict) or not node.get('id') or not node.get('label'):
                    return 'every entry in "add_nodes" needs an "id" and a "label"'
                new_ids.add(node['id'])
            for upd in data.get('update_nodes') or []:
                if not isinstance(upd, dict) or upd.get('id') not in known | new_ids:
                    return (f'"update_nodes" refers to {upd.get("id")!r}, which is not an '
                            'existing node id; add it under "add_nodes" instead')
            for ev in data.get('add_events') or []:
                if not isinstance(ev, dict) or not ev.get('action'):
                    return 'every event needs an "action"'
                if ev.get('subject') and ev['subject'] not in known | new_ids:
                    return (f'event subject {ev["subject"]!r} is not a known node id; add the '
                            'node under "add_nodes" or use an existing id')
            return None

        user = prompts.memory_delta(seed, index, clip_start, clip_end, view, event_window, older)
        return self._call(chunk_id, 'memory', prompts.MEMORY_SYSTEM, user, video, MEMORY_BUDGET,
                          ('add_nodes', 'update_nodes', 'add_events'), check)

    def compact_history(self, chunk_id, events, existing_chapters):
        return self._call(chunk_id, 'compact', prompts.COMPACT_SYSTEM,
                          prompts.compact(events, existing_chapters), None,
                          COMPACT_BUDGET, ('title', 'summary'))

    def pick_identity(self, chunk_id, frames, character):
        """Index into `frames` of the clearest view of `character`'s face, or None if none shows it."""
        def check(data):
            if not isinstance(data['best'], int) or not 1 <= data['best'] <= len(frames):
                return f'"best" must be an image number from 1 to {len(frames)}'
            return None if isinstance(data['face_visible'], bool) else '"face_visible" must be true or false'

        images = [{'path': f['path'], 'label': f'Image {i}:'} for i, f in enumerate(frames, 1)]
        data = self._call(chunk_id, 'identity', prompts.IDENTITY_SYSTEM,
                          prompts.identity_pick(character, len(frames)), None, IDENTITY_BUDGET,
                          ('best', 'face_visible'), check, images=images)
        return data['best'] - 1 if data['face_visible'] else None

    def build_request(self, role, system, user, video, attempt, budget, rejected=None, images=()):
        """`rejected` is (previous raw answer, error) on a retry."""
        text = user
        if rejected:
            raw, error = rejected
            text = (f'user:\n{user}\n\nassistant:\n{raw[:2000]}\n\nuser:\n'
                    f'That answer was rejected: {error}. Fix that problem. Reply with complete JSON and nothing else.')
        if role == 'director':
            text += prompts.director_suffix(self.guidance, self.branch)
        videos = [{'path': str(video), 'sha256': sha256_file(video), 'label': prompts.VIDEO_LABEL}] if video else []
        return {'role': role, 'system': system, 'user': text, 'images': list(images), 'videos': videos,
                'max_tokens': min(int(budget * (1 + .6 * (attempt - 1))), MAX_TOKENS),
                'temperature': max(.4, .9 - .15 * (attempt - 1))}

    def _send(self, request, request_path, error_path):
        """Journal the request, send it once, return the response.

        A request the model rejected without producing text (HTTP error, API error) is recorded
        under `error_path` and its request file removed: the model never saw it, so `launch --resume`
        may send it again. A request with no answer at all keeps its file and stops the run.
        """
        if self.provider:
            payload, audit = openrouter.prepare(self.provider, request)
            atomic_json(request_path, dict(request, transport=audit))
            response = openrouter.complete(payload)
            if response.get('transport_error'):
                atomic_json(error_path, dict(response, at=time.time()))
                request_path.unlink()
                raise RuntimeError(response['transport_error'] + '; the run stopped, resume to retry')
            return response
        atomic_json(request_path, request)
        http = urllib.request.Request(self.url + '/generate', data=json.dumps(request).encode(),
                                      headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(http, timeout=1800) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            atomic_json(error_path, {'status': e.code, 'body': e.read().decode(errors='replace')[:2000], 'at': time.time()})
            request_path.unlink()
            raise
        except urllib.error.URLError as e:
            if openrouter.never_sent(e):
                atomic_json(error_path, {'error': f'service unreachable: {e.reason}', 'at': time.time()})
                request_path.unlink()
            raise

    def _call(self, chunk_id, role, system, user, video, budget, required, validate=None, images=()):
        folder = self.root / 'calls' / f'{chunk_id}_{role}'
        folder.mkdir(parents=True, exist_ok=True)
        saved = read_json(folder / 'validated.json')
        if saved is not None:
            return saved
        rejected, last_error = None, 'no response'
        for attempt in range(1, self.max_attempts + 1):
            request = self.build_request(role, system, user, video, attempt, budget, rejected, images)
            request_path = folder / f'{attempt:02d}.request.json'
            response_path = folder / f'{attempt:02d}.response.json'
            if request_path.exists() and not response_path.exists():
                raise RuntimeError(f'Uncertain previous request at {request_path}; inspect before retrying')
            if response_path.exists():
                response = read_json(response_path)
            else:
                response = self._send(request, request_path, folder / f'{attempt:02d}.http_error.json')
                atomic_json(response_path, response)
            raw = response.get('text', '')
            try:
                data = fix_dialogue_tags(parse_memory_json(raw) if role == 'memory' else parse_json(raw))
                if role == 'director':
                    data = normalize_director_section(data)
                require(all(k in data for k in required), f'Missing keys {required}')
                if validate:
                    error = validate(data)
                    require(not error, error)
                if role == 'director':
                    validate_director(data)
                atomic_json(folder / 'validated.json', data)
                return data
            except (ValueError, json.JSONDecodeError) as e:
                last_error = str(e)
                error_path = folder / f'{attempt:02d}.error.json'
                if not error_path.exists():
                    atomic_json(error_path, {'error': last_error, 'at': time.time()})
                rejected = (raw, last_error)
        raise RuntimeError(f'{role} exhausted {self.max_attempts} requests: {last_error}')
