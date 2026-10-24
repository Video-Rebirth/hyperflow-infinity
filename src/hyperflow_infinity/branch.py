"""`hyperflow-infinity branch`: a new run that continues a stopped run after one of its segments.

Preparation is CPU-only. The shared segments stay in the source run and are linked, not copied; the
Memory snapshot and the Qwen journal up to the fork are copied. Nothing in the source is written.
"""
import copy
import fcntl
import re
import shutil
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .store import atomic_json, digest, read_json, require, sha256_file

PACKAGE = Path(__file__).resolve().parent
OVERLAP_FRAMES = 22
EMPTY_MEMORY = {'graph': {'nodes': [], 'events': [], 'chapters': []}, 'archive': [], 'observed_through': 0}
MEDIA_KEYS = (('edited_video', 'edited_video_sha256'), ('raw_video', 'raw_video_sha256'),
              ('latents', 'latent_sha256'), ('raw_audio', None))


def chunk_index(value):
    match = re.fullmatch(r'C?([0-9]+)', str(value))
    require(match is not None and int(match[1]) > 0, 'Use a positive segment number, for example 28 or C0028')
    return int(match[1])


@contextmanager
def stopped_source(source):
    """Hold the source launcher's locks, so it cannot start while the branch is prepared."""
    with ExitStack() as stack:
        for name in ('supervisor.lock', 'run.lock'):
            path = source / name
            require(path.is_file(), f'Missing source {name}; branch from a run that has been launched')
            handle = stack.enter_context(path.open('rb'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Source is running. Pause it and wait for launch to exit before branching') from None
        yield


def shared_prefix(source, config, count):
    """The first `count` committed segments, checked against their commits and media hashes."""
    state = read_json(source / 'state.json')
    require(state and count <= len(state['chunks']), f'Segment C{count:04d} has not been committed')
    chunks = copy.deepcopy(state['chunks'][:count])
    # Not state['cast']: it may hold people introduced after the fork.
    cast = {c['name']: c['identity_text'] for c in config['seed_story']['characters']}
    assets = {}
    cursor = 0
    for index, chunk in enumerate(chunks, 1):
        cid = f'C{index:04d}'
        folder = source / 'chunks' / cid
        require(chunk['index'] == index and chunk['chunk_id'] == cid, 'Segments are not contiguous')
        require(read_json(folder / 'commit.json') == chunk, f'Commit differs from state: {cid}')
        require(chunk['film_start_frame'] == cursor and chunk['kept_frames'] > 0, f'Invalid film clock at {cid}')
        cursor += chunk['kept_frames']
        require(chunk['film_end_frame'] == cursor, f'Invalid film end at {cid}')
        require(chunk['overlap_frames'] == (0 if index == 1 else OVERLAP_FRAMES), f'Unexpected overlap at {cid}')
        for key, hash_key in MEDIA_KEYS:
            path = Path(chunk[key])
            require(path.is_absolute() and path.is_file(), f'Missing {key} for {cid}: {path}')
            actual = sha256_file(path)
            require(hash_key is None or chunk[hash_key] == actual, f'Changed {key}: {cid}')
            assets[str(path)] = actual
        plan = read_json(folder / 'director.json')
        require(plan is not None, f'Missing Director record for {cid}')
        for person in plan.get('new_characters') or []:
            if person.get('name') and person.get('appearance'):
                cast.setdefault(person['name'], person['appearance'])
    last = chunks[-1]
    # The next segment starts from the last 7 latents, which must belong to frames the film kept.
    require(last['kept_frames'] + last['overlap_frames'] == last['actual_frames'],
            f'C{count:04d} was trimmed after generation; branch after a complete segment')
    return {'chunks': chunks, 'film_frames': cursor, 'cast': cast}, assets


def memory_at(source, chunks):
    """The Memory snapshot after the last observed shared segment, and every snapshot up to it."""
    snapshots = []
    for chunk in chunks:
        path = source / 'memory_snapshots' / f"{chunk['chunk_id']}.json"
        snapshot = read_json(path)
        if snapshot is None:
            require(chunk is chunks[-1], f"Missing earlier Memory snapshot: {chunk['chunk_id']}")
            # A never-observed last segment is observed once in the branch. A failed attempt keeps its spent budget.
            for role in ('memory', 'compact'):
                folder = source / 'calls' / f"{chunk['chunk_id']}_{role}"
                require(not folder.exists() or not any(folder.iterdir()),
                        f"Memory for {chunk['chunk_id']} was attempted but has no snapshot; resolve it in the source first")
            continue
        require(snapshot['observed_through'] == chunk['index'] and
                snapshot['video_sha256'] == chunk['edited_video_sha256'],
                f"Memory snapshot is not bound to {chunk['chunk_id']}")
        snapshots.append(path)
    memory = read_json(snapshots[-1]) if snapshots else EMPTY_MEMORY
    return {key: copy.deepcopy(memory[key]) for key in EMPTY_MEMORY}, snapshots


def bound_reference(source, config, key):
    """`config[key]` with the hash that the engine bound after init, when init could not know it."""
    ref = copy.deepcopy(config[key])
    if ref.get('sha256') is None:
        bound = read_json(source / 'runtime' / f'{key}.json')
        require(bound and bound['path'] == ref['path'], f'Source {key} was never bound; resolve it in the source first')
        ref['sha256'] = bound['sha256']
    relative = Path(ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, f'Source {key} must lie inside the run')
    require(sha256_file(source / relative) == ref['sha256'], f'Source {key} changed')
    return ref


def prepare(source, root, after_chunk, guidance, *, seed=None, new_chunks=None):
    """Create a paused branch run. Never starts a model, pauses the source or reuses a directory."""
    source, root = Path(source).resolve(), Path(root).resolve()
    count = chunk_index(after_chunk)
    require(isinstance(guidance, str) and guidance.strip(), 'Story guidance must not be empty')
    require(seed is None or (type(seed) is int and seed >= 0), 'Seed must be a nonnegative integer')
    require(new_chunks is None or (type(new_chunks) is int and new_chunks > 0), '--new-segments must be positive')
    require(source != root and source not in root.parents and root not in source.parents,
            'Source and branch must be separate, non-nested directories')
    require(not root.exists(), 'Run directory already exists; choose a new path')
    with stopped_source(source):
        config = read_json(source / 'config.json')
        manifest = read_json(source / 'deployment_manifest.json')
        require(config is not None and manifest is not None, 'Not a run directory: missing config or manifest')
        for name, expected in manifest['files'].items():
            require(sha256_file(source / name) == expected, f'Source static file changed: {name}')
        state, assets = shared_prefix(source, config, count)
        memory, snapshots = memory_at(source, state['chunks'])
        history = []
        for folder in sorted((source / 'calls').glob('*')):
            match = re.fullmatch(r'C([0-9]+)_[a-z]+', folder.name)
            if not folder.is_dir() or not match or int(match[1]) > count:
                continue
            for request in folder.glob('*.request.json'):
                require(request.with_name(request.name.replace('.request.json', '.response.json')).is_file(),
                        f'Source request has no recorded result; resolve it before branching: {request}')
            history.append(folder)
        config = copy.deepcopy(config)
        references = {key: bound_reference(source, config, key) for key in ('opening_reference', 'identity_anchor')}
        config.update(references, run_id=root.name, output_dir=str(root),
                      branch_guidance={'after_chunk': count, 'text': guidance.strip()})
        config.pop('stop_after_chunk', None)
        if seed is not None:
            config['seed'] = seed
        if new_chunks is not None:
            config['stop_after_chunk'] = count + new_chunks
        record = {'format': 1, 'created_at': time.time(), 'source_run': str(source),
                  'source_manifest_sha256': sha256_file(source / 'deployment_manifest.json'),
                  'after_chunk': count, 'fork_chunk': f'C{count:04d}', 'fork_frames': state['film_frames'],
                  'prefix_digest': digest(state['chunks']), 'shared_assets': assets,
                  'guidance': config['branch_guidance'], 'stop_after_chunk': config.get('stop_after_chunk'),
                  'memory_observed_through': memory['observed_through']}
        # Created only after every check passed. A failed preparation is left in place, never reused.
        root.mkdir(parents=True)
        try:
            shutil.copytree(PACKAGE, root / 'runtime/code/hyperflow_infinity',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for chunk in state['chunks']:
                link = root / 'chunks' / chunk['chunk_id']
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to((source / 'chunks' / chunk['chunk_id']).resolve(), target_is_directory=True)
            (root / 'memory_snapshots').mkdir()
            for path in snapshots:
                shutil.copyfile(path, root / 'memory_snapshots' / path.name)
            for folder in history:
                shutil.copytree(folder, root / 'runtime/source_calls' / folder.name)
            static = []
            for ref in references.values():
                path = root / ref['path']
                if path not in static:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / ref['path'], path)
                    static.append(path)
            atomic_json(root / 'runtime/branch.json', record)
            atomic_json(root / 'config.json', config)
            atomic_json(root / 'state.json', state)
            atomic_json(root / 'manifest.json', {'chunks': state['chunks'], 'film_frames': state['film_frames']})
            atomic_json(root / 'memory.json', memory)
            atomic_json(root / 'control.json', {'pause_after_commit': True, 'reason': 'Branch prepared; not launched'})
            static += sorted((root / 'runtime/code').rglob('*.py')) + [root / 'config.json', root / 'runtime/branch.json']
            static += sorted((root / 'memory_snapshots').glob('*.json'))
            static += sorted(p for p in (root / 'runtime/source_calls').rglob('*') if p.is_file())
            static += [root / 'chunks' / c['chunk_id'] / 'commit.json' for c in state['chunks']]
            atomic_json(root / 'deployment_manifest.json', {
                'format': 1, 'files': {str(p.relative_to(root)): sha256_file(p) for p in static}})
            atomic_json(root / 'status.json', {
                'status': 'branch_prepared', 'committed_chunks': count, 'film_frames': state['film_frames'],
                'film_seconds': state['film_frames'] / 24, 'updated_at': time.time()})
            verify(root)
        except Exception as exc:
            atomic_json(root / 'runtime/branch_failed.json', {'error': str(exc), 'at': time.time()})
            raise
    return record


def verify(root, *, shared_media=True):
    """Check that the shared segments, the Memory and the direction are what `prepare` recorded."""
    root = Path(root)
    record = read_json(root / 'runtime/branch.json')
    if record is None:
        return
    require(not (root / 'runtime/branch_failed.json').exists(), 'Branch preparation failed; prepare a new one')
    config, state = read_json(root / 'config.json'), read_json(root / 'state.json')
    count = record['after_chunk']
    require(config['branch_guidance'] == record['guidance'] and
            config.get('stop_after_chunk') == record['stop_after_chunk'], 'Branch direction or limit changed')
    require(len(state['chunks']) >= count and digest(state['chunks'][:count]) == record['prefix_digest'],
            'Shared segments changed')
    require(state['film_frames'] == sum(c['kept_frames'] for c in state['chunks']), 'Branch film clock changed')
    memory = read_json(root / 'memory.json')
    observed = memory['observed_through']
    require(record['memory_observed_through'] <= observed <= len(state['chunks']), 'Invalid branch Memory position')
    if observed:
        snapshot = read_json(root / 'memory_snapshots' / f'C{observed:04d}.json')
        require(snapshot and snapshot['observed_through'] == observed and
                snapshot['video_sha256'] == state['chunks'][observed - 1]['edited_video_sha256'] and
                memory == {key: snapshot[key] for key in EMPTY_MEMORY}, 'Branch Memory differs from its snapshot')
    else:
        require(memory == EMPTY_MEMORY, 'Unobserved branch Memory is not empty')
    if shared_media:
        for path, expected in record['shared_assets'].items():
            require(Path(path).is_file() and sha256_file(path) == expected, f'Shared segment media changed: {path}')
