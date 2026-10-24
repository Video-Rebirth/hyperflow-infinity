"""`hyperflow-infinity init | launch | branch | pause | status | verify`.

`init` freezes the package source, the config and any reference files into a new run directory and
leaves it paused. `launch` verifies those files and the checkpoint hash, then runs one torchrun
child in the foreground. Nothing restarts automatically.
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import branch, openrouter, prompts
from .store import atomic_json, read_json, sha256_file

PACKAGE = Path(__file__).resolve().parent


def verify(root):
    manifest = json.loads((root / 'deployment_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f'Static file changed: {name}')
    branch.verify(root)
    return manifest


def split_gpus(generation_gpus, conditioning_gpu):
    gen = generation_gpus.split(',')
    if len(gen) < 2 or len(set(gen)) != len(gen) or str(conditioning_gpu) in gen:
        raise ValueError('Use at least two distinct generation GPUs and a separate conditioning GPU')
    return gen


def is_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def check_opening_video(path):
    from .engine import FPS, OPENING_VIDEO_FRAMES
    from .media import probe
    try:
        info = probe(path)
    except Exception:
        raise ValueError(f'Opening reference is neither an image nor a readable video: {path}') from None
    low, high = OPENING_VIDEO_FRAMES
    if info['fps'] <= 0 or not low <= round(info['frames'] * FPS / info['fps']) <= high:
        raise ValueError(f'An opening video must last {low} to {high} frames at {FPS} fps '
                         f"({low / FPS:.2f} to {high / FPS:.2f} s); got {info['frames']} frames at {info['fps']:g} fps")
    return info


def init(args):
    root = args.run_dir.resolve()
    config = json.loads(args.config.read_text())
    for key in ('vlm_provider', 'openrouter_model', 'openrouter_video_mode', 'attention_backend'):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    openrouter.validate_config(config)
    for path in (args.identity_image, args.opening_reference):
        if path and not path.is_file():
            raise ValueError(f'File not found: {path}')
    if args.identity_image and not is_image(args.identity_image):
        raise ValueError(f'Identity image is not a readable image: {args.identity_image}')
    video = None
    if args.opening_reference and not is_image(args.opening_reference):
        video = check_opening_video(args.opening_reference)
    if not (args.opening_reference or args.identity_image):
        if not args.qwen_image_dir or not (args.qwen_image_dir / 'model_index.json').is_file():
            raise ValueError('Without --opening-reference or --identity-image, the opening image is drawn by '
                             'Qwen-Image: pass --qwen-image-dir (Qwen-Image in diffusers format)')
    if root.exists():
        raise ValueError('Run directory already exists; choose a new path')
    split_gpus(args.generation_gpus, args.conditioning_gpu)
    root.mkdir(parents=True)
    (root / 'runtime').mkdir()
    shutil.copytree(PACKAGE, root / 'runtime/code/hyperflow_infinity',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (root / 'references').mkdir()
    identity = config.pop('identity')
    config.update(run_id=root.name, output_dir=str(root), h3_dir=str(args.h3_dir.resolve()),
                  hyperflow_weights=str(args.hyperflow_weights.resolve()), generation_gpus=args.generation_gpus,
                  conditioning_gpu=args.conditioning_gpu, qwen_url=args.qwen_url)
    static = sorted((root / 'runtime/code').rglob('*.py')) + [root / 'config.json']

    def freeze(source, name):
        path = root / 'references' / (name + source.suffix.lower())
        shutil.copyfile(source, path)
        static.append(path)
        return str(path.relative_to(root)), sha256_file(path)

    identity_path = freeze(args.identity_image, 'identity') if args.identity_image else None
    if args.opening_reference:
        path, sha = freeze(args.opening_reference, 'opening')
        opening = dict(kind='video' if video else 'image', source='user', path=path, sha256=sha)
        if video:
            opening.update(frames=video['frames'], fps=video['fps'], use_audio=False)
    elif identity_path:
        opening = dict(kind='image', source='identity_image', path=identity_path[0], sha256=identity_path[1])
    else:
        # Drawn on the conditioning GPU before segment 1; see Engine.make_opening_image.
        character = next(c for c in config['seed_story']['characters'] if c['id'] == identity['character_id'])
        opening = dict(kind='image', source='qwen_image', path='references/opening.png', sha256=None,
                       prompt=prompts.opening_image(config['seed_story'], config['visual_style'], character))
        config['qwen_image_dir'] = str(args.qwen_image_dir.resolve())
    config['opening_reference'] = opening
    if identity_path:
        config['identity_anchor'] = dict(identity, path=identity_path[0], source='user_image',
                                         sha256=identity_path[1], effective_from_chunk=2)
    elif opening['kind'] == 'image':
        config['identity_anchor'] = dict(identity, path=opening['path'], source='opening_reference',
                                         sha256=opening['sha256'], effective_from_chunk=2)
    else:
        # Frozen by the engine once segment 1 is committed; see Engine.bind_identity.
        config['identity_anchor'] = dict(identity, path='references/identity.png', source='segment_1_best_face',
                                         sha256=None, effective_from_chunk=2)
    atomic_json(root / 'config.json', config)
    # Only an explicit `launch --resume` clears this flag.
    atomic_json(root / 'control.json', {'pause_after_commit': True, 'reason': 'Initialized; not launched'})
    atomic_json(root / 'deployment_manifest.json', {
        'format': 1, 'files': {str(p.relative_to(root)): sha256_file(p) for p in static}})
    verify(root)
    print(json.dumps({'run_dir': str(root), 'status': 'initialized_paused'}, indent=2))


def launch(args):
    root = args.run_dir.resolve()
    supervisor_lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root / 'run.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    openrouter.validate_config(config, credentials=True)
    if sha256_file(Path(config['hyperflow_weights'])) != config['hyperflow_weights_sha256']:
        raise ValueError('HyperFlow checkpoint SHA256 mismatch')
    if config.get('qwen_image_dir') and not (Path(config['qwen_image_dir']) / 'model_index.json').is_file():
        raise ValueError(f"Qwen-Image not found at {config['qwen_image_dir']}")
    for request in (root / 'calls').glob('*/*.request.json'):
        if not request.with_name(request.name.replace('.request.json', '.response.json')).exists():
            raise ValueError(f'Uncertain model request; inspect before resuming: {request}')
    for started in (root / 'chunks').glob('*/generation_started.json'):
        if not (started.parent / 'generation_result.json').exists():
            raise ValueError(f'Uncertain generation; inspect before resuming: {started}')
    gen = split_gpus(config['generation_gpus'], config['conditioning_gpu'])
    control = read_json(root / 'control.json')
    if control.get('pause_after_commit') and not args.resume:
        raise ValueError('Run is paused. An explicit --resume is required.')
    if args.resume:
        control.update(pause_after_commit=False, reason='Explicit launch --resume', resumed_at=time.time())
        atomic_json(root / 'control.json', control)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gen + [str(config['conditioning_gpu'])]),
               PYTHONPATH=os.pathsep.join(filter(None, [str(root / 'runtime/code'), os.environ.get('PYTHONPATH')])),
               PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    # Rendezvous on loopback: `--standalone` advertises the hostname, which need not resolve.
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--nnodes=1', '--master_addr=127.0.0.1',
               f'--master_port={port}', f'--nproc_per_node={len(gen)}', '--module', 'hyperflow_infinity.worker',
               '--config', str(root / 'config.json')]

    def stop(*_):
        c = read_json(root / 'control.json')
        c.update(pause_after_commit=True, reason='Supervisor received stop signal', requested_at=time.time())
        atomic_json(root / 'control.json', c)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (root / 'runtime/generation.log').open('ab', buffering=0) as log:
        # A separate session: Ctrl-C asks for a pause at the next commit instead of killing every rank.
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(root / 'runtime/launches' / f'{uuid.uuid4()}.json',
                    {'pid': child.pid, 'command': command, 'started_at': time.time()})
        while child.poll() is None:
            atomic_json(root / 'supervisor.json', {'status': 'running', 'pid': os.getpid(), 'child_pid': child.pid,
                                                   'updated_at': time.time()})
            time.sleep(5)
        atomic_json(root / 'supervisor.json', {'status': 'exited', 'pid': os.getpid(), 'child_pid': child.pid,
                                               'exit_code': child.returncode, 'updated_at': time.time()})
        return child.returncode


def branch_command(args):
    text = args.guidance_file.read_text(encoding='utf-8') if args.guidance_file else args.guidance
    record = branch.prepare(args.source_run, args.run_dir, args.after_segment, text,
                            seed=args.seed, new_chunks=args.new_segments)
    print(json.dumps({'run_dir': str(args.run_dir.resolve()), 'status': 'branch_prepared_paused',
                      'after_segment': record['fork_chunk'], 'film_seconds': record['fork_frames'] / 24,
                      'stop_after_segment': record['stop_after_chunk']}, indent=2))


def pause(args):
    root = args.run_dir.resolve()
    control = read_json(root / 'control.json')
    control.update(pause_after_commit=True, reason='Explicit operator pause', requested_at=time.time())
    atomic_json(root / 'control.json', control)
    print('Pause requested; the run stops after the current segment is committed.')


def status(args):
    root = args.run_dir.resolve()
    for name in ('status.json', 'supervisor.json', 'control.json'):
        path = root / name
        print(name, path.read_text() if path.exists() else '(not created)')


def verify_command(args):
    verify(args.run_dir.resolve())
    print('Static files verified.')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hyperflow-infinity', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init', help='create a paused run directory')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--identity-image', type=Path,
                   help='face reference for segments 2+, and the opening image when there is no --opening-reference. '
                        'Default: the opening image; after a video opening, the frame of segment 1 that Qwen '
                        "judges shows the protagonist's face best")
    p.add_argument('--opening-reference', type=Path,
                   help='an image, or a 1-14 s video (picture only, its sound is ignored), that segment 1 is '
                        'generated from. Default: --identity-image, else an image drawn by Qwen-Image')
    p.add_argument('--qwen-image-dir', type=Path,
                   help='Qwen-Image in diffusers format; needed when neither image above is given')
    p.add_argument('--h3-dir', type=Path, required=True, help='MiniMax-H3 in diffusers format')
    p.add_argument('--hyperflow-weights', type=Path, required=True,
                   help='the HyperFlow .safetensors file; its SHA256 is checked against the config at launch')
    p.add_argument('--generation-gpus', default='0,1,2,3')
    p.add_argument('--conditioning-gpu', default='4')
    p.add_argument('--attention-backend', choices=['sdpa', 'fa3'],
                   help='dense attention on the H3 DiT: sdpa (PyTorch SDPA, default) or fa3 (FlashAttention-3)')
    p.add_argument('--qwen-url', default='http://127.0.0.1:30050')
    p.add_argument('--vlm-provider', choices=['local', 'openrouter'],
                   help='local: the hyperflow-infinity-qwen service at --qwen-url (default). '
                        'openrouter: the OpenRouter API, key in OPENROUTER_API_KEY at launch')
    p.add_argument('--openrouter-model', choices=sorted(openrouter.MODEL_VIDEO_MODES))
    p.add_argument('--openrouter-video-mode', choices=['auto', 'native', 'frames'],
                   help='how the previous segment reaches the model; auto picks per model')
    p.set_defaults(func=init)
    p = sub.add_parser('launch', help='run in the foreground until paused or failed')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='clear the pause flag after all checks pass')
    p.set_defaults(func=launch)
    p = sub.add_parser('branch', help='create a paused run that continues a stopped run after one of its segments')
    p.add_argument('--source-run', type=Path, required=True, help='a run that is not running')
    p.add_argument('--after-segment', required=True, help='the last shared segment: 28 or C0028')
    p.add_argument('--run-dir', type=Path, required=True, help='a new directory for the branch')
    guidance = p.add_mutually_exclusive_group(required=True)
    guidance.add_argument('--guidance', help='where the story should go from here; sent to every Director request')
    guidance.add_argument('--guidance-file', type=Path, help='the same, read from a UTF-8 text file')
    p.add_argument('--seed', type=int, help='a new H3 seed. Default: the source seed')
    p.add_argument('--new-segments', type=int, help='stop after this many new segments. Default: run until paused')
    p.set_defaults(func=branch_command)
    for name, func, text in (('pause', pause, 'stop after the current segment'),
                             ('status', status, 'print status, supervisor and control files'),
                             ('verify', verify_command, 'check frozen code, config and identity image hashes')):
        p = sub.add_parser(name, help=text)
        p.add_argument('--run-dir', type=Path, required=True)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
