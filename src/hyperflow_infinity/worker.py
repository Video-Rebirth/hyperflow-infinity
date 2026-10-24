"""torchrun entry point. Rank 0 runs the loop; the other ranks only join H3 generation."""
import argparse
import fcntl
import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path

from .store import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['output_dir']).resolve()
    rank = int(os.environ.get('RANK', '0'))
    generator, lock, done = None, None, threading.Event()
    if rank == 0:
        lock = (root / 'run.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def request_stop(*_):
            control = read_json(root / 'control.json', {})
            control.update(pause_after_commit=True, requested_at=time.time())
            atomic_json(root / 'control.json', control)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def heartbeat():
            while not done.is_set():
                atomic_json(root / 'heartbeat.json', {'pid': os.getpid(), 'at': time.time(),
                                                      'status': read_json(root / 'status.json', {})})
                done.wait(15)
        threading.Thread(target=heartbeat, daemon=True).start()
    try:
        from .generator import HyperFlow
        generator = HyperFlow(config)
        if rank == 0:
            from .engine import Engine
            Engine(config, generator).execute()
            generator.stop()
        else:
            generator.worker_loop()
    except Exception as exc:
        if rank == 0:
            failure = {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(), 'at': time.time()}
            atomic_json(root / 'failure.json', failure)
            atomic_json(root / 'status.json', dict(read_json(root / 'status.json', {}), status='needs_attention',
                                                   error=failure['error'], updated_at=time.time()))
        if generator and generator.in_generation and generator.world > 1:
            # A rank that dies mid-collective would leave the others blocked forever.
            traceback.print_exc()
            os._exit(1)
        if rank == 0 and generator:
            generator.stop()
        raise
    finally:
        done.set()
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
