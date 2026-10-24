"""HTTP service around H3's own Qwen3-VL text encoder, read through its LM head.

`POST /generate` takes `{system, user, videos: [{path, sha256}], max_tokens, temperature}` and
returns `{text, seconds, input_tokens, output_tokens, input_audit}`. Videos are passed to Qwen's
own video processor as files, so sampling, resizing and timestamps are the processor's defaults
under both `--backend hf` (Hugging Face `generate`) and `--backend sglang` (SGLang `Engine` fed
the exact `processor_output` tensors). One request runs at a time.
"""
import argparse
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .store import sha256_file


def build_messages(req):
    from PIL import Image
    content, video_hashes = [], []
    for item in req.get('images', []):
        im = Image.open(item['path']).convert('RGB')
        im.thumbnail((560, 560))
        content += [{'type': 'text', 'text': item.get('label', '')}, {'type': 'image', 'image': im}]
    for item in req.get('videos', []):
        path = item.get('path')
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError('Native video input requires one existing video file, never a frame list')
        h = sha256_file(path)
        if item.get('sha256') and item['sha256'] != h:
            raise ValueError('Native video source SHA differs from the request')
        video_hashes.append(h)
        content += [{'type': 'text', 'text': item.get('label', 'Previous generated video')},
                    {'type': 'video', 'video': path}]
    content.append({'type': 'text', 'text': req['user']})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': req['system']}]},
            {'role': 'user', 'content': content}], video_hashes


def prepare_inputs(processor, req):
    messages, hashes = build_messages(req)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors='pt')
    grid = inputs.get('video_grid_thw')
    if hashes and (grid is None or len(grid) != len(hashes) or 'pixel_values_videos' not in inputs):
        raise ValueError('Qwen processor did not produce the expected native video tensors')
    audit = {'native_video_count': len(hashes), 'video_sha256': hashes,
             'video_grid_thw': grid.tolist() if grid is not None else [],
             'input_shapes': {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')},
             'processor_default_fps': getattr(processor.video_processor, 'fps', None) if hashes else None}
    return inputs, audit


def _install_sglang_shims():
    """Let SGLang and its spawned scheduler/detokenizer workers run with transformers 5.x and without flashinfer."""
    import tempfile
    os.environ.setdefault('SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK', '1')
    try:
        from transformers import AutoConfig
        orig_register = AutoConfig.register

        def register(model_type, config, exist_ok=False):
            return orig_register(model_type, config, exist_ok=True)
        AutoConfig.register = staticmethod(register)
    except Exception:
        pass
    try:
        import flashinfer  # noqa: F401
        has_flashinfer = True
    except ImportError:
        has_flashinfer = False
        os.environ.setdefault('SGLANG_IS_FLASHINFER_AVAILABLE', 'false')

    def stub_module(name):
        module = types.ModuleType(name)
        module.__path__ = []

        def getattr_(attr):
            if attr.startswith('__'):
                raise AttributeError(attr)

            def missing(*_, **__):
                raise RuntimeError(f'flashinfer stub: {name}.{attr} was called')

            class Meta(type):
                def __getattr__(cls, sub):
                    if sub.startswith('__'):
                        raise AttributeError(sub)
                    return lambda *_, **__: None
            return Meta(attr, (), {'__init__': missing, '__init_subclass__': classmethod(lambda cls, **_: None)})
        module.__getattr__ = getattr_
        return module

    class SglKernelFlagFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in ('sgl_kernel.elementwise', 'sgl_kernel.sampling'):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original = spec.loader.exec_module

            def exec_module(module):
                original(module)
                module._has_flashinfer = False
            spec.loader.exec_module = exec_module
            return spec

    class FlashinferStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'flashinfer' or fullname.startswith('flashinfer.'):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return stub_module(spec.name)

        def exec_module(self, module):
            pass

    if not has_flashinfer:
        sys.meta_path.insert(0, FlashinferStubFinder())
        sys.meta_path.insert(0, SglKernelFlagFinder())

    shim_dir = Path(tempfile.gettempdir()) / 'hyperflow_infinity_sglang_shim'
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / 'sitecustomize.py').write_text(
        f'import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); '
        'from hyperflow_infinity.qwen_service import _install_sglang_shims; _install_sglang_shims()\n')
    pypath = os.environ.get('PYTHONPATH', '')
    if str(shim_dir) not in pypath.split(os.pathsep):
        os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [str(shim_dir), pypath]))


def sglang_mm_kwargs(inputs):
    """Wrap Qwen processor tensors as SGLang `processor_output` dicts so SGLang skips its own resampler."""
    kw = {}
    if 'pixel_values_videos' in inputs:
        kw['video_data'] = [{'format': 'processor_output', 'input_ids': inputs['input_ids'],
                             'pixel_values_videos': inputs['pixel_values_videos'],
                             'video_grid_thw': inputs['video_grid_thw']}]
    if 'pixel_values' in inputs:
        kw['image_data'] = [{'format': 'processor_output', 'input_ids': inputs['input_ids'],
                             'pixel_values': inputs['pixel_values'],
                             'image_grid_thw': inputs['image_grid_thw']}]
    return kw


def load_backend(backend, model_dir, tp_size=1, mem_fraction_static=0.85):
    import torch
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    if backend == 'sglang':
        _install_sglang_shims()
        from sglang.srt.entrypoints.engine import Engine
        engine = Engine(model_path=model_dir, dtype='bfloat16', tp_size=tp_size,
                        mem_fraction_static=mem_fraction_static, context_length=32768,
                        attention_backend='fa3', sampling_backend='pytorch',
                        grammar_backend='none', log_level='warning')

        def generate(inputs, max_tokens, temperature):
            out = engine.generate(
                input_ids=inputs['input_ids'][0].tolist(),
                sampling_params={'max_new_tokens': max_tokens, 'temperature': temperature, 'top_p': 0.9},
                **sglang_mm_kwargs(inputs))
            meta = out.get('meta_info') or {}
            return (out['text'], int(meta.get('prompt_tokens', inputs['input_ids'].shape[1])),
                    int(meta.get('completion_tokens', 0)))
        return processor, generate

    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map='auto', max_memory={0: '125GiB'}, local_files_only=True)
    model.eval()
    model.requires_grad_(False)

    def generate(inputs, max_tokens, temperature):
        dev_inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**dev_inputs, max_new_tokens=max_tokens,
                                 do_sample=True, temperature=temperature, top_p=0.9)
        in_tok = int(dev_inputs['input_ids'].shape[1])
        raw = processor.decode(out[0][in_tok:], skip_special_tokens=True)
        out_tok = int(out.shape[1] - in_tok)
        del out, dev_inputs
        return raw, in_tok, out_tok
    return processor, generate


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True, help='MiniMax-H3 text_encoder directory')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=30050)
    p.add_argument('--backend', choices=('hf', 'sglang'), default='hf',
                   help='hf: Hugging Face generate (default). sglang: SGLang Engine with exact processor tensors')
    p.add_argument('--tp-size', type=int, default=1, help='tensor parallel size for --backend sglang')
    p.add_argument('--mem-fraction-static', type=float, default=0.85,
                   help='KV cache and weight memory fraction for --backend sglang')
    a = p.parse_args()
    print(json.dumps({'status': 'loading', 'backend': a.backend, 'model': str(Path(a.model).resolve())}), flush=True)
    processor, generate = load_backend(a.backend, a.model, tp_size=a.tp_size,
                                       mem_fraction_static=a.mem_fraction_static)
    lock = threading.Lock()
    metrics = {'generation_calls': 0, 'backend': a.backend}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {'status': 'ready', 'model': str(Path(a.model).resolve()), 'busy': lock.locked(), **metrics})

        def do_POST(self):
            if self.path != '/generate':
                return self.reply(404, {'error': 'unknown endpoint'})
            try:
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    started = time.time()
                    inputs, audit = prepare_inputs(processor, req)
                    metrics['generation_calls'] += 1
                    max_tokens = min(int(req.get('max_tokens', 2400)), 8000)
                    temperature = float(req.get('temperature', 0.3))
                    raw, in_tok, out_tok = generate(inputs, max_tokens, temperature)
                    result = {'text': raw, 'seconds': round(time.time() - started, 3),
                              'input_tokens': in_tok, 'output_tokens': out_tok, 'input_audit': audit}
                    del inputs
                print(json.dumps({'role': req.get('role'), **{k: v for k, v in result.items() if k != 'text'}}), flush=True)
                self.reply(200, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.reply(500, {'error': f'{type(e).__name__}: {e}'})
    print(json.dumps({'status': 'ready', 'backend': a.backend, 'host': a.host, 'port': a.port}), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
