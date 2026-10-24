"""H3 + HyperFlow generator. Every torchrun rank executes the same request; only rank 0 writes files."""
import gc
import os
import time
from pathlib import Path
from types import SimpleNamespace

from .store import atomic_json, digest, require, sha256_file


def load_reference(item):
    """A still as an image reference; a video as a native video reference at its own frame rate, without its sound."""
    import dataclasses

    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    from PIL import Image
    if item['kind'] == 'image':
        return MiniMaxH3ImageReference(image=Image.open(item['path']).convert('RGB'))
    ref = MiniMaxH3VideoReference.from_file(item['path'])
    require(len(ref.frames) == item['frames'], 'Opening video decodes to a different frame count than at init')
    return dataclasses.replace(ref, audio=None, sample_rate=None)


class HyperFlow:
    def __init__(self, config):
        import torch
        import torch.distributed as dist

        from . import handoff
        self.config = config
        self.rank = int(os.environ.get('RANK', '0'))
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(self.local_rank)
        if self.world > 1 and not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group('nccl', device_id=torch.device(f'cuda:{self.local_rank}'), timeout=timedelta(hours=2))
        # Workers wait on a gloo broadcast so idle GPUs do not spin NCCL kernels during Qwen calls.
        self.control_group = dist.new_group(backend='gloo') if self.world > 1 else None
        self.pipe = None
        self.workflow = None
        self.in_generation = False
        handoff.install()

    def broadcast(self, value):
        import torch.distributed as dist
        if self.world == 1:
            return value
        box = [value if self.rank == 0 else None]
        dist.broadcast_object_list(box, src=0, group=self.control_group)
        return box[0]

    def _build(self, workflow):
        """Build the HyperFlow pipeline for `workflow` once. Returns seconds spent."""
        import torch
        from hyperflow_h3.examples.common import build_pipeline
        if self.pipe is not None:
            require(self.workflow == workflow, f'The pipeline is built for {self.workflow}, not {workflow}')
            return 0.0
        started = time.time()
        weights = Path(self.config['hyperflow_weights'])
        attention_backend = self.config.get('attention_backend', 'sdpa')
        args = SimpleNamespace(
            model=self.config['h3_dir'], weights=str(weights.parent),
            weights_filename=weights.name, baseline=False, attention_backend=attention_backend,
            sol_attn=False, dense_steps=None, dense_layers=None, sol_tau=None, gpus=self.world,
            device=f'cuda:{self.local_rank}', no_offload=False, memory_reserve_margin='24GB',
            verbose=False, num_inference_steps=None)
        self.pipe = build_pipeline(args, workflow=workflow)

        # Keep the 480p canvas for the identity image instead of upscaling it to the default 2048p.
        # Reapplied whenever the split conditioner is reloaded.
        def configure(p):
            if p is not None:
                p.register_to_config(reference_image_short_edge=480, canvas_short_edge=480,
                                     canvas_max_pixels=self.config['width'] * self.config['height'])
        configure(getattr(self.pipe, 'conditioner', self.pipe))
        if getattr(self.pipe, '_conditioner_factory', None):
            factory = self.pipe._conditioner_factory

            def configured_factory():
                p, m = factory()
                configure(p)
                return p, m
            self.pipe._conditioner_factory = configured_factory
        if self.rank == 0 and self.config.get('conditioning_gpu') is not None:
            # The launcher appends the conditioning GPU after the generation GPUs, so it is cuda:<world>.
            require(self.world > 1, 'A dedicated conditioner GPU requires the split multi-GPU pipeline')
            device = f'cuda:{self.world}'
            require(torch.cuda.device_count() > self.world, 'Dedicated conditioner GPU is not visible')
            conditioner = self.pipe.conditioner
            if self.pipe._conditioner_manager is not None:
                self.pipe._conditioner_manager.disable_auto_cpu_offload()
            conditioner.to(device)
            self.pipe._resident_conditioner = conditioner
            # Upstream drops and reloads the encoder between requests; keep it resident instead.
            from types import MethodType

            def restore_conditioner(split):
                split.conditioner = split._resident_conditioner
                split._conditioner_manager = None
            self.pipe._reload_conditioner = MethodType(restore_conditioner, self.pipe)
            print(f"H3 conditioner resident on physical GPU {self.config['conditioning_gpu']} ({device})", flush=True)
        self.workflow = workflow
        return time.time() - started

    def make_image(self, spec, path):
        """Rank 0, before any H3 pipeline exists: draw the opening still with Qwen-Image, then free it."""
        import torch
        from diffusers import QwenImagePipeline
        require(self.pipe is None, 'Qwen-Image runs before H3 is loaded')
        device = f'cuda:{self.world}' if self.config.get('conditioning_gpu') is not None else f'cuda:{self.local_rank}'
        started = time.time()
        pipe = QwenImagePipeline.from_pretrained(spec['model_dir'], torch_dtype=torch.bfloat16).to(device)
        loaded = time.time()
        image = pipe(prompt=spec['prompt'], negative_prompt=spec['negative_prompt'], width=spec['width'],
                     height=spec['height'], num_inference_steps=spec['steps'], true_cfg_scale=spec['true_cfg_scale'],
                     generator=torch.Generator().manual_seed(spec['seed'])).images[0]
        image.save(path)
        timing = {'load_seconds': round(loaded - started, 3), 'generate_seconds': round(time.time() - loaded, 3),
                  'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
        del pipe, image
        gc.collect()
        torch.cuda.empty_cache()
        return timing

    def generate(self, request):
        self.in_generation = True
        if self.rank == 0:
            self.broadcast({'op': 'generate', 'request': request})
        result = self._execute(request)
        self.in_generation = False
        return result

    def _execute(self, request):
        import wave

        import torch
        from diffusers.utils.export_utils import encode_video

        from . import handoff
        from .media import probe
        mode = request['mode']
        opening = request['generation_contract'] == handoff.OPENING_CONTRACT
        require(opening or request['generation_contract'] == handoff.CONTRACT, 'Unknown generation contract')
        handoff.set_boundary(request.get('boundary_latents'), mode, identity_anchor=request.get('identity_anchor'),
                             opening=request.get('opening_reference'))
        build_seconds = self._build(mode)
        extra = {}
        if self.rank == 0:
            media = request['media']
            require(len(media) == 1, 'Exactly one reference per request')
            expected = request['opening_reference' if opening else 'identity_anchor']['sha256']
            require(sha256_file(media[0]['path']) == expected == media[0]['sha256'], 'Reference SHA mismatch before encoding')
            require(opening or media[0]['kind'] == 'image', 'Continuations take one identity image')
            extra = {'references': [load_reference(media[0])]}
        conditioner_device = (f'cuda:{self.world}' if self.rank == 0 and self.config.get('conditioning_gpu') is not None
                              else None)
        if conditioner_device:
            torch.cuda.reset_peak_memory_stats(conditioner_device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        result = self.pipe(prompt=request['prompt'], height=request['height'], width=request['width'],
                           num_frames=request['generated_frames'],
                           generator=torch.Generator().manual_seed(request['seed']),
                           output=['videos', 'audio', 'sampling_rate', 'latents', 'audio_latents'], **extra)
        torch.cuda.synchronize()
        pipeline_seconds = time.time() - started
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        peaks = [peak_gib]
        if self.world > 1:
            import torch.distributed as dist
            peaks = [None] * self.world
            dist.all_gather_object(peaks, peak_gib, group=self.control_group)
        output = None
        if self.rank == 0:
            root = Path(request['chunk_dir'])
            root.mkdir(parents=True, exist_ok=True)
            raw, wav, latent = root / 'raw.mp4', root / 'raw.wav', root / 'latents.pt'
            frames = result['videos'][0]
            audio = result['audio']
            if audio.ndim == 3:
                audio = audio[0]
            rate = int(result['sampling_rate'])
            desired = round(len(frames) / 24 * rate)
            audio = audio[..., :desired]
            if audio.shape[-1] < desired:
                audio = torch.nn.functional.pad(audio, (0, desired - audio.shape[-1]))
            encode_video(frames, fps=24, output_path=str(raw), audio=audio, audio_sample_rate=rate)
            pcm = (audio.detach().float().cpu().clamp(-1, 1).numpy().T * 32767).astype('<i2')
            with wave.open(str(wav), 'wb') as f:
                f.setnchannels(pcm.shape[1])
                f.setsampwidth(2)
                f.setframerate(rate)
                f.writeframes(pcm.tobytes())
            torch.save({'video': result['latents'].detach().float().cpu(),
                        'audio': result['audio_latents'].detach().float().cpu()}, latent)
            info = probe(raw)
            require(info['frames'] == request['generated_frames'], 'H3 returned unexpected frame count')
            require(handoff.audit() is not None, 'Ref2VA conditioning hooks did not execute')
            output = {'raw_video': str(raw.resolve()), 'raw_audio': str(wav.resolve()),
                      'latents': str(latent.resolve()), 'actual_frames': info['frames'],
                      'conditioning_audit': handoff.audit(), 'media_probe': info,
                      'execution_devices': {key: self.config.get(key) for key in ('generation_gpus', 'conditioning_gpu')},
                      'reference_inputs': [{k: m.get(k) for k in ('id', 'kind', 'path', 'sha256')} for m in request['media']],
                      'timing': {'pipeline_build_seconds': round(build_seconds, 3),
                                 'pipeline_seconds': round(pipeline_seconds, 3),
                                 'peak_allocated_gib_per_rank': [round(p, 2) for p in peaks],
                                 'conditioner_peak_allocated_gib': round(
                                     torch.cuda.max_memory_allocated(conditioner_device) / 2**30, 2)
                                     if conditioner_device else None}}
            atomic_json(root / 'generation_result.json', {'request_sha256': digest(request), 'result': output})
        del result, extra
        gc.collect()
        torch.cuda.empty_cache()
        return output

    def worker_loop(self):
        while True:
            task = self.broadcast(None)
            if task['op'] == 'stop':
                break
            self.in_generation = True
            self._execute(task['request'])
            self.in_generation = False

    def stop(self):
        if self.rank == 0:
            self.broadcast({'op': 'stop'})
