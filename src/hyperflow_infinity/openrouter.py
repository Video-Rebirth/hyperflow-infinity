"""OpenRouter transport for the Qwen roles, as an alternative to the local service.

`prepare` turns a journaled request into a chat-completions payload: the previous segment goes
in as native video, or as timestamped frames for models that only take images. `complete` sends
it exactly once. The API key is read from `OPENROUTER_API_KEY` and never written to the run
directory; media is embedded in the payload and never journaled.
"""
import base64
import hashlib
import io
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
# What each supported model can watch. `frames` models get sampled stills instead of the video.
MODEL_VIDEO_MODES = {
    'qwen/qwen3-vl-32b-instruct': 'frames',
    'qwen/qwen3.8-flash': 'native',
    'qwen/qwen3.5-flash-02-23': 'native',
}
FRAMES_PREAMBLE = ('The following images are chronological samples of one previous video. Timestamps are '
                   'relative to its start. Infer only visible events; no audio is provided.')


def enabled(config):
    return config.get('vlm_provider', 'local') == 'openrouter'


def validate_config(config, credentials=False):
    provider = config.get('vlm_provider', 'local')
    if provider not in ('local', 'openrouter'):
        raise ValueError('vlm_provider must be local or openrouter')
    if provider == 'local':
        return
    if config.get('openrouter_model') not in MODEL_VIDEO_MODES:
        raise ValueError('openrouter_model must be one of: ' + ', '.join(MODEL_VIDEO_MODES))
    if config.get('openrouter_video_mode', 'auto') not in ('auto', 'native', 'frames'):
        raise ValueError('openrouter_video_mode must be auto, native or frames')
    if config.get('openrouter_video_mode') == 'native' and MODEL_VIDEO_MODES[config['openrouter_model']] == 'frames':
        raise ValueError(f"{config['openrouter_model']} accepts images, not native video")
    if credentials and not os.environ.get('OPENROUTER_API_KEY', '').strip():
        raise ValueError('Set OPENROUTER_API_KEY in the environment of `launch`')


def video_mode(config):
    mode = config.get('openrouter_video_mode', 'auto')
    return MODEL_VIDEO_MODES[config['openrouter_model']] if mode == 'auto' else mode


def data_url(mime, data):
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def sampled_frames(path, fps=2.0):
    """Sample a video the way Qwen3-VL's own processor would, as timestamped PNGs.

    Frame count and size follow `qwen_vl_utils` (QwenLM/Qwen3-VL, patch size 16). This is a
    fallback for models that take images: it cannot reproduce native video temporal tokens.
    """
    import av
    import torch
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_nframes, smart_resize
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        # Count decoded frames; container frame counts are not always reliable.
        total = sum(1 for _ in container.decode(stream))
    count = smart_nframes({'fps': fps}, total, rate)
    indices = torch.linspace(0, total - 1, count).round().long().tolist()
    chosen = set(indices)
    parts, frames = [], []
    factor = 32
    minimum = 128 * factor ** 2
    maximum = max(min(768 * factor ** 2, 128000 * factor ** 2 * .9 / count * 2), int(minimum * 1.05))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        for index, frame in enumerate(container.decode(stream)):
            if index not in chosen:
                continue
            image = frame.to_image().convert('RGB')
            height, width = smart_resize(image.height, image.width, factor=factor,
                                         min_pixels=minimum, max_pixels=maximum)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            encoded = buffer.getvalue()
            timestamp = float(frame.time) - start if frame.time is not None else index / rate
            parts.extend([{'type': 'text', 'text': f'Frame {index}, time {timestamp:.3f} seconds:'},
                          {'type': 'image_url', 'image_url': {'url': data_url('image/png', encoded)}}])
            frames.append(dict(index=index, seconds=timestamp, width=width, height=height,
                               sha256=hashlib.sha256(encoded).hexdigest()))
    if len(frames) != count:
        raise ValueError('Video changed or failed to decode during frame sampling')
    return parts, dict(protocol='qwen3_vl_uniform_frames_v1', requested_fps=fps,
                       source_fps=rate, decoded_frames=total, sampled_frames=frames)


def prepare(config, request):
    """Build the payload and its journal entry. Runs before anything is sent, so it is safe to repeat.

    Media comes first and the user text last, the same order the local service uses, because the
    prompts say "above is the segment".
    """
    validate_config(config, credentials=True)
    mode = video_mode(config)
    content, media = [], []
    for image in request.get('images', []):
        path = Path(image['path'])
        data = path.read_bytes()
        mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
        content.extend([{'type': 'text', 'text': image.get('label', '')},
                        {'type': 'image_url', 'image_url': {'url': data_url(mime, data)}}])
        media.append(dict(image, kind='image', sha256=hashlib.sha256(data).hexdigest()))
    for video in request.get('videos', []):
        path = Path(video['path'])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != video['sha256']:
            raise ValueError(f'Video changed since the request was built: {path}')
        entry = dict(video, kind='video', mode=mode)
        content.append({'type': 'text', 'text': video.get('label', '')})
        if mode == 'native':
            if path.suffix.lower() != '.mp4':
                raise ValueError('Native video transport requires MP4')
            content.append({'type': 'video_url', 'video_url': {'url': data_url('video/mp4', data)}})
        else:
            content.append({'type': 'text', 'text': FRAMES_PREAMBLE})
            parts, metadata = sampled_frames(path, config.get('openrouter_sample_fps', 2.0))
            content.extend(parts)
            entry.update(metadata)
        media.append(entry)
    content.append({'type': 'text', 'text': request['user']})
    payload = dict(model=config['openrouter_model'],
                   messages=[{'role': 'system', 'content': request['system']},
                             {'role': 'user', 'content': content}],
                   max_tokens=request['max_tokens'], temperature=request['temperature'], stream=False,
                   response_format={'type': 'json_object'}, provider={'allow_fallbacks': False})
    # Hidden reasoning would spend the completion budget before the JSON starts.
    if config['openrouter_model'] != 'qwen/qwen3-vl-32b-instruct':
        payload['reasoning'] = {'enabled': False}
    audit = dict(provider='openrouter', model=config['openrouter_model'], endpoint=ENDPOINT, video_mode=mode,
                 media=media, payload_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    return payload, audit


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('OpenRouter redirect refused; the API key stays on the configured origin')


def never_sent(error):
    """True when a URLError means no connection was made: DNS failure or refused connection.

    Such a request is safe to send again. A timeout or a reset is not: the server may have it.
    """
    return isinstance(error, urllib.error.URLError) and isinstance(error.reason, (socket.gaierror, ConnectionRefusedError))


def complete(payload):
    """One HTTP submission. Returns the service-style `{text, ...}` dict, or `{transport_error, ...}`.

    Response bodies of failed requests are not stored; they can echo the prompt or the key.
    """
    started = time.monotonic()
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        return dict(text='', provider='openrouter', http_status=error.code,
                    transport_error=f'OpenRouter HTTP {error.code}', seconds=time.monotonic() - started)
    except urllib.error.URLError as error:
        if not never_sent(error):
            raise
        return dict(text='', provider='openrouter', transport_error=f'OpenRouter unreachable: {error.reason}',
                    seconds=time.monotonic() - started)
    if result.get('error'):
        return dict(text='', provider='openrouter', error_code=result['error'].get('code'),
                    transport_error='OpenRouter returned an API error', seconds=time.monotonic() - started)
    choice = (result.get('choices') or [{}])[0]
    text = choice.get('message', {}).get('content') or ''
    if not isinstance(text, str):
        text = ''.join(item.get('text', '') for item in text if item.get('type') == 'text')
    usage = result.get('usage') or {}
    # Same token fields as the local service, so `summarize_run.py` reads both.
    return dict(text=text, provider='openrouter', model=result.get('model'), id=result.get('id'),
                upstream_provider=result.get('provider'), usage=usage,
                input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'),
                finish_reason=choice.get('finish_reason'), seconds=time.monotonic() - started)
